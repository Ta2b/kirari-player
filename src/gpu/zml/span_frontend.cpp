#include "gpu/zml/span_frontend.h"

#include <algorithm>

#include "core/utf8_path.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/span_fp8/onnx_weights.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/zml/sdk_process.h"
#ifdef KIRARI_ZML_SPAN_TRITON_GATES
#include "gpu/zml/span_gate_artifact.h"
#include "gpu/zml/span_gate_profile.h"
#endif
#ifdef KIRARI_ZML_SPAN_STRIPS
#include "gpu/zml/hlo.h"
#include "gpu/zml/span_strips.h"
#endif

namespace kirari::gpu::zml {
namespace {
bool exact_ends(const ir::onnx::Model& onnx, const vk::net::InputShape& shape) {
  // Le dégroupage et les deux découpes terminales dépendent de la parité de H et W. Vérifier
  // le trajet jusqu'au premier filtre après spécialisation, y compris un éventuel Pad.
  auto g = ir::import_model(onnx);
  const ir::InputShape input{shape.name, shape.dims};
  if (!g || !ir::infer_shapes(*g, std::span(&input, 1)) || !ir::fold_constants(*g)) return false;
  ir::eliminate_dead_code(*g);
  ir::eliminate_identities(*g);
  ir::fuse_pixel_shuffles(*g);
  ir::eliminate_dead_code(*g);
  if (g->inputs.size() != 1 || g->outputs.size() != 1 ||
      g->tensors[g->outputs.front()].dims != ir::Dims{1, 3, 2 * shape.dims[2], 2 * shape.dims[3]})
    return false;
  const auto conv_weight = ir::find_tensor(*g, "conv_1.1.eval_conv.weight");
  if (!conv_weight) return false;
  ir::GraphIndex index(*g);
  const auto first = index.sole_reader(*conv_weight);
  if (!first) return false;
  const auto at = index.producer(g->nodes[*first].inputs[0]);
  const auto end = index.producer(g->outputs.front());
  if (at == ir::kNoNode || end == ir::kNoNode) return false;
  const auto& unpack = g->nodes[at];
  const auto& pack = g->nodes[end];
  if (unpack.op != ir::OpKind::SpaceToDepth || unpack.inputs[0] != g->inputs.front() ||
      pack.op != ir::OpKind::DepthToSpace)
    return false;
  const auto a = std::get<ir::BlockAttrs>(unpack.attrs), b = std::get<ir::BlockAttrs>(pack.attrs);
  return a.blocksize == 2 && b.blocksize == 4 && a.mode == ir::BlockMode::Crd && b.mode == ir::BlockMode::Crd;
}
bool span_image(const ir::Dims& d) {
  return d.size() == 4 && d[0] == 1 && d[1] == 3 && d[2] > 0 && d[3] > 0 && d[2] <= span_fp8::kMaxSide &&
         d[3] <= span_fp8::kMaxSide && d[2] % 2 == 0 && d[3] % 2 == 0;
}

bool fp16_graph(const ir::onnx::Graph& graph) {
  return graph.inputs.size() == 1 && graph.outputs.size() == 1 &&
         graph.inputs.front().elem_type == ir::onnx::kFloat16 &&
         graph.outputs.front().elem_type == ir::onnx::kFloat16 &&
         std::ranges::none_of(graph.initializers, [](const auto& t) {
           return !t.floats.empty() && t.data_type != ir::onnx::kFloat16;
         });
}

struct SpanEmission {
  const std::filesystem::path& weight_file;
  const std::filesystem::path& staged;
  std::int64_t width;
  std::size_t blocks;
  int channels;
  std::string_view model_hash;
  SpanSdk sdk;
  Progress& progress;

  std::expected<bool, Refusal> emit_sdk(std::int64_t height) const {
    std::vector<std::string> arguments{core::to_utf8(weight_file), std::to_string(height),
                                       std::to_string(width), core::to_utf8(staged)};
#ifdef KIRARI_ZML_SPAN_ATTENTION_DOT
    arguments.emplace_back("--attention-dot");
#endif
#ifdef KIRARI_ZML_SPAN_PROJECTION_DOT
    arguments.emplace_back("--projection-dot");
#elif defined(KIRARI_ZML_SPAN_PROJECTION_STRIPS)
    if (height == 1080 && width == 1920) arguments.emplace_back("--projection-strips");
#else
    if (auto option = sdk.projection_option(channels, blocks, height, width); !option.empty())
      arguments.emplace_back(option);
#endif
#ifdef KIRARI_ZML_SPAN_TRITON_GATES
    if (height == 1080 && width == 1920) {
      for (const auto& profile : kSpanGateProfiles) {
        const bool enabled = profile.channels == 64 ? sdk.balanced_gates : sdk.performance_gates;
        if (enabled && profile.channels == channels && profile.blocks == blocks &&
            std::ranges::contains(profile.weights, model_hash))
          return emit_span_gates(std::move(arguments), staged, profile,
                                 sdk.fused_projection && channels == 64, progress);
      }
    }
#endif
    return run_sdk_frontend(std::move(arguments), progress);
  }

#ifdef KIRARI_ZML_SPAN_STRIPS
  std::expected<bool, Refusal> build_strips(const ModelSpec& model, const ir::onnx::Model& onnx,
                                            const vk::net::InputShape& input) const {
    const auto plan = span_strips_plan(input.dims, blocks);
    if (!plan) return false;
    auto shape = input;
    shape.dims[2] = plan->window;
    if (!exact_ends(onnx, shape)) return false;
    std::optional<std::string> tile;
    if (sdk.available) {
      if (auto built = emit_sdk(plan->window); built && *built) {
        auto module = vk::net::read_file(staged);
        if (!module) return std::unexpected(module.error());
        tile.emplace(reinterpret_cast<const char*>(module->data()), module->size());
      } else if (!built && built.error().code != RefusalCode::BuilderFailed) {
        return std::unexpected(built.error());
      }
    } else {
      const ir::InputShape tile_input{shape.name, shape.dims};
      auto module = lower_onnx(model.onnx_path, std::span(&tile_input, 1), "f16");
      if (module) tile = std::move(*module);
    }
    if (progress.stop.stop_requested())
      return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation SPAN ZML annulée"});
    if (tile) {
      if (auto module = wrap_span_strips(std::move(*tile), *plan)) {
        if (auto saved = vk::net::write_file(staged, std::as_bytes(std::span(*module))); !saved)
          return std::unexpected(saved.error());
        return true;
      }
    }
    return false;
  }
#endif
};
}  // namespace

std::expected<bool, Refusal> build_recognized_span(const ModelSpec& model,
                                                   std::span<const vk::net::InputShape> inputs,
                                                   const std::filesystem::path& staged, SpanSdk sdk,
                                                   Progress& progress) {
  const bool use_sdk = sdk.available;
#ifndef KIRARI_ZML_SPAN_STRIPS
  if (!use_sdk) return false;
#endif
  if (model.precision != TensorType::F16 || inputs.size() != 1) return false;
  const auto& d = inputs[0].dims;
  if (!span_image(d)) return false;
  std::error_code ec;
  const auto size = std::filesystem::file_size(model.onnx_path, ec);
  if (ec || size > span_fp8::kMaxPackageBytes) return false;
  auto bytes = vk::net::read_file(model.onnx_path);
  if (!bytes) return std::unexpected(bytes.error());
  auto onnx = ir::onnx::read_onnx(*bytes);
  if (!onnx) return std::unexpected(onnx.error());
  if (!fp16_graph(onnx->graph)) return false;
  auto package = span_fp8::package_from_onnx(*onnx);
  if (!package ||
      !((package->real_channels == 64 && package->blocks.size() == 8) ||
        (package->real_channels == 48 && package->blocks.size() == 5)) ||
      !exact_ends(*onnx, inputs[0]))
    return false;
  if (progress.stop.stop_requested())
    return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation SPAN ZML annulée"});
  auto weight_file = staged;
  weight_file += ".poids";
  if (use_sdk) {
    package->fp16 = true;
    if (auto saved = vk::net::write_file(weight_file, span_fp8::serialize(*package)); !saved)
      return std::unexpected(saved.error());
    if (progress.report) progress.report(0.2F, "compilation du graphe SPAN avec le frontal Zig de ZML");
  }
  const SpanEmission emission{weight_file,  staged, d[3],    package->blocks.size(), package->real_channels,
                              model.sha256, sdk,    progress};
#ifdef KIRARI_ZML_SPAN_STRIPS
  auto strips = emission.build_strips(model, *onnx, inputs[0]);
  if (!strips || *strips) return strips;
#endif
  if (!use_sdk) return false;
  return emission.emit_sdk(d[2]);
}
}  // namespace kirari::gpu::zml
