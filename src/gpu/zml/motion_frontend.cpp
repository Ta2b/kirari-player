#include "gpu/zml/motion_frontend.h"

#include <algorithm>
#include <bit>
#include <format>
#include <optional>
#include <sstream>
#include <utility>

#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/ops.h"
#include "gpu/ir/passes.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/zml/artcnn_body.h"
#include "gpu/zml/drba_graph.h"
#include "gpu/zml/drba_output_warp.h"
#include "gpu/zml/drba_resize.h"
#include "gpu/zml/gmflow_global.h"
#include "gpu/zml/gmfss_graph.h"
#include "gpu/zml/gmfss_mlp.h"
#include "gpu/zml/rife_conv.h"
#include "gpu/zml/rife_sampling_plan.h"
#include "gpu/zml/sdk_process.h"
#include "kernels/ref/half.h"

namespace kirari::gpu::zml {
namespace {
using Op = ir::OpKind;
enum class Family { Motion, Upscale, Temporal, Gmfss };

std::string_view family_name(Family family) {
  switch (family) {
    case Family::Motion:
      return "motion";
    case Family::Upscale:
      return "upscale";
    case Family::Temporal:
      return "temporal";
    case Family::Gmfss:
      return "gmfss";
  }
  std::unreachable();
}

bool temporal_fix(const ir::Graph& graph) {
  if (graph.inputs.empty() || graph.outputs.size() != 1) return false;
  const auto& input = graph.tensors[graph.inputs.front()];
  if (input.dims.size() != 4) return false;
  auto weight = [&](std::string_view name) {
    const auto id = ir::find_tensor(graph, name);
    return id && graph.tensors[*id].kind == ir::TensorKind::Constant;
  };
  if (graph.inputs.size() == 1 && input.dims[1] == 3)
    return weight("model.encode.cnn0.weight") && weight("model.encode.proj.weight");
  // Sept RVB empilés, séparés, ou accompagnés des sept traits du cache de la chaîne.
  const bool window = (graph.inputs.size() == 1 && input.dims[1] == 21) ||
                      ((graph.inputs.size() == 7 || graph.inputs.size() == 14) && input.dims[1] == 3);
  return window && weight("model.aligner.blocks.0.conv0.0.0.weight") &&
         weight("model.aligner.blocks.0.convblock.0.conv.weight");
}

std::string json_string(std::string_view value) {
  std::string result = "\"";
  for (const char c : value) {
    if (c == '"' || c == '\\') result += '\\';
    const auto byte = static_cast<unsigned char>(c);
    if (byte < 0x20)
      result += std::format("\\u{:04x}", byte);
    else
      result += c;
  }
  return result + '"';
}

std::string source_name(const ir::Graph& graph, const ir::Node& node) {
  if (!node.name.empty()) return node.name;
  const auto id = node.outputs.front();
  const auto& tensor = graph.tensors[id];
  if (!tensor.name.empty()) return std::format("{}/{}", ir::name(node.op), tensor.name);
  return std::format("{}/v{}", ir::name(node.op), id);
}

template <typename Range>
std::string array_json(const Range& range) {
  std::string result = "[";
  for (auto v : range) {
    if (result.size() > 1) result += ',';
    result += std::format("{}", v);
  }
  return result + ']';
}

// Seules les métadonnées sont du JSON. Le tableau d'octets contigu garde en binaire les
// constantes et les fragments StableHLO de la bibliothèque d'opérations déjà partagée.
class Manifest {
 public:
  std::expected<std::vector<std::byte>, Refusal> encode(ir::Graph& graph, bool fp16, Family family,
                                                        RifeRecipes recipes, DrbaRecipes drba,
                                                        GmfssRecipes gmfss) {
    const bool motion = family == Family::Motion;
    std::vector<MotionGroup> groups;
    if (gmfss.mlp_strips && fp16) {
      if (auto mlp = gmflow_quarter_mlp(graph)) groups.push_back(std::move(*mlp));
    }
    if (gmfss.global_match && fp16) {
      if (auto match = gmflow_global_match(graph)) groups.push_back(std::move(*match));
    }
    MotionRecipe phases;
    if (family == Family::Motion || family == Family::Upscale) {
      groups = motion_groups(graph);
      phases = configure_motion_groups(graph, groups, fp16 ? recipes : RifeRecipes{});
    }
#ifdef KIRARI_ZML_FOLD_RESCONV
    if (fp16 && motion) fold_motion_residuals(graph, groups);
#endif
    std::ostringstream json;
    const auto name = family_name(family);
    json << "{\"family\":\"" << name << "\",\"io\":\"" << (fp16 ? "f16" : "f32")
         << "\",\"inputs\":" << array_json(graph.inputs) << ",\"outputs\":" << array_json(graph.outputs)
         << ",\"tensors\":[";
    write_tensors(json, graph);
    json << "],\"nodes\":[";
    const auto hints = configure_sampling(graph, groups, phases, motion, fp16, recipes, drba);
    if (auto result = write_nodes(json, graph, groups, phases, hints, family); !result)
      return std::unexpected(result.error());
    json << ']';
    if (hints.output_pixel) json << ",\"drba_output_pixel\":true";
    json << '}';
    auto content = std::move(json).str();
    auto body = std::move(bytes_);
    bytes_.clear();
    constexpr std::string_view magic = "KZMLIR01";
    const auto tag = std::as_bytes(std::span(magic));
    const auto desc = std::as_bytes(std::span(content));
    bytes_.reserve(16 + desc.size() + body.size());
    bytes_.insert(bytes_.end(), tag.begin(), tag.end());
    append_le(static_cast<std::uint64_t>(content.size()));
    bytes_.insert(bytes_.end(), desc.begin(), desc.end());
    bytes_.insert(bytes_.end(), body.begin(), body.end());
    return std::move(bytes_);
  }

 private:
  std::vector<std::byte> bytes_;

  template <typename UInt>
  void append_le(UInt bits) {
    for (std::size_t i = 0; i < sizeof(bits); ++i) bytes_.push_back(static_cast<std::byte>(bits >> (8 * i)));
  }

  struct MotionHints {
    bool output_pixel = false;
    ir::TensorId half_warp = ir::kNoTensor;
    ir::TensorId projection = ir::kNoTensor;
    ir::TensorId central_warp = ir::kNoTensor;
    ir::TensorId central_pixels = ir::kNoTensor;
  };

  static MotionHints configure_sampling(const ir::Graph& graph, std::vector<MotionGroup>& groups,
                                        MotionRecipe phases, bool motion, [[maybe_unused]] bool fp16,
                                        [[maybe_unused]] RifeRecipes recipes,
                                        [[maybe_unused]] DrbaRecipes drba) {
    MotionHints hints;
#ifdef KIRARI_ZML_RIFE422_WARP_XY
    hints.half_warp = fp16 && recipes.lite422 ? rife422_half_warp(graph, groups) : ir::kNoTensor;
#endif
    if (fp16 && motion && recipes.lite425_d4) {
      if (auto d4 = rife425_lite_d4(graph, groups)) groups.push_back(std::move(*d4));
    }
    if (phases.d2_centre && motion) {
      if (auto centre = rife_d2_centre(graph, groups, phases.resize)) groups.push_back(std::move(*centre));
    }
#if defined(KIRARI_ZML_DRBA_OUTPUT_WARP) || defined(KIRARI_ZML_DRBA_SHARED_PROJECTION) || \
    defined(KIRARI_ZML_DRBA_GATHER_FP16) || defined(KIRARI_ZML_DRBA_OUTPUT_PIXEL)
    auto finish = motion ? drba_output_warp(graph) : std::nullopt;
    if (fp16 && finish && drba.output_pixel)
      hints.output_pixel = graph.tensors[finish->outputs.front()].dims == ir::Dims{1, 3, 1088, 1920};
    if (fp16 && finish && (drba.shared_projection || drba.gather_fp16)) {
      const auto projection = drba_shared_projection(graph, *finish);
      if (drba.shared_projection) hints.projection = projection;
      if (drba.gather_fp16 && projection != ir::kNoTensor) identify_central_gather(graph, *finish, hints);
    }
#ifdef KIRARI_ZML_DRBA_OUTPUT_WARP
    if (finish) groups.push_back(std::move(*finish));
#endif
#endif
    return hints;
  }

  // La reconnaissance de la projection a déjà vérifié les deux paires [1,7,768,1280] FP16,
  // les tranches RGB adjacentes, le masque, les deux grilles et les consommateurs du résultat.
  static void identify_central_gather(const ir::Graph& graph, const MotionGroup& finish, MotionHints& hints) {
    ir::GraphIndex index(graph);
    const auto pixels = finish.inputs[1];  // tranche [3:6] de pair01 avant son Cast FP32
    for (auto k : finish.members) {
      const auto& warp = graph.nodes[k];
      if (warp.name != "/GridSample_8" || warp.op != Op::GridSample) continue;
      const auto producer = index.producer(warp.inputs[0]);
      if (producer == ir::kNoNode) return;
      const auto& cast = graph.nodes[producer];
      if (cast.op != Op::Cast || cast.inputs.size() != 1 || cast.inputs.front() != pixels ||
          std::get<ir::CastAttrs>(cast.attrs).to != ir::DataType::F32)
        return;
      hints.central_warp = warp.outputs.front();
      hints.central_pixels = pixels;
      return;
    }
  }

  void write_tensors(std::ostream& json, const ir::Graph& graph) {
    for (std::size_t k = 0; k < graph.tensors.size(); ++k) {
      if (k) json << ',';
      const auto& t = graph.tensors[k];
      json << "{\"dtype\":\"" << element_type(t.type) << "\",\"dims\":" << array_json(t.dims);
      if (t.kind == ir::TensorKind::Constant) {
        const auto start = bytes_.size();
        if (t.type == ir::DataType::F16) {
          for (float v : t.value.f) append_le(kernels::ref::float_to_half(v));
        } else if (t.type == ir::DataType::F32) {
          for (float v : t.value.f) append_le(std::bit_cast<std::uint32_t>(v));
        } else {
          for (auto v : t.value.i) append_le(static_cast<std::uint64_t>(v));
        }
        json << ",\"offset\":" << start << ",\"size\":" << bytes_.size() - start;
      }
      json << '}';
    }
  }

  static void write_group(std::ostream& json, const ir::Graph& graph, const MotionGroup& group) {
    const auto& anchor = graph.nodes[group.members.back()];
    json << "{\"op\":\"" << group.op
         << "\",\"name\":" << json_string(group.op + "/" + source_name(graph, anchor)) << ",\"layers\":[";
    bool first_layer = true;
    for (auto index : group.members) {
      const auto& member = graph.nodes[index];
      if (group.op != "KirariGmflowMlp" && group.op != "KirariGmflowMatch" && member.op != Op::Conv &&
          member.op != Op::ConvTranspose)
        continue;
      if (!first_layer) json << ',';
      first_layer = false;
      json << json_string(source_name(graph, member));
    }
    json << "],\"ins\":" << array_json(group.inputs) << ",\"outs\":" << array_json(group.outputs)
         << ",\"i\":" << array_json(group.integers) << '}';
  }

  std::expected<void, Refusal> write_nodes(std::ostream& json, const ir::Graph& graph,
                                           const std::vector<MotionGroup>& groups, MotionRecipe phases,
                                           [[maybe_unused]] MotionHints hints, Family family) {
    std::vector<const MotionGroup*> membership(graph.nodes.size());
    for (const auto& group : groups)
      for (const auto at : group.members) membership[at] = &group;
    bool first = true;
    for (std::size_t k = 0; k < graph.nodes.size(); ++k) {
      const auto* group = membership[k];
      if (group && group->members.back() != k) continue;
      if (!first) json << ',';
      first = false;
      if (group) {
        write_group(json, graph, *group);
        continue;
      }
      const auto& n = graph.nodes[k];
      json << "{\"op\":\"" << ir::name(n.op) << "\",\"name\":" << json_string(source_name(graph, n))
           << ",\"ins\":" << array_json(n.inputs) << ",\"outs\":" << array_json(n.outputs);
#ifdef KIRARI_ZML_RIFE422_WARP_XY
      if (n.op == Op::GridSample && n.outputs.front() == hints.half_warp) json << ",\"interleaved_xy\":true";
#endif
#ifdef KIRARI_ZML_DRBA_GATHER_FP16
      if (n.op == Op::GridSample && n.outputs.front() == hints.central_warp)
        json << ",\"drba_central_pixels\":" << hints.central_pixels;
#endif
#ifdef KIRARI_ZML_DRBA_SHARED_PROJECTION
      if (n.op == Op::Add && n.outputs.front() == hints.projection)
        json << ",\"drba_shared_projection\":true";
#endif
      if (!(family == Family::Gmfss && n.op == Op::MatMul) && !parameters(json, graph, n, phases)) {
        auto fragment = generic(graph, n);
        if (!fragment) return std::unexpected(fragment.error());
        json << ",\"offset\":" << bytes_.size() << ",\"size\":" << fragment->size();
        auto raw = std::as_bytes(std::span(*fragment));
        bytes_.insert(bytes_.end(), raw.begin(), raw.end());
      }
      json << '}';
    }
    return {};
  }

  static bool convolution_parameters(std::ostream& out, const ir::Graph& g, const ir::Node& n,
                                     MotionRecipe phases) {
    const auto& x = g.tensors[n.inputs[0]].dims;
    const auto& d = g.tensors[n.outputs[0]].dims;
    ir::Pair strides, dilations, output_padding{};
    ir::Quad pads;
    std::int64_t groups;
    if (n.op == Op::Conv) {
      const auto& a = std::get<ir::ConvAttrs>(n.attrs);
      strides = a.strides;
      dilations = a.dilations;
      groups = a.group;
      pads = a.pads;
      if (a.auto_pad == ir::AutoPad::Valid) pads = {};
      if (a.auto_pad == ir::AutoPad::SameLower || a.auto_pad == ir::AutoPad::SameUpper) {
        const auto& w = g.tensors[n.inputs[1]].dims;
        for (std::size_t j = 0; j < 2; ++j) {
          const auto total = std::max<std::int64_t>(
              0, (d[j + 2] - 1) * strides[j] + 1 + (w[j + 2] - 1) * dilations[j] - x[j + 2]);
          pads[j] = a.auto_pad == ir::AutoPad::SameUpper ? total / 2 : total - total / 2;
          pads[j + 2] = total - pads[j];
        }
      }
    } else {
      const auto& a = std::get<ir::ConvTransposeAttrs>(n.attrs);
      if (a.auto_pad != ir::AutoPad::NotSet || !a.output_shape.empty()) return false;
      strides = a.strides;
      dilations = a.dilations;
      groups = a.group;
      pads = a.pads;
      output_padding = a.output_padding;
    }
    out << ",\"i\":"
        << array_json(Dims{strides[0], strides[1], dilations[0], dilations[1], pads[0], pads[1], pads[2],
                           pads[3], output_padding[0], output_padding[1], groups, phases.transposed});
    return true;
  }

  static bool resize_parameters(std::ostream& out, const ir::Graph& g, const ir::Node& n,
                                MotionRecipe phases) {
    const auto& x = g.tensors[n.inputs[0]].dims;
    const auto& d = g.tensors[n.outputs[0]].dims;
    const auto& a = std::get<ir::ResizeAttrs>(n.attrs);
    if (d.size() != 4 || d[0] != x[0] || d[1] != x[1]) return false;
    std::array<float, 4> scales{};
    for (std::size_t i = 0; i < 4; ++i) scales[i] = static_cast<float>(d[i]) / static_cast<float>(x[i]);
    if (const auto* v = ir::constant_operand(g, n, 2); v && !v->f.empty()) {
      auto mapping = a.axes.empty() ? axes(4) : a.axes;
      if (v->f.size() != mapping.size()) return false;
      for (std::size_t i = 0; i < mapping.size(); ++i) {
        const auto axis = mapping[i] < 0 ? mapping[i] + 4 : mapping[i];
        if (axis < 0 || axis > 3) return false;
        scales[static_cast<std::size_t>(axis)] = v->f[i];
      }
    }
    out << ",\"i\":"
        << array_json(
               Dims{a.mode == ir::ResizeMode::Linear, static_cast<int>(a.coord), static_cast<int>(a.nearest)})
        << ",\"f\":" << array_json(std::array{scales[2], scales[3]});
    if (phases.resize) out << ",\"upsample_phases\":true";
#ifdef KIRARI_ZML_DRBA_DIRECT_RESIZE
    if (drba_resize_factor(g, n, scales)) out << ",\"drba_direct_resize\":true";
#endif
    if (auto plan = compact_warp_plan(g, n, scales[2], scales[3])) {
      const auto& attrs = std::get<ir::GridSampleAttrs>(plan->warp.attrs);
      const auto& split = plan->split;
      out << std::boolalpha << ",\"compact\":{\"image\":" << plan->warp.inputs[0]
          << ",\"grid\":" << plan->warp.inputs[1] << ",\"factor\":" << plan->factor
          << ",\"first_channels\":" << g.tensors[split.outputs[0]].dims[1]
          << ",\"round_warp\":" << (plan->cast_warp && g.tensors[split.inputs[0]].type == ir::DataType::F16)
          << ",\"round_planes\":" << (plan->cast_planes && g.tensors[n.inputs[0]].type == ir::DataType::F16)
          << ",\"grid_options\":{\"mode\":\""
          << (attrs.mode == ir::GridMode::Nearest ? "nearest" : "bilinear") << "\",\"padding\":\""
          << (attrs.padding == ir::GridPadding::Border ? "border" : "zeros")
          << "\",\"align_corners\":" << attrs.align_corners << ",\"grid_nchw\":" << attrs.grid_nchw
          << "},\"planes\":" << array_json(plan->join.inputs) << '}' << std::noboolalpha;
    }
    return true;
  }

  static bool no_parameters(Op op) {
    switch (op) {
      case Op::Relu:
      case Op::Sigmoid:
      case Op::Tanh:
      case Op::Sub:
      case Op::Add:
      case Op::Mul:
      case Op::Div:
      case Op::Clip:
      case Op::Cast:
      case Op::Identity:
      case Op::Reshape:
      case Op::Unsqueeze:
      case Op::Squeeze:
      case Op::FakeQuantE4M3:
      case Op::PRelu:
      case Op::Neg:
      case Op::Abs:
      case Op::Expand:
        return true;
      default:
        return false;
    }
  }

  static bool parameters(std::ostream& out, const ir::Graph& g, const ir::Node& n, MotionRecipe phases) {
    const auto& x = g.tensors[n.inputs[0]].dims;
    switch (n.op) {
      case Op::Conv:
      case Op::ConvTranspose:
        return convolution_parameters(out, g, n, phases);
      case Op::DepthToSpace: {
        const auto& a = std::get<ir::BlockAttrs>(n.attrs);
        out << ",\"i\":" << array_json(Dims{a.blocksize, a.mode == ir::BlockMode::Crd});
        break;
      }
      case Op::GridSample: {
        const auto& a = std::get<ir::GridSampleAttrs>(n.attrs);
        out << ",\"i\":"
            << array_json(Dims{a.mode == ir::GridMode::Nearest, a.padding == ir::GridPadding::Border,
                               a.align_corners, a.grid_nchw});
#ifdef KIRARI_ZML_LINEAR_WARP
        if (auto source = linear_warp_source(g, n)) out << ",\"linear_source\":" << *source;
#endif
        break;
      }
      case Op::Resize:
        return resize_parameters(out, g, n, phases);
      case Op::Transpose: {
        auto perm = std::get<ir::TransposeAttrs>(n.attrs).perm;
        if (perm.empty()) {
          perm = axes(x.size());
          std::ranges::reverse(perm);
        }
        out << ",\"i\":" << array_json(perm);
        break;
      }
      case Op::Concat:
        out << ",\"i\":" << array_json(Dims{std::get<ir::AxisAttrs>(n.attrs).axis});
        break;
      case Op::LeakyRelu:
        out << ",\"f\":" << array_json(std::array{std::get<ir::LeakyReluAttrs>(n.attrs).alpha});
        break;
      default:
        return no_parameters(n.op);
    }
    return true;
  }

  static std::expected<std::string, Refusal> generic(const ir::Graph& graph, const ir::Node& node) {
    Hlo hlo;
    std::vector<Value> values(graph.tensors.size()), args;
    for (const auto id : node.inputs) {
      if (id == ir::kNoTensor) continue;
      const auto& t = graph.tensors[id];
      args.push_back({std::format("%arg{}", args.size()), t.dims, std::string(element_type(t.type))});
      values[id] = args.back();
    }
    auto outputs = operation(hlo, graph, node, values);
    if (!outputs) return std::unexpected(outputs.error());
    return hlo.module(args, *outputs);
  }
};
bool has_op(const ir::Graph& graph, Op op) {
  return std::ranges::any_of(graph.nodes, [op](const auto& n) { return n.op == op; });
}

bool single_rgb_convolution(const ir::Graph& graph, std::span<const vk::net::InputShape> inputs) {
  return inputs.size() == 1 && graph.outputs.size() == 1 && inputs[0].dims.size() == 4 &&
         inputs[0].dims[0] == 1 && inputs[0].dims[1] == 3 && has_op(graph, Op::Conv);
}

// Après la réécriture des pixel-shuffles, les deux familles de mouvement
// se distinguent des agrandisseurs RVB et des graphes de stabilisation.
std::optional<Family> model_family(const ir::Graph& graph, GmfssGraph gmfss, bool temporal, bool upscale,
                                   bool motion) {
  if (gmfss != GmfssGraph::Other) return Family::Gmfss;
  if (temporal) return Family::Temporal;
  if (upscale) return Family::Upscale;
  if (motion && has_op(graph, Op::DepthToSpace)) return Family::Motion;
  return {};
}

std::expected<bool, Refusal> trace_with_sdk(ir::Graph& graph, bool fp16, Family family, RifeRecipes recipes,
                                            DrbaRecipes drba, GmfssRecipes gmfss,
                                            const std::filesystem::path& staged, Progress& progress) {
  auto data = Manifest{}.encode(graph, fp16, family, recipes, drba, gmfss);
  if (!data) return std::unexpected(data.error());
  auto source = staged;
  source += ".kzir";
  if (auto ok = vk::net::write_file(source, *data); !ok) return std::unexpected(ok.error());
  if (progress.report) {
    progress.report(0.2F,
                    std::format("compilation du réseau {} avec le frontal Zig de ZML", family_name(family)));
  }
  std::vector<std::string> arguments{core::to_utf8(source), core::to_utf8(staged)};
  if (recipes.features && family == Family::Motion && fp16 && graph.inputs.size() == 1 &&
      graph.tensors[graph.inputs[0]].dims == Dims{1, 11, 768, 1280}) {
    arguments.emplace_back("--rife-cache");
  }
  auto result = run_sdk_frontend(arguments, progress);
  if (!result && arguments.size() == 3 && result.error().code == RefusalCode::BuilderFailed &&
      result.error().detail.contains("ExpectedWeightsHeightWidthOutput")) {
    arguments.pop_back();
    result = run_sdk_frontend(std::move(arguments), progress);
  }
  // Un ancien paquet SDK ne connaît que « motion » et « upscale ». Son chemin StableHLO reste utilisable.
  if (!result && (family == Family::Temporal || family == Family::Gmfss) &&
      result.error().code == RefusalCode::BuilderFailed && result.error().detail.contains("InvalidEnumTag"))
    return false;
  return result;
}
}  // namespace

std::expected<bool, Refusal> build_motion_with_sdk(const ModelSpec& model,
                                                   std::span<const vk::net::InputShape> inputs,
                                                   const std::filesystem::path& staged, RifeRecipes recipes,
                                                   DrbaRecipes drba, GmfssRecipes gmfss_recipes,
                                                   bool artcnn_mixed_body, Progress& progress) {
  auto g = ir::load_onnx_file(model.onnx_path);
  if (!g) return std::unexpected(g.error());
  const bool motion = has_op(*g, Op::GridSample) && has_op(*g, Op::ConvTranspose);
  const bool rgb = single_rgb_convolution(*g, inputs);
  const bool temporal = temporal_fix(*g);
  const auto gmfss = gmfss_graph(*g);
  if (gmfss == GmfssGraph::PairWideWeights) return false;
  if (gmfss == GmfssGraph::Pair && !gmfss_recipes.pair) return false;
  if (!motion && !rgb && !temporal && gmfss == GmfssGraph::Other) return false;
  std::vector<ir::InputShape> shapes;
  for (const auto& in : inputs) shapes.push_back({in.name, in.dims});
  if (auto result = ir::infer_shapes(*g, shapes); !result) return std::unexpected(result.error());
  const bool upscale =
      rgb && g->tensors[g->outputs[0]].dims == Dims{1, 3, 2 * shapes[0].dims[2], 2 * shapes[0].dims[3]};
  ir::promote_grid_chains(*g);
  if (auto result = ir::fold_constants(*g, compact_rife_grid(*g)); !result)
    return std::unexpected(result.error());
  ir::eliminate_dead_code(*g);
  // Selon l'export ONNX, pixel_shuffle peut apparaître comme Reshape/Transpose/Reshape.
  ir::fuse_pixel_shuffles(*g);
  const auto family = model_family(*g, gmfss, temporal, upscale, motion);
  if (!family) return false;
  if (gmfss == GmfssGraph::Pair) widen_gmflow_intermediates(*g);
  if (*family == Family::Motion || *family == Family::Upscale) {
    optimize_drba_graph(*g);
    prune_motion_heads(*g);
  }
  if (*family == Family::Upscale && artcnn_mixed_body) use_artcnn_half_body(*g, model);
  if (auto ok = ir::validate(*g); !ok) return std::unexpected(ok.error());
  if (progress.stop.stop_requested())
    return std::unexpected(Refusal{RefusalCode::Cancelled, "préparation ZML annulée"});
  return trace_with_sdk(*g, model.precision == TensorType::F16, *family, recipes, drba,
                        gmfss == GmfssGraph::Pair ? gmfss_recipes : GmfssRecipes{}, staged, progress);
}
}  // namespace kirari::gpu::zml
