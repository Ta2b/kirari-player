// Rejoue les 14 sorties du module GMFSS de lecture sur des paires RGB prélevées dans les captures.
// Le chargement et l'exécution passent par le runtime du lecteur, avec les sorties indépendantes.
#include <algorithm>
#include <array>
#include <iostream>

#include "core/utf8_path.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_build.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/zml/runtime.h"

namespace {
using namespace kirari;
using namespace kirari::gpu;
using Binding = vk::net::HostBinding;

std::expected<void, Refusal> replay(const std::filesystem::path& model, const std::filesystem::path& module,
                                    const std::filesystem::path& output, std::span<char*> frames,
                                    bool trace) {
  auto graph = ir::load_onnx_file(model);
  if (!graph) return std::unexpected(graph.error());
  if (graph->inputs.size() != 1 || graph->outputs.size() != 14 ||
      graph->tensors[graph->inputs[0]].name != "frames")
    return refuse(RefusalCode::InputMismatch, "paire GMFSS à quatorze sorties attendue");
  const auto& dims = graph->tensors[graph->inputs.front()].dims;
  if (dims.size() != 4 || dims[0] != 1 || dims[1] != 6 || dims[2] <= 0 || dims[3] <= 0)
    return refuse(RefusalCode::ShapeInvalid,
                  "une paire GMFlow préparée aux dimensions du lecteur est requise");
  const std::array<ir::InputShape, 1> fixed{{{"frames", dims}}};
  if (auto ok = ir::infer_shapes(*graph, fixed); !ok) return ok;
  Shape in_shape;
  std::ranges::copy(fixed[0].dims, in_shape.dims.begin());
  auto ctx = vk::VkContext::create();
  if (!ctx) return std::unexpected(ctx.error());
  auto runtime = zml::make_runtime((*ctx)->caps());
  if (!runtime) return std::unexpected(runtime.error());
  vk::net::IoNames names{{"frames"}, {}};
  std::vector<std::vector<std::byte>> buffers;
  std::vector<Binding> outputs;
  constexpr std::array<std::string_view, 14> expected{
      "feature0_1", "motion0_1", "source0_2", "motion0_2", "source0_3", "motion0_3", "feature1_1",
      "motion1_1",  "source1_2", "motion1_2", "source1_3", "motion1_3", "half0",     "half1"};
  for (auto id : graph->outputs) {
    const auto& t = graph->tensors[id];
    if (t.name != expected[names.outputs.size()])
      return refuse(RefusalCode::InputMismatch, "sortie GMFSS inattendue");
    if (t.dims.size() != 4 || t.type != ir::DataType::F16 || !t.known)
      return refuse(RefusalCode::ShapeInvalid, "sortie GMFSS NCHW FP16 attendue");
    Shape shape;
    std::ranges::copy(t.dims, shape.dims.begin());
    names.outputs.push_back(t.name);
    if (trace) continue;
    auto& data = buffers.emplace_back(2 * ir::element_count(t.dims));
    outputs.push_back({data.data(), data.size(), TensorType::F16, shape});
  }
  Progress progress{};
  if (trace) {
    const std::array<vk::net::InputShape, 1> shapes{{{"frames", fixed[0].dims}}};
    if (auto ok =
            (*runtime)->build({.id = "gmfss_fortuna_s05_pair", .onnx_path = model}, shapes, module, progress);
        !ok)
      return ok;
    auto manifest = module;
    manifest += ".kzir";
    if (!std::filesystem::is_regular_file(manifest))
      return refuse(RefusalCode::ModelMismatch, "le frontal SDK n’a pas émis de manifeste pour la paire");
    return vk::net::write_io_names(module, names.inputs, names.outputs);
  }
  if (auto ok = vk::net::write_io_names(module, names.inputs, names.outputs); !ok) return ok;
  if (auto ok = (*runtime)->prepare_compiled_artifact(module, names, progress); !ok) return ok;
  auto network = (*runtime)->load(module);
  if (!network) return std::unexpected(network.error());
  for (std::size_t sample = 0; sample < frames.size(); ++sample) {
    auto data = vk::net::read_file(core::path_from_utf8(frames[sample]));
    if (!data) return std::unexpected(data.error());
    if (data->size() != 2 * ir::element_count(fixed[0].dims))
      return refuse(RefusalCode::InputMismatch,
                    "taille de capture différente de l’entrée de la paire en FP16");
    const std::array<Binding, 1> input{{{data->data(), data->size(), TensorType::F16, in_shape}}};
    if (auto ok = (*network)->run(input, outputs); !ok) return ok;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      auto name = std::to_string(sample) + "-" + names.outputs[i] + ".f16";
      if (auto ok = vk::net::write_file(output / name, buffers[i]); !ok) return ok;
    }
  }
  return {};
}
}  // namespace

int main(int argc, char** argv) {
  const bool trace = argc > 1 && std::string_view(argv[1]) == "--tracer-sdk";
  if ((trace && argc != 4) || (!trace && argc < 5)) {
    std::cerr << "Usage : paire_gmflow PAIRE.onnx MODULE.mlir DOSSIER_NEUF CAPTURE.f16 […]\n"
                 "        paire_gmflow --tracer-sdk PAIRE.onnx DOSSIER_NEUF\n";
    return 2;
  }
  const auto output = kirari::core::path_from_utf8(argv[3]);
  if (!std::filesystem::create_directory(output)) {
    std::cerr << "Le dossier des sorties doit être neuf.\n";
    return 2;
  }
  const auto model = kirari::core::path_from_utf8(argv[trace ? 2 : 1]);
  const auto module = trace ? output / "paire.mlir" : kirari::core::path_from_utf8(argv[2]);
  auto ok = replay(model, module, output, {argv + 4, static_cast<std::size_t>(argc - 4)}, trace);
  if (!ok) std::cerr << ok.error().detail << '\n';
  return ok ? 0 : 1;
}
