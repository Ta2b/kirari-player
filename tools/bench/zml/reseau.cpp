// Exécute le même ONNX par PJRT et par l'interpréteur de référence, sur une petite image déterministe.
#include <algorithm>
#include <charconv>
#include <cmath>
#include <iostream>

#include "core/utf8_path.h"
#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/interpreter.h"
#include "gpu/ir/shape.h"
#include "gpu/vulkan/backend/net_gate.h"
#include "gpu/vulkan/compute/context.h"
#include "gpu/zml/runtime.h"
#include "kernels/ref/half.h"

namespace {
using namespace kirari;
using namespace kirari::gpu;
using Binding = vk::net::HostBinding;

std::expected<void, Refusal> compare(vk::net::Runtime& rt, const std::filesystem::path& path,
                                     const std::filesystem::path& artifact, int side) {
  auto graph = ir::load_onnx_file(path);
  if (!graph) return std::unexpected(graph.error());
  std::vector<ir::InputShape> shapes;
  std::vector<vk::net::InputShape> fixed;
  vk::net::IoNames names;
  for (auto id : graph->inputs) {
    const auto& t = graph->tensors[id];
    auto dims = t.dims;
    if (dims.size() != 4) return refuse(RefusalCode::ShapeInvalid, "le banc attend une entrée NCHW");
    for (std::size_t k = 0; k < 4; ++k)
      if (dims[k] < 0) dims[k] = k < 2 ? 1 : side;
    shapes.push_back({t.name, dims});
    fixed.push_back({t.name, dims});
    names.inputs.push_back(t.name);
  }
  for (auto id : graph->outputs) names.outputs.push_back(graph->tensors[id].name);
  if (auto ok = ir::infer_shapes(*graph, shapes); !ok) return ok;
  Progress progress{};
  if (auto ok = rt.build({.onnx_path = path}, fixed, artifact, progress); !ok) return ok;
  if (auto ok = vk::net::write_io_names(artifact, names.inputs, names.outputs); !ok) return ok;
  auto net = rt.load(artifact);
  if (!net) return std::unexpected(net.error());
  std::vector<ir::NamedValue> reference_inputs;
  std::vector<std::vector<std::uint16_t>> data;
  std::vector<Binding> inputs, outputs;
  auto bind = [&](ir::TensorId id, std::vector<Binding>& bindings,
                  bool input) -> std::expected<void, Refusal> {
    const auto& tensor = graph->tensors[id];
    if (tensor.dims.size() != 4 || ir::element_count(tensor.dims) > 2000000)
      return refuse(RefusalCode::ShapeInvalid, "le banc de référence attend de petites images");
    Shape shape;
    std::ranges::copy(tensor.dims, shape.dims.begin());
    auto& bits = data.emplace_back(ir::element_count(tensor.dims));
    if (input) {
      auto value = ir::make_value(tensor.type, tensor.dims);
      for (std::size_t k = 0; k < bits.size(); ++k) {
        // Décale les images RGB successives de la fenêtre temporelle, en préservant les plans de grille.
        const auto width = static_cast<std::size_t>(tensor.dims[3]);
        const auto sample = tensor.dims[1] == 3 ? k - k % width + (k % width + 2 * inputs.size()) % width : k;
        bits[k] = kernels::ref::float_to_half(vk::net::gate_value(tensor.dims, sample));
        value.f[k] = kernels::ref::half_to_float(bits[k]);
      }
      reference_inputs.push_back({tensor.name, std::move(value)});
    }
    bindings.push_back({reinterpret_cast<std::byte*>(bits.data()), bits.size() * sizeof(bits.front()),
                        TensorType::F16, shape});
    return {};
  };
  for (auto id : graph->inputs)
    if (auto ok = bind(id, inputs, true); !ok) return ok;
  for (auto id : graph->outputs)
    if (auto ok = bind(id, outputs, false); !ok) return ok;
  if (auto ok = (*net)->run(inputs, outputs); !ok) return ok;
  auto expected = ir::run(*graph, reference_inputs);
  if (!expected) return std::unexpected(expected.error());
  for (std::size_t i = 0; i < outputs.size(); ++i) {
    const auto& ref = (*expected)[i].value.f;
    const auto& got = data[inputs.size() + i];
    double squared = 0;
    for (std::size_t j = 0; j < ref.size(); ++j) {
      const auto delta = static_cast<double>(kernels::ref::half_to_float(got[j])) - ref[j];
      if (!std::isfinite(delta)) return refuse(RefusalCode::ModelMismatch, "sortie non finie");
      squared += delta * delta;
    }
    const double rms = std::sqrt(squared / static_cast<double>(ref.size()));
    std::cout << names.outputs[i] << " : écart RMS " << rms << "\n";
    // Les sorties de mouvement peuvent être en pixels. Le PSNR du lecteur se mesure sur les exports RGB.
    double energy = 0;
    for (float sample : ref) energy += static_cast<double>(sample) * sample;
    if (rms > 0.01 * std::max(1.0, std::sqrt(energy / static_cast<double>(ref.size()))))
      return refuse(RefusalCode::ModelMismatch, "écart RMS supérieur à 1 % de l’échelle de référence");
  }
  return {};
}
}  // namespace
int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage : kirari_zml_check DOSSIER_TEMPORAIRE CÔTÉ MODÈLE.onnx […]\n";
    return 2;
  }
  int side = 0;
  auto s = std::string_view(argv[2]);
  auto [end, ec] = std::from_chars(s.data(), s.data() + s.size(), side);
  if (ec != std::errc{} || end != s.data() + s.size() || side < 1 || side > 256) return 2;
  auto ctx = kirari::gpu::vk::VkContext::create();
  if (!ctx) {
    std::cerr << ctx.error().detail << '\n';
    return 1;
  }
  auto rt = kirari::gpu::zml::make_runtime((*ctx)->caps());
  if (!rt) {
    std::cerr << rt.error().detail << '\n';
    return 1;
  }
  const auto folder = kirari::core::path_from_utf8(argv[1]);
  if (!std::filesystem::create_directory(folder)) {
    std::cerr << "Le dossier doit être neuf.\n";
    return 2;
  }
  bool success = true;
  for (int i = 3; i < argc; ++i) {
    auto path = kirari::core::path_from_utf8(argv[i]);
    std::cout << argv[i] << std::endl;
    if (auto ok = compare(**rt, path, folder / (std::to_string(i) + ".mlir"), side); !ok) {
      std::cerr << ok.error().detail << '\n';
      success = false;
    }
  }
  return success ? 0 : 1;
}
