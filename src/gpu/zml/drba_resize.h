#pragma once

#include <span>

#include "gpu/ir/graph.h"

namespace kirari::gpu::zml {
namespace detail {
inline bool drba_resize_graph(const ir::Graph& graph) {
  const auto& a = graph.tensors[graph.inputs[0]];
  const auto& b = graph.tensors[graph.inputs[1]];
  const auto& rgb = graph.tensors[graph.outputs[0]].dims;
  if (a.name != "pair01" || b.name != "pair12" || a.dims.size() != 4 || a.dims != b.dims || a.dims[0] != 1 ||
      a.dims[1] != 7 || rgb.size() != 4 || rgb[0] != 1 || rgb[1] != 3 || rgb[2] != a.dims[2] ||
      rgb[3] != a.dims[3] || !ir::find_tensor(graph, "net.encode.cnn0.weight"))
    return false;
  return true;
}

inline bool drba_resize_layout(const ir::Tensor& input, const ir::Tensor& output,
                               std::span<const float> scales) {
  const auto& src = input.dims;
  const auto& dst = output.dims;
  if (src.size() != 4 || dst.size() != 4 ||
      (input.type != ir::DataType::F16 && input.type != ir::DataType::F32) || input.type != output.type ||
      src[0] != dst[0] || src[1] != dst[1] || scales[0] != 1 || scales[1] != 1 || scales[2] != scales[3] ||
      dst[2] <= 0 || dst[3] <= 0 || src[2] <= 0 || src[3] <= 0 || src[2] > (1 << 23) || src[3] > (1 << 23))
    return false;
  return true;
}
}  // namespace detail

// Un rapport pair entier en half_pixel lit (r/2-1,r/2) dans chaque cellule.
// Les deux axes sont contrôlés, y compris les scales ONNX explicites : une taille
// arrondie identique peut provenir d'une échelle différente.
inline int drba_resize_factor(const ir::Graph& graph, const ir::Node& node, std::span<const float> scales) {
  if (node.op != ir::OpKind::Resize || scales.size() != 4 || graph.inputs.size() != 2 ||
      graph.outputs.size() != 1)
    return 0;
  if (!detail::drba_resize_graph(graph)) return 0;
  const auto& attrs = std::get<ir::ResizeAttrs>(node.attrs);
  const auto& input = graph.tensors[node.inputs.front()];
  const auto& output = graph.tensors[node.outputs.front()];
  const auto& src = input.dims;
  const auto& dst = output.dims;
  if (attrs.mode != ir::ResizeMode::Linear || !detail::drba_resize_layout(input, output, scales)) return 0;
  if (attrs.coord != ir::CoordMode::HalfPixel &&
      !(attrs.coord == ir::CoordMode::PytorchHalfPixel && dst[2] > 1 && dst[3] > 1))
    return 0;
  const int factor = scales[2] == 0.25F ? 4 : scales[2] == 0.125F ? 8 : 0;
  if (factor == 0 || src[2] % factor || src[3] % factor || dst[2] != src[2] / factor ||
      dst[3] != src[3] / factor)
    return 0;
  return factor;
}
}  // namespace kirari::gpu::zml
