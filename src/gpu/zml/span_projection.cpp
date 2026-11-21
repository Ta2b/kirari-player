#include "gpu/zml/span_projection.h"

#include "gpu/ir/graph.h"

namespace kirari::gpu::zml {

namespace {
bool projection_kernel(const Value& kernel, const ir::ConvAttrs& attrs) {
  return kernel.element == "f16" && kernel.dims.size() == 4 &&
         (kernel.dims[0] == 48 || kernel.dims[0] == 64) &&
         kernel.dims == Dims{kernel.dims[0], 4 * kernel.dims[0], 1, 1} && attrs.group == 1 &&
         attrs.strides == ir::Pair{1, 1} && attrs.pads == ir::Quad{} &&
         attrs.auto_pad == ir::AutoPad::NotSet && attrs.in_quant == 0 && attrs.out_quant == 0 &&
         attrs.epilogue.act == ir::Activation::None && !attrs.epilogue.space && !attrs.epilogue.gate &&
         !attrs.epilogue.residual;
}
}  // namespace

std::optional<Value> split_projection(Hlo& h, const ir::Graph& g, const ir::Node& n,
                                      const std::vector<Value>& values) {
  if (n.op != ir::OpKind::Conv || n.inputs.size() < 2 || n.inputs.size() > 3) return {};
  const auto& attrs = std::get<ir::ConvAttrs>(n.attrs);
  const auto& kernel = values[n.inputs[1]];
  if (!projection_kernel(kernel, attrs)) return {};
  const auto producer = ir::GraphIndex(g).producer(n.inputs[0]);
  if (producer == ir::kNoNode) return {};
  const auto& concatenate = g.nodes[producer];
  if (concatenate.op != ir::OpKind::Concat || concatenate.inputs.size() != 4 ||
      std::get<ir::AxisAttrs>(concatenate.attrs).axis != 1)
    return {};
  const auto& shape = g.tensors[n.outputs[0]].dims;
  if (shape.size() != 4) return {};
  for (const auto id : concatenate.inputs)
    if (values[id].dims != shape || values[id].element != "f16") return {};

  const auto channels = shape[1], pixels = shape[0] * shape[2] * shape[3];
  const auto weights = h.reshape(kernel, {channels, 4 * channels});
  std::optional<Value> sum;
  std::int64_t start = 0;
  for (const auto id : concatenate.inputs) {
    auto lhs = h.reshape(h.transpose(values[id], {0, 2, 3, 1}), {pixels, channels});
    auto rhs = h.slice(weights, {0, start}, {channels, start + channels}, {1, 1});
    auto product = h.op("dot_general", {lhs, rhs}, {pixels, channels}, "f32",
                        "dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], "
                        "rhs_contracting_dimensions = [1]>");
    sum = sum ? h.binary("add", *sum, product, product.dims) : std::move(product);
    start += channels;
  }
  auto out =
      h.transpose(h.reshape(h.convert(*sum, "f16"), {shape[0], shape[2], shape[3], channels}), {0, 3, 1, 2});
  if (n.inputs.size() == 3 && n.inputs[2] != ir::kNoTensor)
    out = h.binary("add", out, h.broadcast(values[n.inputs[2]], shape, {1}), shape);
  return out;
}
}  // namespace kirari::gpu::zml
