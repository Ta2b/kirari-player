#include "gpu/zml/span_winograd.h"

#include <array>

namespace kirari::gpu::zml {
namespace {
using Four = std::array<Value, 4>;
using Two = std::array<Value, 2>;
Four transform_input(Hlo& h, const Four& d) {
  return {h.binary("subtract", d[0], d[2], d[0].dims), h.binary("add", d[1], d[2], d[0].dims),
          h.binary("subtract", d[2], d[1], d[0].dims), h.binary("subtract", d[1], d[3], d[0].dims)};
}
Two transform_output(Hlo& h, const Four& m) {
  return {h.binary("add", h.binary("add", m[0], m[1], m[0].dims), m[2], m[0].dims),
          h.binary("subtract", h.binary("subtract", m[1], m[2], m[0].dims), m[3], m[0].dims)};
}
Value transformed_weights(Hlo& h, const ir::Value& w) {
  constexpr float g[4][3] = {{1, 0, 0}, {.5F, .5F, .5F}, {.5F, -.5F, .5F}, {0, 0, 1}};
  const auto channels = static_cast<std::size_t>(w.dims[1]);
  ir::Value result{.type = ir::DataType::F16, .dims = {16, w.dims[1], w.dims[0]}};
  result.f.reserve(16 * channels * channels);
  for (const auto& y : g)
    for (const auto& x : g) {
      for (std::size_t ic = 0; ic < channels; ++ic)
        for (std::size_t oc = 0; oc < channels; ++oc) {
          float sum = 0;
          for (std::size_t ky = 0; ky < 3; ++ky)
            for (std::size_t kx = 0; kx < 3; ++kx)
              sum += y[ky] * x[kx] * w.f[((oc * channels + ic) * 3 + ky) * 3 + kx];
          result.f.push_back(sum);
        }
    }
  return h.constant(result);
}
bool winograd_kernel(const Value& x, const ir::Tensor& w, const ir::ConvAttrs& a) {
  return x.dims.size() == 4 && x.dims[0] == 1 && x.element == "f16" && (x.dims[1] == 64 || x.dims[1] == 48) &&
         w.kind == ir::TensorKind::Constant && w.dims == Dims{x.dims[1], x.dims[1], 3, 3} &&
         w.value.type == ir::DataType::F16 && a.in_quant == 0 && a.out_quant == 0 && !a.epilogue.residual &&
         !a.epilogue.gate && !a.epilogue.space && a.epilogue.act == ir::Activation::None &&
         a.auto_pad == ir::AutoPad::NotSet && a.group == 1 && a.pads == ir::Quad{1, 1, 1, 1} &&
         a.strides == ir::Pair{1, 1} && a.dilations == ir::Pair{1, 1};
}
}  // namespace
std::optional<Value> span_winograd(Hlo& h, const ir::Graph& graph, const ir::Node& n,
                                   const std::vector<Value>& values) {
  if (n.op != ir::OpKind::Conv || n.inputs.size() < 2) return {};
  const auto& a = std::get<ir::ConvAttrs>(n.attrs);
  const auto& x = values[n.inputs[0]];
  const auto& w = graph.tensors[n.inputs[1]];
  if (!winograd_kernel(x, w, a)) return {};
  const auto c = x.dims[1], height = x.dims[2], width = x.dims[3], rows = (height + 1) / 2,
             cols = (width + 1) / 2;
  const auto pad = h.op(
      "pad", {x, h.number(0, "f16")}, {1, c, rows * 2 + 2, cols * 2 + 2}, "f16",
      std::format("edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}", array({0, 0, 1, 1}),
                  array({0, 0, 1 + height % 2, 1 + width % 2}), array({0, 0, 0, 0})));
  std::array<Four, 4> horizontal;
  for (std::int64_t y = 0; y < 4; ++y) {
    Four d;
    for (std::int64_t k = 0; k < 4; ++k)
      d[static_cast<std::size_t>(k)] =
          h.slice(pad, {0, 0, y, k}, {1, c, y + 2 * rows - 1, k + 2 * cols - 1}, {1, 1, 2, 2});
    horizontal[static_cast<std::size_t>(y)] = transform_input(h, d);
  }
  std::vector<Value> input(16);
  for (std::size_t k = 0; k < 4; ++k) {
    auto u = transform_input(h, {horizontal[0][k], horizontal[1][k], horizontal[2][k], horizontal[3][k]});
    for (std::size_t j = 0; j < 4; ++j)
      input[4 * j + k] = h.reshape(h.transpose(u[j], {0, 2, 3, 1}), {1, rows * cols, c});
  }
  auto lhs = h.op("concatenate", input, {16, rows * cols, c}, "f16", "dimension = 0 : i64");
  auto dots =
      h.op("dot_general", {lhs, transformed_weights(h, w.value)}, lhs.dims, "f16",
           "dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = "
           "[0], lhs_contracting_dimensions = [2], rhs_contracting_dimensions = [1]>");
  std::array<Two, 4> sums;
  for (std::int64_t j = 0; j < 4; ++j) {
    Four d;
    for (std::int64_t k = 0; k < 4; ++k) {
      const auto i = 4 * j + k;
      d[static_cast<std::size_t>(k)] =
          h.reshape(h.slice(dots, {i, 0, 0}, {i + 1, rows * cols, c}, {1, 1, 1}), {rows, cols, c});
    }
    sums[static_cast<std::size_t>(j)] = transform_output(h, d);
  }
  std::vector<Value> columns;
  for (std::size_t k = 0; k < 2; ++k) {
    auto v = transform_output(h, {sums[0][k], sums[1][k], sums[2][k], sums[3][k]});
    auto column =
        h.op("concatenate", {h.reshape(v[0], {rows, 1, cols, c}), h.reshape(v[1], {rows, 1, cols, c})},
             {rows, 2, cols, c}, "f16", "dimension = 1 : i64");
    columns.push_back(h.reshape(column, {rows * 2, cols, 1, c}));
  }
  auto joined = h.op("concatenate", columns, {rows * 2, cols, 2, c}, "f16", "dimension = 2 : i64");
  auto y = h.transpose(h.reshape(joined, {1, rows * 2, cols * 2, c}), {0, 3, 1, 2});
  if (y.dims != x.dims) y = h.slice(y, {0, 0, 0, 0}, x.dims, {1, 1, 1, 1});
  if (n.inputs.size() > 2 && n.inputs[2] != ir::kNoTensor)
    y = h.op("add", {y, h.broadcast(values[n.inputs[2]], x.dims, {1})}, x.dims, "f16");
  return y;
}
}  // namespace kirari::gpu::zml
