#include <algorithm>
#include <cmath>
#include <format>
#include <numeric>

#include "gpu/ir/ops.h"
#include "gpu/zml/hlo.h"

namespace kirari::gpu::zml {
namespace {
using Op = ir::OpKind;
Value normalize(Hlo& h, const Value& x, const ir::Node& n, const std::vector<Value>& v) {
  const auto attr = std::get<ir::NormAttrs>(n.attrs);
  const auto rank = static_cast<std::int64_t>(x.dims.size());
  const auto first = n.op == Op::InstanceNorm ? 2 : attr.axis < 0 ? attr.axis + rank : attr.axis;
  auto reduced = n.op == Op::Softmax ? Dims{first} : axes(static_cast<std::size_t>(rank - first), first);
  Dims kept = x.dims;
  double count = 1;
  for (auto axis : reduced) {
    count *= static_cast<double>(kept[static_cast<std::size_t>(axis)]);
    kept[static_cast<std::size_t>(axis)] = 1;
  }
  auto a = h.convert(x, "f32");
  if (n.op == Op::Softmax) {
    auto high = h.reshape(h.reduce(a, reduced, true), kept);
    auto e = h.op("exponential", {h.binary("subtract", a, high, x.dims)}, x.dims, "f32");
    return h.convert(h.binary("divide", e, h.reshape(h.reduce(e, reduced), kept), x.dims), x.element);
  }
  auto mean = [&](const Value& from) {
    return h.binary("divide", h.reshape(h.reduce(from, reduced), kept), h.number(count, "f32"), kept);
  };
  auto centered = h.binary("subtract", a, mean(a), x.dims);
  auto variance = mean(h.binary("multiply", centered, centered, x.dims));
  auto inv = h.op("rsqrt", {h.binary("add", variance, h.number(attr.epsilon, "f32"), kept)}, kept, "f32");
  auto scale = [&](std::size_t index) {
    auto p = h.convert(v[n.inputs[index]], "f32");
    return n.op == Op::InstanceNorm ? h.broadcast(p, x.dims, {1}) : h.expand(p, x.dims);
  };
  auto y = h.binary("multiply", h.binary("multiply", centered, inv, x.dims), scale(1), x.dims);
  if (n.inputs.size() > 2) y = h.binary("add", y, scale(2), x.dims);
  return h.convert(y, x.element);
}
}  // namespace
std::expected<Value, Refusal> attention(Hlo& h, const ir::Graph& graph, const ir::Node& n,
                                        const std::vector<Value>& v) {
  const auto& x = v[n.inputs[0]];
  const auto& d = graph.tensors[n.outputs[0]].dims;
  if (n.op == Op::LayerNorm || n.op == Op::InstanceNorm || n.op == Op::Softmax) return normalize(h, x, n, v);
  if (n.op == Op::Tile) {
    const auto& repeats = graph.tensors[n.inputs[1]].value.i;
    Dims interleaved, mapping;
    for (std::size_t i = 0; i < repeats.size(); ++i) {
      interleaved.push_back(repeats[i]);
      mapping.push_back(static_cast<std::int64_t>(interleaved.size()));
      interleaved.push_back(x.dims[i]);
    }
    return h.reshape(h.broadcast(x, interleaved, mapping), d);
  }
  if (n.op == Op::Erf) {
    // Abramowitz et Stegun 7.1.26 (erreur absolue maximale 1,5 × 10⁻⁷), en FP32.
    auto a = h.convert(x, "f32"), ab = h.op("abs", {a}, d, "f32");
    auto num = [&](double k) { return h.number(k, "f32"); };
    auto mul = [&](const Value& lhs, const Value& rhs) { return h.binary("multiply", lhs, rhs, d); };
    auto add = [&](const Value& lhs, const Value& rhs) { return h.binary("add", lhs, rhs, d); };
    auto t = h.binary("divide", num(1), add(num(1), mul(num(0.3275911), ab)), d);
    auto poly = num(1.061405429);
    for (double c : {-1.453152027, 1.421413741, -0.284496736, 0.254829592}) poly = add(mul(poly, t), num(c));
    auto e = h.op("exponential", {h.op("negate", {mul(ab, ab)}, d, "f32")}, d, "f32");
    return h.convert(mul(h.op("sign", {a}, d, "f32"), h.binary("subtract", num(1), mul(mul(poly, t), e), d)),
                     x.element);
  }
  if (n.op == Op::MatMul) {
    const auto& w = v[n.inputs[1]];
    auto a = d, b = d;
    a.back() = x.dims.back();
    b[b.size() - 2] = w.dims[w.dims.size() - 2];
    std::string batch;
    for (std::size_t k = 0; k + 2 < d.size(); ++k) batch += (k ? ", " : "") + std::to_string(k);
    return h.op(
        "dot_general", {h.expand(x, a), h.expand(w, b)}, d, x.element,
        std::format(
            "dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [{}], rhs_batching_dimensions "
            "= [{}], lhs_contracting_dimensions = [{}], rhs_contracting_dimensions = [{}]>",
            batch, batch, d.size() - 1, d.size() - 2));
  }
  if (n.op == Op::LogicalAnd)
    return h.convert(h.binary("and", h.convert(x, "i1"), h.convert(v[n.inputs[1]], "i1"), d), "i64");
  auto dir = n.op == Op::LogicalNot     ? "EQ"
             : n.op == Op::ToBool       ? "NE"
             : n.op == Op::Greater      ? "GT"
             : n.op == Op::GreaterEqual ? "GE"
                                        : "LT";
  auto rhs = n.inputs.size() == 1 ? h.number(0, x.element) : v[n.inputs[1]];
  return h.convert(h.op("compare", {h.expand(x, d), h.expand(rhs, d)}, d, "i1",
                        std::format("comparison_direction = #stablehlo<comparison_direction {}>", dir)),
                   "i64");
}
}  // namespace kirari::gpu::zml
