#include <algorithm>
#include <format>

#include "gpu/ir/graph.h"
#include "gpu/ir/ops.h"
#include "gpu/zml/hlo.h"
#include "gpu/zml/rife_conv.h"

namespace kirari::gpu::zml {
namespace {
using Op = ir::OpKind;
using Results = std::vector<Value>;
std::int64_t axis_of(std::int64_t axis, const Dims& dims) {
  return axis < 0 ? axis + static_cast<std::int64_t>(dims.size()) : axis;
}

Value convolution(Hlo& h, const Value& x, const Value& weights, const Dims& dims, const ir::Pair& strides,
                  const ir::Pair& lhs, const ir::Pair& rhs, const ir::Quad& pads, std::int64_t group,
                  bool transposed) {
  const auto attrs = std::format(
      "window_strides = {}, lhs_dilation = {}, rhs_dilation = {}, "
      "padding = dense<[[{}, {}], [{}, {}]]> : tensor<2x2xi64>, "
      "dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[{}, 0, 1]->[b, f, 0, 1]>, "
      "feature_group_count = {} : i64, batch_group_count = 1 : i64",
      array({strides[0], strides[1]}), array({lhs[0], lhs[1]}), array({rhs[0], rhs[1]}), pads[0], pads[2],
      pads[1], pads[3], transposed ? "i, o" : "o, i", group);
  auto w = transposed
               ? h.op("reverse", {weights}, weights.dims, weights.element, "dimensions = array<i64: 2, 3>")
               : weights;
  return h.op("convolution", {x, w}, dims, x.element, attrs);
}

Value rearrange(Hlo& h, const Value& x, const Dims& dst, const ir::BlockAttrs& a, bool shuffle) {
  const auto b = a.blocksize;
  if (!shuffle) {
    auto split = h.reshape(x, {x.dims[0], x.dims[1], dst[2], b, dst[3], b});
    return h.reshape(
        h.transpose(split, a.mode == ir::BlockMode::Dcr ? Dims{0, 3, 5, 1, 2, 4} : Dims{0, 1, 3, 5, 2, 4}),
        dst);
  }
  const bool dcr = a.mode == ir::BlockMode::Dcr;
  auto split = h.reshape(x, {dst[0], dcr ? b : dst[1], b, dcr ? dst[1] : b, x.dims[2], x.dims[3]});
  return h.reshape(h.transpose(split, dcr ? Dims{0, 3, 4, 1, 5, 2} : Dims{0, 1, 4, 2, 5, 3}), dst);
}
using Lowered = std::expected<Results, Refusal>;

class Lowering {
 public:
  Lowering(Hlo& hlo, const ir::Graph& graph, const ir::Node& node, const std::vector<Value>& values)
      : h(hlo), g(graph), n(node), v(values), x(input(0)), out(g.tensors[n.outputs[0]]), d(out.dims) {}
  Lowered emit() {
    switch (n.op) {
      case Op::Conv:
      case Op::ConvTranspose:
        return convolve();
      case Op::FakeQuantInt8:
      case Op::FakeQuantE4M3:
        return fake_quant();
      case Op::AveragePool:
        return average_pool();
      case Op::ReduceSum:
      case Op::ReduceMean:
      case Op::ReduceL2:
        return reduce();
      case Op::CumSum:
        return scan();
      case Op::ScatterND:
        return Results{h.scatter_nd(x, input(1), input(2))};
      case Op::Gather:
        return gather();
      case Op::Resize:
      case Op::GridSample: {
        auto y = sample(h, g, n, v);
        if (!y) return std::unexpected(y.error());
        return Results{*y};
      }
      default:
        return elementwise();
    }
  }

 private:
  Hlo& h;
  const ir::Graph& g;
  const ir::Node& n;
  const std::vector<Value>& v;
  const Value& x;
  const ir::Tensor& out;
  const Dims& d;

  bool has(std::size_t k) const { return k < n.inputs.size() && n.inputs[k] != ir::kNoTensor; }
  const Value& input(std::size_t k) const { return v[n.inputs[k]]; }
  const Dims& ints(std::size_t k) const { return g.tensors[n.inputs[k]].value.i; }
  Results unary(std::string_view op) { return {h.op(op, {x}, d, x.element)}; }
  Results binary(std::string_view op) { return {h.binary(op, x, input(1), d)}; }
  std::unexpected<Refusal> unsupported() const {
    return std::unexpected(Refusal{RefusalCode::FormatUnsupported,
                                   std::format("ZML : {} ({}) non pris en charge", ir::name(n.op), n.name)});
  }

  Lowered elementwise() {
    switch (n.op) {
      case Op::Neg:
        return unary("negate");
      case Op::Cos:
        return unary("cosine");
      case Op::Sin:
        return unary("sine");
      case Op::Pow:
        return Results{h.binary("power", x, h.convert(input(1), x.element), d)};
      case Op::Abs:
        return unary("abs");
      case Op::Equal: {
        auto eq = h.op("compare", {h.expand(x, d), h.expand(input(1), d)}, d, "i1",
                       "comparison_direction = #stablehlo<comparison_direction EQ>");
        return Results{h.convert(eq, "i64")};
      }
      case Op::Where:
        return Results{h.op("select",
                            {h.expand(h.convert(x, "i1"), d), h.expand(input(1), d), h.expand(input(2), d)},
                            d, input(1).element)};
      case Op::Add:
        return binary("add");
      case Op::Sub:
        return binary("subtract");
      case Op::Mul:
        return binary("multiply");
      case Op::Div:
        return binary("divide");
      case Op::Sigmoid:
        return unary("logistic");
      case Op::Tanh:
        return unary("tanh");
      case Op::Reciprocal:
        return Results{h.binary("divide", h.number(1, x.element), x, d)};
      case Op::Cast:
        return Results{h.convert(x, element_type(out.type))};
      case Op::Relu:
        return Results{h.binary("maximum", x, h.number(0, x.element), d)};
      case Op::LeakyRelu:
      case Op::PRelu: {
        const auto zero = h.number(0, x.element);
        auto slope =
            n.op == Op::PRelu ? input(1) : h.number(std::get<ir::LeakyReluAttrs>(n.attrs).alpha, x.element);
        return Results{h.binary("add", h.binary("maximum", x, zero, d),
                                h.binary("multiply", slope, h.binary("minimum", x, zero, d), d), d)};
      }
      case Op::Clip: {
        auto y = x;
        if (has(1)) y = h.binary("maximum", y, input(1), d);
        if (has(2)) y = h.binary("minimum", y, input(2), d);
        return Results{y};
      }
      default:
        return layout();
    }
  }
  Lowered layout() {
    switch (n.op) {
      case Op::ConstantOfShape: {
        auto val = h.reshape(h.constant(std::get<ir::FillAttrs>(n.attrs).scalar), {});
        return Results{h.expand(val, d)};
      }
      case Op::Expand:
        return Results{h.expand(x, d)};
      case Op::Identity:
      case Op::Reshape:
      case Op::Unsqueeze:
      case Op::Squeeze:
        return Results{h.reshape(x, d)};
      case Op::SpaceToDepth:
      case Op::DepthToSpace:
        return Results{rearrange(h, x, d, std::get<ir::BlockAttrs>(n.attrs), n.op == Op::DepthToSpace)};
      case Op::Transpose: {
        auto perm = std::get<ir::TransposeAttrs>(n.attrs).perm;
        if (perm.empty()) {
          perm = axes(x.dims.size());
          std::ranges::reverse(perm);
        }
        return Results{h.transpose(x, perm)};
      }
      case Op::Concat: {
        Results parts;
        for (auto id : n.inputs) parts.push_back(v[id]);
        return Results{
            h.op("concatenate", parts, d, x.element,
                 std::format("dimension = {} : i64", axis_of(std::get<ir::AxisAttrs>(n.attrs).axis, d)))};
      }
      case Op::Split: {
        const auto axis = static_cast<std::size_t>(axis_of(std::get<ir::SplitAttrs>(n.attrs).axis, x.dims));
        Dims start(d.size(), 0), end = x.dims, strides(d.size(), 1);
        Results parts;
        for (auto id : n.outputs) {
          end[axis] = start[axis] + g.tensors[id].dims[axis];
          parts.push_back(h.slice(x, start, end, strides));
          start[axis] = end[axis];
        }
        return parts;
      }
      case Op::Slice:
        return slice();
      case Op::Pad:
        return pad();
      default:
        return attention_or_logic();
    }
  }
  Lowered attention_or_logic() {
    switch (n.op) {
      case Op::ToBool:
      case Op::LogicalNot:
      case Op::Greater:
      case Op::GreaterEqual:
      case Op::Less:
      case Op::LogicalAnd:
      case Op::Erf:
      case Op::MatMul:
      case Op::Tile:
      case Op::Softmax:
      case Op::InstanceNorm:
      case Op::LayerNorm: {
        auto y = attention(h, g, n, v);
        if (!y) return std::unexpected(y.error());
        return Results{std::move(*y)};
      }
      default:
        return unsupported();
    }
  }
  Value forward_convolution(const Value& w) {
    Value y;
    const auto& a = std::get<ir::ConvAttrs>(n.attrs);
    auto pads = a.auto_pad == ir::AutoPad::Valid ? ir::Quad{} : a.pads;
    if (a.auto_pad == ir::AutoPad::SameUpper || a.auto_pad == ir::AutoPad::SameLower) {
      for (std::size_t i = 0; i < 2; ++i) {
        const auto total = std::max<std::int64_t>(
            0, (d[i + 2] - 1) * a.strides[i] + 1 + (w.dims[i + 2] - 1) * a.dilations[i] - x.dims[i + 2]);
        pads[i] = a.auto_pad == ir::AutoPad::SameUpper ? total / 2 : total - total / 2;
        pads[i + 2] = total - pads[i];
      }
    }
    if (w.dims[2] == 1 && w.dims[3] == 1 && w.dims[1] > w.dims[0] && a.group == 1 && pads == ir::Quad{} &&
        a.strides == ir::Pair{1, 1}) {
      const auto pixels = x.dims[0] * x.dims[2] * x.dims[3];
      auto features = h.reshape(h.transpose(x, {0, 2, 3, 1}), {pixels, w.dims[1]});
      auto kernel = h.reshape(w, {w.dims[0], w.dims[1]});
      auto product = h.op("dot_general", {features, kernel}, {pixels, w.dims[0]}, x.element,
                          "dot_dimension_numbers = #stablehlo.dot<lhs_contracting_dimensions = [1], "
                          "rhs_contracting_dimensions = [1]>");
      y = h.transpose(h.reshape(product, {d[0], d[2], d[3], d[1]}), {0, 3, 1, 2});
    } else {
      y = convolution(h, x, w, d, a.strides, {1, 1}, a.dilations, pads, a.group, false);
    }
    return y;
  }
  Lowered convolve() {
    Value y;
    const auto& w = input(1);
    if (n.op == Op::Conv) {
      y = forward_convolution(w);
    } else {
      const auto& a = std::get<ir::ConvTransposeAttrs>(n.attrs);
      if (a.auto_pad != ir::AutoPad::NotSet || !a.output_shape.empty() || a.group != 1) return unsupported();
      ir::Quad pads{};
      for (std::size_t i = 0; i < 2; ++i) {
        const auto extent = (w.dims[2 + i] - 1) * a.dilations[i];
        pads[i] = extent - a.pads[i];
        pads[i + 2] = extent - a.pads[i + 2] + a.output_padding[i];
      }
#ifdef KIRARI_ZML_POLYPHASE
      if (w.dims[2] == 4 && w.dims[3] == 4 && a.strides == ir::Pair{2, 2} && a.dilations == ir::Pair{1, 1} &&
          a.output_padding == ir::Pair{} && a.pads == ir::Quad{1, 1, 1, 1}) {
        y = transpose_conv_4x4(h, x, w);
      } else
#endif
      {
        y = convolution(h, x, w, d, {1, 1}, a.strides, a.dilations, pads, 1, true);
      }
    }
    if (has(2)) y = h.binary("add", y, h.broadcast(input(2), d, {1}), d);
    return Results{y};
  }

  Lowered fake_quant() {
    auto scale = h.reshape(h.convert(input(1), "f32"), {});
    auto y = h.binary("divide", h.convert(x, "f32"), scale, d);
    if (n.op == Op::FakeQuantInt8) {
      auto finite_or_inf =
          h.op("compare", {y, y}, d, "i1", "comparison_direction = #stablehlo<comparison_direction EQ>");
      y = h.op("select", {finite_or_inf, y, h.expand(h.number(0, "f32"), d)}, d, "f32");
      y = h.op("round_nearest_even", {y}, d, "f32");
      y = h.op("clamp", {h.number(-128, "f32"), y, h.number(127, "f32")}, d, "f32");
    } else {
      y = h.op("clamp", {h.number(-448, "f32"), y, h.number(448, "f32")}, d, "f32");
      y = h.convert(h.convert(y, "f8E4M3FN"), "f32");
    }
    return Results{h.convert(h.binary("multiply", y, scale, d), x.element)};
  }

  Lowered average_pool() {
    const auto& a = std::get<ir::PoolAttrs>(n.attrs);
    auto pads = a.auto_pad == ir::AutoPad::Valid ? ir::Quad{} : a.pads;
    if (a.auto_pad == ir::AutoPad::SameUpper || a.auto_pad == ir::AutoPad::SameLower) {
      for (std::size_t k = 0; k < 2; ++k) {
        const auto total =
            std::max<std::int64_t>(0, (d[k + 2] - 1) * a.strides[k] + a.kernel[k] - x.dims[k + 2]);
        pads[k] = a.auto_pad == ir::AutoPad::SameUpper ? total / 2 : total - total / 2;
        pads[k + 2] = total - pads[k];
      }
    }
    auto sum = [&](const Value& from) {
      const auto channels = from.dims[1];
      auto kernel = h.expand(h.number(1, "f32"), {channels, 1, a.kernel[0], a.kernel[1]});
      return convolution(h, from, kernel, {from.dims[0], channels, d[2], d[3]}, a.strides, {1, 1}, {1, 1},
                         pads, channels, false);
    };
    auto total = sum(h.convert(x, "f32"));
    auto count = h.number(static_cast<double>(a.kernel[0] * a.kernel[1]), "f32");
    if (!a.count_include_pad && pads != ir::Quad{})
      count = sum(h.expand(h.number(1, "f32"), {1, 1, x.dims[2], x.dims[3]}));
    return Results{h.convert(h.binary("divide", total, count, d), x.element)};
  }

  Lowered reduce() {
    Dims reduced = has(1) ? ints(1) : Dims{};
    if (reduced.empty() && std::get<ir::ReduceAttrs>(n.attrs).noop_with_empty_axes) return Results{x};
    if (reduced.empty()) reduced = axes(x.dims.size());
    std::int64_t count = 1;
    for (auto& a : reduced) {
      a = axis_of(a, x.dims);
      count *= x.dims[static_cast<std::size_t>(a)];
    }
    auto y = h.convert(x, "f32");
    if (n.op == Op::ReduceL2) y = h.binary("multiply", y, y, y.dims);
    y = h.reduce(y, reduced);
    if (n.op == Op::ReduceL2) y = h.op("sqrt", {y}, y.dims, y.element);
    if (n.op == Op::ReduceMean)
      y = h.binary("divide", y, h.number(static_cast<double>(count), y.element), y.dims);
    return Results{h.reshape(h.convert(y, x.element), d)};
  }

  Lowered scan() {
    const auto a = static_cast<std::size_t>(axis_of(ints(1).front(), d));
    const auto& attrs = std::get<ir::ScanAttrs>(n.attrs);
    auto y = h.convert(x, "f32");
    auto shift = [&](const Value& from, std::int64_t distance) {
      Dims start(d.size()), end = d, low(d.size()), high(d.size()), zeros(d.size());
      if (attrs.reverse) {
        start[a] = distance;
        high[a] = distance;
      } else {
        end[a] -= distance;
        low[a] = distance;
      }
      auto part = h.slice(from, start, end, Dims(d.size(), 1));
      return h.op("pad", {part, h.number(0, "f32")}, d, "f32",
                  std::format("edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}",
                              array(low), array(high), array(zeros)));
    };
    for (std::int64_t step = 1; step < d[a]; step *= 2) y = h.binary("add", y, shift(y, step), d);
    if (attrs.exclusive && d[a] > 0) y = shift(y, 1);
    return Results{h.convert(y, x.element)};
  }

  Lowered gather() {
    auto axis = static_cast<std::size_t>(axis_of(std::get<ir::AxisAttrs>(n.attrs).axis, x.dims));
    const auto& index = input(1);
    auto negative = h.op("compare", {index, h.expand(h.number(0, index.element), index.dims)}, index.dims,
                         "i1", "comparison_direction = #stablehlo<comparison_direction LT>");
    auto positive =
        h.binary("add", index, h.number(static_cast<double>(x.dims[axis]), index.element), index.dims);
    auto ids = h.op("select", {negative, positive, index}, index.dims, index.element);
    Dims sizes = x.dims;
    sizes[axis] = 1;
    std::string offsets;
    for (std::size_t j = 0; j < d.size(); ++j)
      if (j < axis || j >= axis + index.dims.size())
        offsets += (offsets.empty() ? "" : ", ") + std::to_string(j);
    return Results{
        h.op("gather", {x, ids}, d, x.element,
             std::format("dimension_numbers = #stablehlo.gather<offset_dims = [{}], collapsed_slice_dims = "
                         "[{}], start_index_map = [{}], index_vector_dim = {}>, slice_sizes = {}",
                         offsets, axis, axis, index.dims.size(), array(sizes)))};
  }

  Lowered slice() {
    Dims start(d.size(), 0), end = x.dims, strides(d.size(), 1);
    const auto& starts = ints(1);
    auto dims = has(3) ? ints(3) : axes(starts.size());
    if (dims.size() != starts.size() || (has(4) && ints(4).size() != dims.size())) return unsupported();
    for (std::size_t j = 0; j < dims.size(); ++j) {
      const auto i = static_cast<std::size_t>(axis_of(dims[j], d));
      strides[i] = has(4) ? ints(4)[j] : 1;
      if (strides[i] < 1 || strides[i] > ir::kMaxDim) return unsupported();
      start[i] = std::clamp<std::int64_t>(starts[j] < 0 ? x.dims[i] + starts[j] : starts[j], 0, x.dims[i]);
      end[i] = d[i] == 0 ? start[i] : start[i] + (d[i] - 1) * strides[i] + 1;
    }
    return Results{h.slice(x, start, end, strides)};
  }

  Lowered pad() {
    const auto& p = ints(1);
    auto dims = has(3) ? ints(3) : axes(d.size());
    if (p.size() != 2 * dims.size()) return unsupported();
    if (x.dims == d && std::ranges::all_of(p, [](auto k) { return k == 0; })) return Results{x};
    if (std::get<ir::PadAttrs>(n.attrs).mode != ir::PadMode::Constant) return unsupported();
    Dims low(d.size()), high(d.size());
    for (std::size_t j = 0; j < dims.size(); ++j) {
      auto i = static_cast<std::size_t>(axis_of(dims[j], d));
      low[i] = p[j];
      high[i] = p[j + dims.size()];
    }
    auto fill = has(2) ? h.reshape(input(2), {}) : h.number(0, x.element);
    return Results{h.op("pad", {x, fill}, d, x.element,
                        std::format("edge_padding_low = {}, edge_padding_high = {}, interior_padding = {}",
                                    array(low), array(high), array(Dims(d.size()))))};
  }
};
}  // namespace

std::expected<std::vector<Value>, Refusal> operation(Hlo& h, const ir::Graph& g, const ir::Node& n,
                                                     const std::vector<Value>& v) {
  return Lowering(h, g, n, v).emit();
}
}  // namespace kirari::gpu::zml
