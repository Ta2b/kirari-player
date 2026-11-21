#ifdef KIRARI_ZML_SPLIT_PROJECTION
#include "gpu/zml/span_projection.h"
#endif
#include <algorithm>
#include <bit>
#include <format>
#include <numeric>

#include "gpu/ir/graph.h"
#include "gpu/zml/drba_graph.h"
#include "gpu/zml/hlo.h"
#include "gpu/zml/rife_conv.h"
#ifdef KIRARI_ZML_SPAN_WINOGRAD
#include "gpu/zml/span_winograd.h"
#endif
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "kernels/ref/half.h"

namespace kirari::gpu::zml {
namespace {
std::string type(const Value& v) {
  std::string result = "tensor<";
  for (auto dim : v.dims) result += std::to_string(dim) + 'x';
  return result + v.element + '>';
}

std::string list(const auto& values, auto describe) {
  std::string result;
  for (const auto& value : values) {
    if (!result.empty()) result += ", ";
    result += describe(value);
  }
  return result;
}
}  // namespace

std::string_view element_type(ir::DataType type) {
  if (type == ir::DataType::F16) return "f16";
  if (type == ir::DataType::I64) return "i64";
  return "f32";
}

Dims axes(std::size_t count, std::int64_t start) {
  Dims result(count);
  std::iota(result.begin(), result.end(), start);
  return result;
}

std::string array(const Dims& values) {
  if (values.empty()) return "array<i64>";
  return "array<i64: " + list(values, [](auto v) { return std::to_string(v); }) + '>';
}

void Hlo::enter_node(std::string_view name) {
  location_.clear();
  if (name.empty()) return;
  location_ = " loc(\"";
  for (const char c : name) {
    const auto byte = static_cast<unsigned char>(c);
    if (byte < 32 || byte >= 127 || c == '\\' || c == '\"')
      location_ += std::format("\\{:02X}", byte);
    else
      location_ += c;
  }
  location_ += "\")";
}

Value Hlo::op(std::string_view name, const std::vector<Value>& args, const Dims& dims,
              std::string_view element, std::string_view attrs) {
  Value out{std::format("%v{}", next_++), dims, std::string(element)};
  body_ << "    " << out.id << " = \"stablehlo." << name << "\"("
        << list(args, [](const Value& v) { return v.id; }) << ')';
  if (!attrs.empty()) body_ << " {" << attrs << '}';
  body_ << " : (" << list(args, type) << ") -> " << type(out) << location_ << '\n';
  return out;
}

Value Hlo::reduce(const Value& v, const Dims& reduced, bool maximum) {
  Dims dims;
  for (std::size_t i = 0; i < v.dims.size(); ++i)
    if (std::ranges::find(reduced, static_cast<std::int64_t>(i)) == reduced.end()) dims.push_back(v.dims[i]);
  auto zero = maximum ? convert(constant({.type = ir::DataType::F32,
                                          .dims = {},
                                          .f = {-std::numeric_limits<float>::infinity()}}),
                                v.element)
                      : number(0, v.element);
  Value out{std::format("%v{}", next_++), dims, v.element};
  auto scalar = type(zero);
  body_ << std::format(
      "    {} = \"stablehlo.reduce\"({}, {}) ({{\n"
      "      ^bb0(%a: {}, %b: {}):\n"
      "        %s = \"stablehlo.{}\"(%a, %b) : ({}, {}) -> {}\n"
      "        \"stablehlo.return\"(%s) : ({}) -> ()\n"
      "    }}) {{dimensions = {}}} : ({}, {}) -> {}\n",
      out.id, v.id, zero.id, scalar, scalar, maximum ? "maximum" : "add", scalar, scalar, scalar, scalar,
      array(reduced), type(v), scalar, type(out));
  return out;
}

Value Hlo::scatter_nd(const Value& data, const Value& indices, const Value& updates) {
  const auto depth = indices.dims.back();
  const auto vector_axis = static_cast<std::int64_t>(indices.dims.size()) - 1;
  const Dims extents(data.dims.begin(), data.dims.begin() + depth);
  const auto sizes = broadcast(constant({.type = ir::DataType::I64, .dims = {depth}, .i = extents}),
                               indices.dims, {vector_axis});
  const auto negative = op("compare", {indices, expand(number(0, "i64"), indices.dims)}, indices.dims, "i1",
                           "comparison_direction = #stablehlo<comparison_direction LT>");
  const auto ids =
      op("select", {negative, binary("add", indices, sizes, indices.dims), indices}, indices.dims, "i64");
  const auto ints = [](const Dims& dims) { return list(dims, [](auto v) { return std::to_string(v); }); };
  const auto indexed = ints(axes(static_cast<std::size_t>(depth)));
  const auto window = ints(axes(data.dims.size() - static_cast<std::size_t>(depth), vector_axis));
  Value result{std::format("%v{}", next_++), data.dims, data.element};
  const auto scalar = type({{}, {}, data.element});
  body_ << std::format(
      "    {} = \"stablehlo.scatter\"({}, {}, {}) ({{\n"
      "      ^bb0(%old: {}, %new: {}):\n"
      "        \"stablehlo.return\"(%new) : ({}) -> ()\n"
      "    }}) {{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [{}], "
      "inserted_window_dims = [{}], scatter_dims_to_operand_dims = [{}], index_vector_dim = {}>, "
      "indices_are_sorted = false, unique_indices = false}} : ({}, {}, {}) -> {}\n",
      result.id, data.id, ids.id, updates.id, scalar, scalar, scalar, window, indexed, indexed, vector_axis,
      type(data), type(ids), type(updates), type(result));
  return result;
}

Value Hlo::constant(const ir::Value& data) {
  const auto element = element_type(data.type);
  std::string bytes;
  bytes.reserve(ir::value_size(data) * ir::element_size(data.type) * 2);
  const auto append = [&](std::uint64_t bits, std::size_t size) {
    constexpr std::string_view digits = "0123456789ABCDEF";
    for (std::size_t i = 0; i < size; ++i, bits >>= 8) {
      bytes += digits[(bits >> 4) & 15];
      bytes += digits[bits & 15];
    }
  };
  if (data.type == ir::DataType::I64) {
    for (auto v : data.i) append(static_cast<std::uint64_t>(v), 8);
  } else {
    for (float v : data.f) {
      if (data.type == ir::DataType::F16)
        append(kernels::ref::float_to_half(v), 2);
      else
        append(std::bit_cast<std::uint32_t>(v), 4);
    }
  }
  return op("constant", {}, data.dims, element,
            std::format("value = dense<\"0x{}\"> : {}", bytes, type({{}, data.dims, std::string(element)})));
}

Value Hlo::number(double value, std::string_view element) {
  const std::string literal = element.starts_with('i') ? std::to_string(static_cast<std::int64_t>(value))
                                                       : std::format("{:.9e}", value);
  return op("constant", {}, {}, element, std::format("value = dense<{}> : tensor<{}>", literal, element));
}

Value Hlo::broadcast(const Value& v, const Dims& dims, const Dims& mapping) {
  return op("broadcast_in_dim", {v}, dims, v.element, "broadcast_dimensions = " + array(mapping));
}
Value Hlo::expand(const Value& v, const Dims& dims) {
  if (v.dims == dims) return v;
  return broadcast(v, dims, axes(v.dims.size(), static_cast<std::int64_t>(dims.size() - v.dims.size())));
}
Value Hlo::convert(const Value& v, std::string_view element) {
  return v.element == element ? v : op("convert", {v}, v.dims, element);
}
Value Hlo::reshape(const Value& v, const Dims& dims) {
  return v.dims == dims ? v : op("reshape", {v}, dims, v.element);
}
Value Hlo::transpose(const Value& v, const Dims& permutation) {
  Dims dims;
  for (auto axis : permutation) dims.push_back(v.dims[static_cast<std::size_t>(axis)]);
  return op("transpose", {v}, dims, v.element, "permutation = " + array(permutation));
}
Value Hlo::binary(std::string_view name, const Value& a, const Value& b, const Dims& dims) {
  return op(name, {expand(a, dims), expand(b, dims)}, dims, a.element);
}
Value Hlo::slice(const Value& v, const Dims& start, const Dims& end, const Dims& strides) {
  Dims dims = end;
  for (std::size_t i = 0; i < dims.size(); ++i)
    dims[i] = end[i] <= start[i] ? 0 : 1 + (end[i] - start[i] - 1) / strides[i];
  return op("slice", {v}, dims, v.element,
            std::format("start_indices = {}, limit_indices = {}, strides = {}", array(start), array(end),
                        array(strides)));
}
std::string Hlo::module(const std::vector<Value>& inputs, const std::vector<Value>& outputs) const {
  auto types = list(outputs, type);
  return std::format("module @kirari {{\n  func.func @main({}) -> {} {{\n{}    return {} : {}\n  }}\n}}\n",
                     list(inputs, [](const Value& v) { return v.id + ": " + type(v); }),
                     outputs.size() == 1 ? types : '(' + types + ')', body_.view(),
                     list(outputs, [](const Value& v) { return v.id; }), types);
}

std::expected<std::string, Refusal> lower_onnx(const std::filesystem::path& path,
                                               std::span<const ir::InputShape> inputs,
                                               std::string_view io_element) {
  auto graph = ir::load_onnx_file(path);
  if (!graph) return std::unexpected(graph.error());
  if (auto ok = ir::infer_shapes(*graph, inputs); !ok) return std::unexpected(ok.error());
  ir::promote_grid_chains(*graph);
  if (auto ok = ir::fold_constants(*graph); !ok) return std::unexpected(ok.error());
  ir::eliminate_dead_code(*graph);
  optimize_drba_graph(*graph);
  prune_motion_heads(*graph);
  Hlo hlo;
  // Les corrélations globales de GMFlow amplifient l'arrondi du flot. Le calcul garde les
  // poids originaux et les Cast explicites du fichier ; les intermédiaires passent en FP32.
  const bool attention_graph =
      std::ranges::any_of(graph->nodes, [](const auto& n) { return n.op == ir::OpKind::LayerNorm; });
  auto intermediate = [&](Value v) {
    return attention_graph && v.element == "f16" ? hlo.convert(v, "f32") : std::move(v);
  };
  std::vector<Value> values(graph->tensors.size()), args, results;
  for (auto id : graph->inputs) {
    const auto& t = graph->tensors[id];
    args.push_back({std::format("%arg{}", args.size()), t.dims, std::string(io_element)});
    values[id] = intermediate(hlo.convert(args.back(), element_type(t.type)));
  }
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (graph->tensors[i].kind == ir::TensorKind::Constant)
      values[i] = intermediate(hlo.constant(graph->tensors[i].value));
  }
#ifdef KIRARI_ZML_SPAN_WINOGRAD
  // L'export ONNX peut décomposer le pixel-unshuffle en Reshape/Transpose/Reshape.
  const auto gates = std::ranges::count(graph->nodes, ir::OpKind::Tanh, &ir::Node::op);
  const bool span = graph->inputs.size() == 1 && graph->outputs.size() == 1 && (gates == 5 || gates == 8);
#endif
  for (const auto& node : graph->nodes) {
    hlo.enter_node(node.name);
#ifdef KIRARI_ZML_SPLIT_PROJECTION
    if (auto result = split_projection(hlo, *graph, node, values)) {
      values[node.outputs.front()] = std::move(*result);
      continue;
    }
#endif
#ifdef KIRARI_ZML_SPAN_WINOGRAD
    if (span) {
      if (auto y = span_winograd(hlo, *graph, node, values)) {
        values[node.outputs.front()] = std::move(*y);
        continue;
      }
    }
#endif
    auto result = operation(hlo, *graph, node, values);
    if (!result) return std::unexpected(result.error());
    for (std::size_t i = 0; i < node.outputs.size(); ++i)
      values[node.outputs[i]] = intermediate(std::move((*result)[i]));
  }
  for (auto id : graph->outputs) results.push_back(hlo.convert(values[id], io_element));
  return hlo.module(args, results);
}
}  // namespace kirari::gpu::zml
