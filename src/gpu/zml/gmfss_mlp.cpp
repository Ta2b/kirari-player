#include "gpu/zml/gmfss_mlp.h"

#include <algorithm>
#include <array>
#include <bit>
#include <ranges>

#include "core/model_index.h"
#include "gpu/ir/graph.h"
#include "kernels/ref/half.h"

namespace kirari::gpu::zml {
namespace {
using ir::Dims;
using ir::OpKind;
using ir::TensorId;
constexpr std::string_view kPrefix = "/flownet/transformer/layers.0/";

// Les 9 premières places sont les opérandes externes du MLP. Les dix suivantes sont les sorties
// dans l'ordre ONNX. Les deux Mul de la GELU conservent leurs opérandes et leurs arrondis.
enum Port : int {
  X,
  Message,
  W1,
  Divisor,
  One,
  Half,
  W2,
  Gamma,
  Bias,
  Cat,
  Up,
  Div,
  Erf,
  Add1,
  Mul,
  Gelu,
  Down,
  Norm,
  Out
};
struct Stage {
  std::string_view suffix;
  OpKind op;
  std::array<int, 3> input;
  int channels;
};
constexpr std::array<Stage, 10> kStages{{
    {"cross_attn_ffn_1/Concat_16", OpKind::Concat, {X, Message, -1}, 256},
    {"cross_attn_ffn/mlp/mlp.0_1/MatMul", OpKind::MatMul, {Cat, W1, -1}, 1024},
    {"cross_attn_ffn/mlp/mlp.1_1/Div", OpKind::Div, {Up, Divisor, -1}, 1024},
    {"cross_attn_ffn/mlp/mlp.1_1/Erf", OpKind::Erf, {Div, -1, -1}, 1024},
    {"cross_attn_ffn/mlp/mlp.1_1/Add", OpKind::Add, {Erf, One, -1}, 1024},
    {"cross_attn_ffn/mlp/mlp.1_1/Mul", OpKind::Mul, {Up, Add1, -1}, 1024},
    {"cross_attn_ffn/mlp/mlp.1_1/Mul_1", OpKind::Mul, {Mul, Half, -1}, 1024},
    {"cross_attn_ffn/mlp/mlp.2_1/MatMul", OpKind::MatMul, {Gelu, W2, -1}, 128},
    {"cross_attn_ffn/norm2_1/LayerNormalization", OpKind::LayerNorm, {Down, Gamma, Bias}, 128},
    {"cross_attn_ffn_1/Add", OpKind::Add, {X, Norm, -1}, 128},
}};
using Slots = std::array<TensorId, Out + 1>;

bool bind(Slots& slots, const ir::Node& node, const Stage& spec) {
  const auto count =
      static_cast<std::size_t>(std::ranges::count_if(spec.input, [](int k) { return k >= 0; }));
  if (node.op != spec.op || node.inputs.size() != count || node.outputs.size() != 1) return false;
  for (std::size_t j = 0; j < count; ++j) {
    auto& port = slots[static_cast<std::size_t>(spec.input[j])];
    const auto tensor = node.inputs[j];
    if (tensor == ir::kNoTensor || (port != ir::kNoTensor && port != tensor)) return false;
    port = tensor;
  }
  return true;
}

bool real_constant(const ir::Tensor& t, const Dims& dims) {
  return t.kind == ir::TensorKind::Constant && (t.type == ir::DataType::F16 || t.type == ir::DataType::F32) &&
         t.dims == dims && t.value.f.size() == ir::element_count(dims);
}

bool exact_parameter(const ir::Tensor& t, const Dims& dims, std::string_view expected) {
  if (!real_constant(t, dims)) return false;
  std::vector<std::byte> bytes;
  bytes.reserve(t.value.f.size() * 2);
  for (const float v : t.value.f) {
    const auto bits = kernels::ref::float_to_half(v);
    if (std::bit_cast<std::uint32_t>(v) != std::bit_cast<std::uint32_t>(kernels::ref::half_to_float(bits)))
      return false;
    bytes.push_back(static_cast<std::byte>(bits & 255U));
    bytes.push_back(static_cast<std::byte>(bits >> 8));
  }
  core::Sha256 hash;
  hash.update(bytes);
  return core::to_hex(hash.finish()) == expected;
}

bool parameters(const ir::Graph& graph, const Slots& slots) {
  auto scalar = [&](Port port, float value) {
    const auto& t = graph.tensors[slots[static_cast<std::size_t>(port)]];
    return real_constant(t, {}) && t.value.f.front() == value;
  };
  auto weight = [&](Port port, const Dims& dims, std::string_view digest) {
    return exact_parameter(graph.tensors[slots[static_cast<std::size_t>(port)]], dims, digest);
  };
  const Dims source{4, 8640, 128};
  return graph.tensors[slots[X]].dims == source && graph.tensors[slots[Message]].dims == source &&
         scalar(Divisor, 1.4140625F) && scalar(One, 1.0F) && scalar(Half, 0.5F) &&
         weight(W1, {256, 1024}, "d2e6124caa0be2f84d537d62afe7ce0a425852fb73a14503040c1ecdb59bc583") &&
         weight(W2, {1024, 128}, "506f8e306cb99d6e3d9b287b27d73ac71d285396bd95e39c4d85b6b40aa93c36") &&
         weight(Gamma, {128}, "4d8936253ea1b260587830cb78fd1c2059b55907acfd5c77af8af148b3e0bb95") &&
         weight(Bias, {128}, "63ba8e917c5b57825f240ab66c611ac4ded6b96c9a3757bd744859ff0462fca2");
}

bool private_members(const ir::Graph& g, const MotionGroup& group, const Slots& slots) {
  ir::GraphIndex index(g);
  auto produced_by = [&](Port port, std::string_view suffix) {
    const auto node = index.producer(slots[static_cast<std::size_t>(port)]);
    return node != ir::kNoNode && g.nodes[node].name == std::string(kPrefix) + std::string(suffix);
  };
  if (!produced_by(X, "self_attn_1/Add") ||
      !produced_by(Message, "cross_attn_ffn/norm1_1/LayerNormalization"))
    return false;
  for (int port = Cat; port < Out; ++port) {
    const auto id = slots[static_cast<std::size_t>(port)];
    if (ir::is_graph_output(g, id)) return false;
    for (auto reader : index.readers(id))
      if (!std::ranges::binary_search(group.members, reader)) return false;
  }
  return true;
}
}  // namespace

std::optional<MotionGroup> gmflow_quarter_mlp(const ir::Graph& graph) {
  if (graph.inputs.size() != 1 || graph.outputs.size() != 14) return {};
  const auto& frames = graph.tensors[graph.inputs.front()];
  if (frames.name != "frames" || frames.dims != Dims{1, 6, 1152, 1920}) return {};
  MotionGroup group{.op = "KirariGmflowMlp"};
  Slots slots;
  slots.fill(ir::kNoTensor);
  auto first = graph.nodes.begin();
  for (std::size_t k = 0; k < kStages.size(); ++k) {
    const auto& spec = kStages[k];
    const auto full = std::string(kPrefix) + std::string(spec.suffix);
    const auto iter = std::ranges::find(first, graph.nodes.end(), full, &ir::Node::name);
    if (iter == graph.nodes.end() || !bind(slots, *iter, spec)) return {};
    const auto id = iter->outputs.front();
    if (graph.tensors[id].type != ir::DataType::F32 || graph.tensors[id].dims != Dims{4, 8640, spec.channels})
      return {};
    slots[Cat + k] = id;
    group.members.push_back(static_cast<std::size_t>(iter - graph.nodes.begin()));
    first = std::next(iter);
  }
  const auto& cat = std::get<ir::AxisAttrs>(graph.nodes[group.members.front()].attrs);
  const auto& norm = std::get<ir::NormAttrs>(graph.nodes[group.members[8]].attrs);
  if ((cat.axis != -1 && cat.axis != 2) || (norm.axis != -1 && norm.axis != 2) || norm.epsilon != 1e-5F)
    return {};
  if (!parameters(graph, slots) || !private_members(graph, group, slots)) return {};
  group.inputs.assign(slots.begin(), slots.begin() + Cat);
  group.outputs = {slots[Out]};
  group.integers = {std::bit_cast<std::uint32_t>(norm.epsilon)};
  return group;
}
}  // namespace kirari::gpu::zml
