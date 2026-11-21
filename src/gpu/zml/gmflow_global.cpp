#include "gpu/zml/gmflow_global.h"

#include <algorithm>
#include <array>

#include "gpu/ir/graph.h"

namespace kirari::gpu::zml {
namespace {
using ir::Dims;
using ir::Graph;
using ir::OpKind;
using ir::TensorId;

enum Port { Q, K, Divisor, Coords, Target5, Target3, Scores, AsGrid, Scaled, Flat, Reverse, Both, Prob, Out };
using Slots = std::array<TensorId, Out + 1>;
struct Stage {
  std::string_view name;
  OpKind op;
  Port left;
  std::optional<Port> right;
  Dims dims;
};
const std::array<Stage, 8> kStages{{
    {"/flownet/MatMul", OpKind::MatMul, Q, K, {1, 2160, 2160}},
    {"/flownet/Reshape_12", OpKind::Reshape, Scores, Target5, {1, 36, 60, 36, 60}},
    {"/flownet/Div_16", OpKind::Div, AsGrid, Divisor, {1, 36, 60, 36, 60}},
    {"/flownet/Reshape_18", OpKind::Reshape, Scaled, Target3, {1, 2160, 2160}},
    {"/flownet/Transpose_7", OpKind::Transpose, Flat, {}, {1, 2160, 2160}},
    {"/flownet/Concat_26", OpKind::Concat, Flat, Reverse, {2, 2160, 2160}},
    {"/flownet/Softmax", OpKind::Softmax, Both, {}, {2, 2160, 2160}},
    {"/flownet/MatMul_1", OpKind::MatMul, Prob, Coords, {2, 2160, 2}},
}};

bool bind(Slots& slots, Port p, TensorId id) {
  if (id == ir::kNoTensor || (slots[p] != ir::kNoTensor && slots[p] != id)) return false;
  slots[p] = id;
  return true;
}

bool parameters(const Graph& graph, const Slots& slots) {
  constexpr std::size_t width = 60;
  constexpr std::size_t pixels = 36 * width;
  constexpr std::size_t points = 2 * pixels;
  const auto& q = graph.tensors[slots[Q]];
  const auto& k = graph.tensors[slots[K]];
  const auto& div = graph.tensors[slots[Divisor]];
  const auto& xy = graph.tensors[slots[Coords]];
  if (q.type != ir::DataType::F32 || q.dims != Dims{1, 2160, 128} || k.type != ir::DataType::F32 ||
      k.dims != Dims{1, 128, 2160})
    return false;
  if (div.kind != ir::TensorKind::Constant || div.type != ir::DataType::F32 || !div.dims.empty() ||
      div.value.f.size() != 1 || div.value.f.front() != 11.3125F)
    return false;
  if (xy.kind != ir::TensorKind::Constant || xy.type != ir::DataType::F32 || xy.dims != Dims{2, 2160, 2} ||
      xy.value.f.size() != points * 2)
    return false;
  for (std::size_t n = 0; n < points; ++n) {
    const auto column = n % width;
    const auto row = (n % pixels) / width;
    if (xy.value.f[2 * n] != static_cast<float>(column) || xy.value.f[2 * n + 1] != static_cast<float>(row))
      return false;
  }
  for (const Port p : {Target5, Target3}) {
    const auto& target = graph.tensors[slots[p]];
    if (target.kind != ir::TensorKind::Constant || target.type != ir::DataType::I64) return false;
  }
  return true;
}

bool links(const Graph& graph, const MotionGroup& group, const Slots& slots) {
  ir::GraphIndex index(graph);
  auto produced_by = [&](Port p, std::string_view name) {
    auto producer = index.producer(slots[p]);
    return producer != ir::kNoNode && graph.nodes[producer].name == name;
  };
  if (!produced_by(Q, "/flownet/Transpose_5") || !produced_by(K, "/flownet/Reshape_11")) return false;
  const auto reader = index.sole_reader(slots[Out]);
  if (!reader || graph.nodes[*reader].name != "/flownet/Reshape_19") return false;
  for (std::size_t p = Scores; p < Out; ++p) {
    if (ir::is_graph_output(graph, slots[p])) return false;
    for (auto r : index.readers(slots[p]))
      if (!std::ranges::binary_search(group.members, r)) return false;
  }
  return true;
}
}  // namespace

std::optional<MotionGroup> gmflow_global_match(const Graph& graph) {
  if (graph.inputs.size() != 1 || graph.outputs.size() != 14) return {};
  const auto& frames = graph.tensors[graph.inputs.front()];
  if (frames.name != "frames" || frames.dims != Dims{1, 6, 1152, 1920}) return {};
  MotionGroup group{.op = "KirariGmflowMatch"};
  Slots slots;
  slots.fill(ir::kNoTensor);
  auto next = graph.nodes.begin();
  for (std::size_t j = 0; j < kStages.size(); ++j) {
    const auto& stage = kStages[j];
    const auto node =
        std::find_if(next, graph.nodes.end(), [&](const auto& n) { return n.name == stage.name; });
    if (node == graph.nodes.end() || node->op != stage.op || node->outputs.size() != 1 ||
        node->inputs.size() != (stage.right ? 2U : 1U))
      return {};
    if (!bind(slots, stage.left, node->inputs[0]) ||
        (stage.right && !bind(slots, *stage.right, node->inputs[1])))
      return {};
    const auto id = node->outputs.front();
    if (graph.tensors[id].type != ir::DataType::F32 || graph.tensors[id].dims != stage.dims) return {};
    slots[Scores + j] = id;
    group.members.push_back(static_cast<std::size_t>(node - graph.nodes.begin()));
    next = std::next(node);
  }
  const auto& perm = std::get<ir::TransposeAttrs>(graph.nodes[group.members[4]].attrs).perm;
  const auto cat = std::get<ir::AxisAttrs>(graph.nodes[group.members[5]].attrs).axis;
  const auto axis = std::get<ir::NormAttrs>(graph.nodes[group.members[6]].attrs).axis;
  if (perm != Dims{0, 2, 1} || cat != 0 || (axis != -1 && axis != 2)) return {};
  if (!parameters(graph, slots) || !links(graph, group, slots)) return {};
  group.inputs = {slots[Q], slots[K], slots[Divisor], slots[Coords]};
  group.outputs = {slots[Out]};
  return group;
}
}  // namespace kirari::gpu::zml
