#include "gpu/zml/drba_warps.h"

#include <algorithm>
#include <bit>
#include <tuple>

#include "gpu/ir/graph.h"
#include "gpu/ir/passes.h"

namespace kirari::gpu::zml {
namespace {
using ir::Graph;
using ir::Node;
using ir::OpKind;
using ir::TensorId;

bool same_coordinate_op(const Node& n, const Node& m) {
  switch (n.op) {
    case OpKind::Cast:
      return std::get<ir::CastAttrs>(n.attrs).to == std::get<ir::CastAttrs>(m.attrs).to;
    case OpKind::Transpose:
      return std::get<ir::TransposeAttrs>(n.attrs).perm == std::get<ir::TransposeAttrs>(m.attrs).perm;
    case OpKind::Concat:
      return std::get<ir::AxisAttrs>(n.attrs).axis == std::get<ir::AxisAttrs>(m.attrs).axis;
    case OpKind::Add:
    case OpKind::Mul:
    case OpKind::Slice:
      return true;
    default:
      return false;
  }
}

// Les deux branches ONNX construisent séparément la grille normalisée. On ne joint les lectures
// que si leurs opérations, arrondis et constantes coïncident jusqu'au même flot.
bool same_coordinates(const Graph& g, ir::GraphIndex& index, TensorId a, TensorId b, unsigned depth = 0) {
  if (a == b) return true;
  if (depth > 10 || a == ir::kNoTensor || b == ir::kNoTensor) return false;
  const auto& x = g.tensors[a];
  const auto& y = g.tensors[b];
  if (x.kind != y.kind || x.type != y.type || x.dims != y.dims) return false;
  if (x.kind == ir::TensorKind::Constant) {
    return x.value.i == y.value.i && std::ranges::equal(x.value.f, y.value.f, [](float l, float r) {
             return std::bit_cast<std::uint32_t>(l) == std::bit_cast<std::uint32_t>(r);
           });
  }
  if (x.kind != ir::TensorKind::Activation) return false;
  const Node& n = g.nodes[index.producer(a)];
  const Node& m = g.nodes[index.producer(b)];
  if (n.op != m.op || n.inputs.size() != m.inputs.size() || n.outputs.size() != 1 || m.outputs.size() != 1)
    return false;
  if (!same_coordinate_op(n, m)) return false;
  for (std::size_t i = 0; i < n.inputs.size(); ++i)
    if (!same_coordinates(g, index, n.inputs[i], m.inputs[i], depth + 1)) return false;
  return true;
}
bool compatible(const Graph& g, ir::GraphIndex& index, std::size_t first, std::size_t second) {
  const auto& a = g.nodes[first];
  const auto& b = g.nodes[second];
  const auto mode = [](const Node& n) {
    const auto& p = std::get<ir::GridSampleAttrs>(n.attrs);
    return std::tie(p.mode, p.padding, p.align_corners, p.grid_nchw);
  };
  if (mode(a) != mode(b)) return false;
  const auto& x = g.tensors[a.inputs[0]];
  const auto& y = g.tensors[b.inputs[0]];
  if (x.type != y.type || x.dims.size() != 4 || y.dims.size() != 4 || x.dims[0] != y.dims[0] ||
      x.dims[2] != y.dims[2] || x.dims[3] != y.dims[3])
    return false;
  if (!((x.dims[1] == 3 && y.dims[1] == 16) || (x.dims[1] == 16 && y.dims[1] == 3))) return false;
  for (const auto id : {a.inputs[0], a.outputs[0]}) {
    auto dims = g.tensors[id].dims;
    dims[1] = x.dims[1] + y.dims[1];
    if (!ir::bounded_dims(dims)) return false;
  }
  auto source = index.producer(b.inputs[0]);
  if (source != ir::kNoNode && source >= first) return false;
  return same_coordinates(g, index, a.inputs[1], b.inputs[1]);
}
TensorId joint(Graph& graph, TensorId id, std::int64_t channels) {
  auto tensor = graph.tensors[id];
  tensor.kind = ir::TensorKind::Activation;
  tensor.value = {};
  tensor.name += "/rgb-et-caracteristiques";
  tensor.dims[1] = channels;
  return ir::add_tensor(graph, std::move(tensor));
}
void join(Graph& g, std::size_t first, std::size_t second) {
  Node warp = g.nodes[first];
  const auto& other = g.nodes[second];
  const TensorId a = warp.inputs[0], b = other.inputs[0], out_a = warp.outputs[0], out_b = other.outputs[0];
  const auto ca = g.tensors[a].dims[1], cb = g.tensors[b].dims[1];
  const auto src = joint(g, a, ca + cb), dst = joint(g, out_a, ca + cb);
  const auto sizes = ir::add_tensor(g, {.name = warp.name + "/canaux",
                                        .kind = ir::TensorKind::Constant,
                                        .type = ir::DataType::I64,
                                        .dims = {2},
                                        .known = true,
                                        .value = {.type = ir::DataType::I64, .dims = {2}, .i = {ca, cb}}});
  warp.inputs[0] = src;
  warp.outputs = {dst};
  g.nodes.erase(g.nodes.begin() + static_cast<std::ptrdiff_t>(second));
  g.nodes[first] = {.name = warp.name + "/entrées",
                    .op = OpKind::Concat,
                    .attrs = ir::AxisAttrs{1},
                    .inputs = {a, b},
                    .outputs = {src}};
  auto next = g.nodes.insert(g.nodes.begin() + static_cast<std::ptrdiff_t>(first + 1), std::move(warp));
  g.nodes.insert(next + 1, Node{.name = next->name + "/sorties",
                                .op = OpKind::Split,
                                .attrs = ir::SplitAttrs{.axis = 1},
                                .inputs = {dst, sizes},
                                .outputs = {out_a, out_b}});
}
}  // namespace
int combine_drba_warps(Graph& graph) {
  int joined = 0;
  ir::GraphIndex index(graph);
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    if (graph.nodes[i].op != OpKind::GridSample) continue;
    for (std::size_t j = i + 1; j < graph.nodes.size(); ++j) {
      if (graph.nodes[j].op != OpKind::GridSample || !compatible(graph, index, i, j)) continue;
      join(graph, i, j);
      index.invalidate();
      ++joined;
      break;
    }
  }
  if (joined) ir::eliminate_dead_code(graph);
  return joined;
}
}  // namespace kirari::gpu::zml
