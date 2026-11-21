#include "gpu/zml/gmfss_graph.h"

#include <algorithm>
#include <array>
#include <string_view>

#include "gpu/ir/graph.h"
#include "kernels/ref/half.h"

namespace kirari::gpu::zml {
namespace {
using ir::DataType;
using ir::Graph;
using ir::OpKind;
using ir::TensorId;
using ir::TensorKind;

bool weight(const Graph& graph, std::string_view name) {
  const auto id = ir::find_tensor(graph, name);
  return id && graph.tensors[*id].kind == TensorKind::Constant;
}

template <std::size_t N>
bool named(const Graph& graph, const std::vector<TensorId>& ids,
           const std::array<std::string_view, N>& names) {
  return ids.size() == N && std::ranges::equal(ids, names, {}, [&](TensorId id) -> std::string_view {
           return graph.tensors[id].name;
         });
}

ir::Node cast(const Graph& graph, TensorId from, TensorId to) {
  return {.name = graph.tensors[to].name + "/gmflow_f32",
          .op = OpKind::Cast,
          .attrs = ir::CastAttrs{.to = DataType::F32},
          .inputs = {from},
          .outputs = {to}};
}

TensorId temporary(Graph& graph, TensorId base, DataType type, std::string_view suffix) {
  auto tensor = graph.tensors[base];
  tensor.kind = TensorKind::Activation;
  tensor.type = type;
  tensor.name += suffix;
  return ir::add_tensor(graph, std::move(tensor));
}
}  // namespace

GmfssGraph gmfss_graph(const Graph& graph) {
  using namespace std::string_view_literals;
  static constexpr std::array pair_out{"feature0_1"sv, "motion0_1"sv, "source0_2"sv,  "motion0_2"sv,
                                       "source0_3"sv,  "motion0_3"sv, "feature1_1"sv, "motion1_1"sv,
                                       "source1_2"sv,  "motion1_2"sv, "source1_3"sv,  "motion1_3"sv,
                                       "half0"sv,      "half1"sv};
  static constexpr std::array base_in{"half0"sv,     "half1"sv,     "warped0_1"sv, "warped0_2"sv,
                                      "warped0_3"sv, "warped1_1"sv, "warped1_2"sv, "warped1_3"sv};
  const auto conv1 = ir::find_tensor(graph, "gmfss.flownet.backbone.conv1.weight");
  if (conv1 && graph.tensors[*conv1].kind == TensorKind::Constant &&
      named(graph, graph.inputs, std::array{"frames"sv}) && named(graph, graph.outputs, pair_out) &&
      std::ranges::any_of(graph.nodes, [](const auto& n) { return n.op == OpKind::LayerNorm; }))
    return graph.tensors[*conv1].type == DataType::F16 ? GmfssGraph::Pair : GmfssGraph::PairWideWeights;
  if (named(graph, graph.inputs, base_in) && named(graph, graph.outputs, std::array{"out"sv}) &&
      weight(graph, "gmfss.fusionnet.residual_model_head.1.weight"))
    return GmfssGraph::BaseFusion;
  return GmfssGraph::Other;
}

void widen_gmflow_intermediates(Graph& graph) {
  // La paire reconnue a une seule entrée. Garder son type ONNX lors du chargement,
  // même si le lecteur demande une autre précision pour son tampon externe.
  const TensorId input = graph.inputs.front();
  auto nodes = std::move(graph.nodes);
  graph.nodes.clear();
  graph.nodes.reserve(nodes.size());
  for (auto& t : graph.tensors) {
    if (t.type != DataType::F16 || t.kind == TensorKind::Input) continue;
    t.type = DataType::F32;
    if (t.kind == TensorKind::Constant) {
      t.value.type = DataType::F32;
      for (float& v : t.value.f) v = kernels::ref::half_to_float(kernels::ref::float_to_half(v));
    }
  }
  auto wide_input = input;
  if (graph.tensors[input].type == DataType::F16) {
    wide_input = temporary(graph, input, DataType::F32, "/gmflow_input");
    graph.nodes.push_back(cast(graph, input, wide_input));
  }
  for (auto& node : nodes) {
    std::ranges::replace(node.inputs, input, wide_input);
    if (node.op == OpKind::Cast && std::get<ir::CastAttrs>(node.attrs).to == DataType::F16) {
      const auto out = node.outputs.front();
      const auto rounded = temporary(graph, out, DataType::F16, "/gmflow_half");
      node.outputs.front() = rounded;
      graph.nodes.push_back(std::move(node));
      graph.nodes.push_back(cast(graph, rounded, out));
    } else {
      graph.nodes.push_back(std::move(node));
    }
  }
}
}  // namespace kirari::gpu::zml
