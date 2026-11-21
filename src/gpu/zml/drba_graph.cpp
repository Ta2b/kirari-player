#include "gpu/zml/drba_graph.h"

#include <algorithm>
#include <array>
#include <optional>
#include <tuple>

#include "gpu/ir/graph.h"
#include "gpu/zml/drba_warps.h"

namespace kirari::gpu::zml {
namespace {
using ir::Graph;
using ir::Node;
using ir::OpKind;
using ir::TensorId;

constexpr std::array kLayers{OpKind::Conv, OpKind::LeakyRelu, OpKind::Conv,         OpKind::LeakyRelu,
                             OpKind::Conv, OpKind::LeakyRelu, OpKind::ConvTranspose};
using Chain = std::array<std::size_t, kLayers.size()>;

// L'identité des poids et des biais est vérifiée par leurs identifiants de tenseurs.
// Les epilogues de la passe Vulkan n'interviennent pas sur l'IR ONNX d'entrée.
bool same_layer(const Node& a, const Node& b) {
  if (!std::ranges::equal(std::span(a.inputs).subspan(1), std::span(b.inputs).subspan(1))) return false;
  if (a.op == OpKind::LeakyRelu)
    return std::get<ir::LeakyReluAttrs>(a.attrs).alpha == std::get<ir::LeakyReluAttrs>(b.attrs).alpha;
  if (a.op == OpKind::Conv) {
    auto params = [](const Node& n) {
      const auto& c = std::get<ir::ConvAttrs>(n.attrs);
      const auto& e = c.epilogue;
      return std::tie(c.kernel, c.strides, c.dilations, c.pads, c.auto_pad, c.group, c.in_quant, c.out_quant,
                      c.quant_grid, e.residual, e.act, e.alpha, e.lo, e.hi, e.gate, e.space, e.space_mode);
    };
    return params(a) == params(b);
  }
  auto params = [](const Node& n) {
    const auto& c = std::get<ir::ConvTransposeAttrs>(n.attrs);
    return std::tie(c.kernel, c.strides, c.dilations, c.pads, c.output_padding, c.output_shape, c.auto_pad,
                    c.group);
  };
  return params(a) == params(b);
}

std::optional<Chain> encoder(const Graph& graph, ir::GraphIndex& index, std::size_t first) {
  Chain layers{};
  std::size_t at = first;
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const auto& n = graph.nodes[at];
    if (n.op != kLayers[i] || n.outputs.size() != 1 || n.inputs.empty()) return std::nullopt;
    layers[i] = at;
    if (i + 1 < layers.size()) {
      const auto next = index.sole_reader(n.outputs[0]);
      if (!next || graph.nodes[*next].inputs[0] != n.outputs[0]) return std::nullopt;
      at = *next;
    }
  }
  return layers;
}

bool fits_batch(const Graph& graph, TensorId id) {
  const auto& tensor = graph.tensors[id];
  return tensor.known && tensor.dims.size() == 4 && tensor.dims[0] == 1 &&
         ir::element_count(tensor.dims) <= ir::kMaxElements / 3;
}
bool matches_batch(const Graph& graph, const Chain& layers, const Chain* reference) {
  for (std::size_t i = 0; i < layers.size(); ++i) {
    const auto& op = graph.nodes[layers[i]];
    if (!fits_batch(graph, op.outputs[0])) return false;
    if (reference) {
      const auto& other = graph.nodes[(*reference)[i]];
      const auto& x = graph.tensors[op.inputs[0]];
      const auto& ref = graph.tensors[other.inputs[0]];
      if (x.dims != ref.dims || x.type != ref.type || !same_layer(op, other)) return false;
    }
  }
  return true;
}

TensorId batch_tensor(Graph& graph, TensorId single) {
  auto tensor = graph.tensors[single];
  tensor.name += "/lot-de-trois";
  tensor.kind = ir::TensorKind::Activation;
  tensor.value = {};
  tensor.dims[0] = 3;
  return ir::add_tensor(graph, std::move(tensor));
}
}  // namespace

bool optimize_drba_graph(ir::Graph& graph) {
  if (graph.inputs.size() != 2 || graph.tensors[graph.inputs[0]].name != "pair01" ||
      graph.tensors[graph.inputs[1]].name != "pair12")
    return false;
  const auto first_weight = ir::find_tensor(graph, "net.encode.cnn0.weight");
  if (!first_weight) return false;
  ir::GraphIndex index(graph);
  if (index.readers(*first_weight).size() != 3) return false;
  std::vector<Chain> chains;
  for (auto first : index.readers(*first_weight)) {
    const auto& n = graph.nodes[first];
    if (n.inputs.size() < 2 || n.inputs[1] != *first_weight) return false;
    auto layers = encoder(graph, index, first);
    if (!layers || !fits_batch(graph, n.inputs[0])) return false;
    if (!matches_batch(graph, *layers, chains.empty() ? nullptr : &chains.front())) return false;
    chains.push_back(*layers);
  }

  // Les trois entrées doivent exister avant la première opération qui lit l'un des encodages.
  // Les branches peuvent être entrelacées dans un ONNX re-sérialisé.
  const auto insert = std::max({chains[0].back(), chains[1].back(), chains[2].back()});
  std::vector<TensorId> images, features;
  std::vector<bool> replace(graph.nodes.size(), false);
  for (const auto& layers : chains) {
    images.push_back(graph.nodes[layers.front()].inputs[0]);
    const auto output = graph.nodes[layers.back()].outputs[0];
    for (const auto reader : index.readers(output))
      if (reader <= insert) return false;
    features.push_back(output);
    for (const auto n : layers) replace[n] = true;
  }
  auto batch = batch_tensor(graph, images.front());
  std::vector<Node> batched;
  batched.push_back({.name = "DRBA/lot/images",
                     .op = OpKind::Concat,
                     .attrs = ir::AxisAttrs{0},
                     .inputs = std::move(images),
                     .outputs = {batch}});
  for (const auto which : chains.front()) {
    Node layer = graph.nodes[which];
    layer.inputs[0] = batch;
    batch = batch_tensor(graph, layer.outputs[0]);
    layer.outputs = {batch};
    layer.name += "/lot";
    batched.push_back(std::move(layer));
  }
  batched.push_back({.name = "DRBA/lot/encodages",
                     .op = OpKind::Split,
                     .attrs = ir::SplitAttrs{.axis = 0, .num_outputs = 3},
                     .inputs = {batch},
                     .outputs = std::move(features)});
  std::vector<Node> nodes;
  for (std::size_t i = 0; i < graph.nodes.size(); ++i) {
    if (!replace[i]) nodes.push_back(std::move(graph.nodes[i]));
    if (i == insert)
      for (auto& n : batched) nodes.push_back(std::move(n));
  }
  graph.nodes = std::move(nodes);
  combine_drba_warps(graph);
  ir::compact(graph);
  return true;
}
}  // namespace kirari::gpu::zml
