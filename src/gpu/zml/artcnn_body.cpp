#include "gpu/zml/artcnn_body.h"

#include <algorithm>
#include <array>
#include <span>

#include "gpu/ir/graph.h"
#include "prepare/sha256.h"

namespace kirari::gpu::zml {
namespace {
using ir::DataType;
using ir::Graph;
using ir::Node;
using ir::OpKind;
using ir::TensorId;

constexpr std::string_view kC4F16 = "0c9f39abf2d0a8cc899192771761cd3ad865aa3d0db1f95417e85c1f89c8b8fe";
constexpr std::string_view kC4F32 = "0f14a4cc990c5a00544fd8b939c35e2607d19ae872c4c4f31bc6e6b2dd643c12";
constexpr std::size_t kFirst = 5, kLast = 16;
constexpr std::array kOps{OpKind::Cast, OpKind::Conv, OpKind::Concat,       OpKind::Sub,  OpKind::Resize,
                          OpKind::Conv, OpKind::Conv, OpKind::Relu,         OpKind::Conv, OpKind::Relu,
                          OpKind::Conv, OpKind::Relu, OpKind::Conv,         OpKind::Relu, OpKind::Conv,
                          OpKind::Add,  OpKind::Conv, OpKind::DepthToSpace, OpKind::Clip, OpKind::Concat,
                          OpKind::Add,  OpKind::Clip, OpKind::Cast};

bool convolution_weights(const Graph& g, const Node& n, int inputs, int outputs) {
  if (n.inputs.size() != 3) return false;
  const auto& w = g.tensors[n.inputs[1]];
  const auto& b = g.tensors[n.inputs[2]];
  return w.type == DataType::F32 && b.type == DataType::F32 && w.kind == ir::TensorKind::Constant &&
         b.kind == ir::TensorKind::Constant && w.dims == ir::Dims{outputs, inputs, 3, 3} &&
         b.dims == ir::Dims{outputs};
}

bool body_edges(const Graph& g, int channels) {
  const auto output = [&](std::size_t i) { return g.nodes[i].outputs.front(); };
  const auto readers = ir::consumers(g);
  for (std::size_t i = kFirst; i <= kLast; ++i) {
    const auto& n = g.nodes[i];
    if (n.inputs.empty() || n.inputs[0] != output(i == kFirst ? 1 : i - 1)) return false;
    const auto id = output(i);
    if (g.tensors[id].type != DataType::F32 || ir::is_graph_output(g, id)) return false;
    if (readers[id] !=
        (i == kFirst ? std::vector<std::size_t>{i + 1, kLast - 1} : std::vector<std::size_t>{i + 1}))
      return false;
    if (n.op == OpKind::Conv) {
      const auto inputs = i == kFirst ? 1 : channels;
      const auto outputs = i == kLast ? 4 : channels;
      if (!convolution_weights(g, n, inputs, outputs)) return false;
    } else if (n.op == OpKind::Add) {
      if (n.inputs.size() != 2 || n.inputs[1] != output(kFirst)) return false;
    } else if (n.inputs.size() != 1) {
      return false;
    }
  }
  return g.nodes[kLast + 1].inputs == std::vector<TensorId>{output(kLast)};
}

TensorId half_tensor(Graph& g, TensorId id) {
  const auto& t = g.tensors[id];
  return ir::add_tensor(g, {.name = t.name + "/ArtCNN_FP16",
                            .kind = ir::TensorKind::Activation,
                            .type = DataType::F16,
                            .dims = t.dims,
                            .known = t.known});
}

Node convert(std::string name, TensorId src, TensorId dst, DataType type) {
  return {.name = std::move(name),
          .op = OpKind::Cast,
          .attrs = ir::CastAttrs{.to = type},
          .inputs = {src},
          .outputs = {dst}};
}

void rewrite(Graph& g) {
  std::vector<TensorId> halves(g.tensors.size(), ir::kNoTensor);
  std::vector<Node> nodes(g.nodes.begin(), g.nodes.begin() + kFirst);
  const auto luma = g.nodes[1].outputs.front();
  halves[luma] = half_tensor(g, luma);
  nodes.push_back(convert("ArtCNN_FP16/luminance", luma, halves[luma], DataType::F16));
  for (auto n : std::span(g.nodes).subspan(kFirst, kLast - kFirst + 1)) {
    // Garder les constantes FP32 d’origine dans le manifeste. XLA plie ces Cast statiques.
    if (n.op == OpKind::Conv) {
      for (auto id : std::span(n.inputs).subspan(1)) {
        halves[id] = half_tensor(g, id);
        nodes.push_back(convert(n.name + "/fp16", id, halves[id], DataType::F16));
      }
    }
    for (auto& id : n.inputs) id = halves[id];
    const auto id = n.outputs.front();
    n.outputs.front() = halves[id] = half_tensor(g, id);
    nodes.push_back(std::move(n));
  }
  const auto last = g.nodes[kLast].outputs.front();
  nodes.push_back(convert("ArtCNN_FP16/retour_fp32", halves[last], last, DataType::F32));
  nodes.insert(nodes.end(), g.nodes.begin() + kLast + 1, g.nodes.end());
  g.nodes = std::move(nodes);
}
}  // namespace

bool use_artcnn_half_body(Graph& g, const ModelSpec& model) {
  if (model.precision != TensorType::F16 || (model.sha256 != kC4F16 && model.sha256 != kC4F32)) return false;
  if (g.inputs.size() != 1 || g.outputs.size() != 1 ||
      g.tensors[g.inputs[0]].dims != ir::Dims{1, 3, 1080, 1920} ||
      g.tensors[g.outputs[0]].dims != ir::Dims{1, 3, 2160, 3840})
    return false;
  if (!std::ranges::equal(g.nodes, kOps, {}, &Node::op) ||
      !std::ranges::all_of(g.nodes, [](const Node& n) { return n.outputs.size() == 1; }))
    return false;
  if (!body_edges(g, model.sha256 == kC4F16 ? 16 : 32)) return false;
  const auto actual = prepare::sha256_file(model.onnx_path);
  if (!actual || *actual != model.sha256) return false;
  rewrite(g);
  return true;
}
}  // namespace kirari::gpu::zml
