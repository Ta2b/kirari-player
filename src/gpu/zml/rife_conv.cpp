#include "gpu/zml/rife_conv.h"

#include <algorithm>
#include <optional>

#include "gpu/ir/graph.h"
#include "gpu/ir/passes.h"

namespace kirari::gpu::zml {
namespace {
using ir::OpKind;
using ir::TensorId;

bool channel_axis(std::int64_t axis) {
  return axis == 1 || axis == -3;  // NCHW
}

bool scalar_is(const ir::Value* v, std::int64_t value) {
  return v && v->i.size() == 1 && v->i.front() == value;
}

TensorId add_constant(ir::Graph& g, ir::Tensor tensor) {
  while (ir::find_tensor(g, tensor.name)) tensor.name += '_';
  return ir::add_tensor(g, std::move(tensor));
}

// Le dernier canal exclusif lu par des tranches ONNX. Les indices négatifs changeraient de
// sens dans le tenseur raccourci ; seules les tranches positives du canal sont admises ici.
std::optional<std::int64_t> slice_end(const ir::Graph& g, const ir::Node& n, TensorId input) {
  if (n.op != OpKind::Slice || n.inputs[0] != input) return {};
  const auto* start = ir::constant_operand(g, n, 1);
  const auto* end = ir::constant_operand(g, n, 2);
  const auto* axis = ir::constant_operand(g, n, 3);
  const auto* step = ir::constant_operand(g, n, 4);
  if (!start || !end || !axis || start->i.size() != 1 || end->i.size() != 1 || axis->i.size() != 1 ||
      !channel_axis(axis->i[0]) || start->i[0] < 0 || end->i[0] <= start->i[0] ||
      (n.inputs.size() > 4 && n.inputs[4] != ir::kNoTensor && !scalar_is(step, 1)))
    return {};
  return end->i[0];
}

// Une constante partagée par d'autres blocs garde sa valeur d'origine.
TensorId narrowed(ir::Graph& g, TensorId id, std::size_t axis, std::int64_t length) {
  const auto& original = g.tensors[id];
  auto dims = original.dims;
  dims[axis] = length;
  auto value = ir::make_value(original.type, dims);
  const std::size_t inner =
      ir::element_count(Dims(dims.begin() + static_cast<std::ptrdiff_t>(axis) + 1, dims.end()));
  const auto stride = static_cast<std::size_t>(original.dims[axis]) * inner;
  const auto keep = static_cast<std::size_t>(length) * inner;
  for (std::size_t start = 0, dest = 0; dest < value.f.size(); start += stride, dest += keep)
    std::copy_n(original.value.f.begin() + static_cast<std::ptrdiff_t>(start), keep,
                value.f.begin() + static_cast<std::ptrdiff_t>(dest));
  return add_constant(g, {.name = original.name + "-canaux-utiles",
                          .kind = ir::TensorKind::Constant,
                          .type = original.type,
                          .dims = std::move(dims),
                          .known = true,
                          .value = std::move(value)});
}
struct HeadChannels {
  std::int64_t used = 0;
  ir::Node* split = nullptr;
  std::size_t split_count = 0;
};

std::optional<HeadChannels> live_head_channels(ir::Graph& g, ir::GraphIndex& index, TensorId output) {
  const auto users = index.readers(output);
  if (users.empty()) return {};
  const auto total = g.tensors[output].dims[1];
  std::int64_t used = 0;
  ir::Node* split = nullptr;
  std::size_t split_count = 0;
  if (users.size() == 1 && g.nodes[users[0]].op == OpKind::Split) {
    split = &g.nodes[users[0]];
    if (!channel_axis(std::get<ir::SplitAttrs>(split->attrs).axis)) return {};
    split_count = split->outputs.size();
    while (split_count) {
      const auto last = split->outputs[split_count - 1];
      if (ir::is_graph_output(g, last) || !index.readers(last).empty()) break;
      --split_count;
    }
    for (std::size_t i = 0; i < split_count; ++i) used += g.tensors[split->outputs[i]].dims[1];
  } else {
    for (const auto i : users) {
      const auto end = slice_end(g, g.nodes[i], output);
      if (!end) {
        used = total;
        break;
      }
      used = std::max(used, *end);
    }
  }
  if (used <= 0 || used >= total) return {};
  return HeadChannels{used, split, split_count};
}
}  // namespace

void prune_motion_heads(ir::Graph& g) {
  ir::GraphIndex index(g);
  bool changed = false;
  for (auto& conv : g.nodes) {
    if (conv.op != OpKind::ConvTranspose || std::get<ir::ConvTransposeAttrs>(conv.attrs).group != 1) continue;
    const auto* w = ir::constant_operand(g, conv, 1);
    if (!w || w->dims.size() != 4 || (w->type != ir::DataType::F16 && w->type != ir::DataType::F32)) continue;
    const bool biased = conv.inputs.size() > 2 && conv.inputs[2] != ir::kNoTensor;
    if (biased && !ir::constant_operand(g, conv, 2)) continue;
    const auto shuffle = index.sole_reader(conv.outputs[0]);
    if (!shuffle) continue;
    auto& pixel = g.nodes[*shuffle];
    if (pixel.op != OpKind::DepthToSpace || ir::is_graph_output(g, pixel.outputs[0])) continue;
    const auto params = std::get<ir::BlockAttrs>(pixel.attrs);
    if (params.mode != ir::BlockMode::Crd || params.blocksize != 2) continue;
    const auto output = pixel.outputs[0];
    const auto channels = live_head_channels(g, index, output);
    if (!channels) continue;
    const auto& [used, split, split_count] = *channels;
    if (split) {
      split->outputs.resize(split_count);
      auto sizes = ir::make_value(ir::DataType::I64, {static_cast<std::int64_t>(split_count)});
      for (std::size_t i = 0; i < split_count; ++i) sizes.i[i] = g.tensors[split->outputs[i]].dims[1];
      const auto lens = add_constant(g, {.name = split->name + "-tailles-utiles",
                                         .kind = ir::TensorKind::Constant,
                                         .type = ir::DataType::I64,
                                         .dims = sizes.dims,
                                         .known = true,
                                         .value = std::move(sizes)});
      split->inputs = {output, lens};
      std::get<ir::SplitAttrs>(split->attrs).num_outputs = 0;
    }
    conv.inputs[1] = narrowed(g, conv.inputs[1], 1, 4 * used);
    if (biased) conv.inputs[2] = narrowed(g, conv.inputs[2], 0, 4 * used);
    g.tensors[conv.outputs[0]].dims[1] = 4 * used;
    g.tensors[output].dims[1] = used;
    index.invalidate();
    changed = true;
  }
  if (changed) ir::compact(g);
}

#ifdef KIRARI_ZML_POLYPHASE
Value transpose_conv_4x4(Hlo& h, const Value& x, const Value& w) {
  const auto n = x.dims[0], in = x.dims[1], out = w.dims[1], height = x.dims[2], width = x.dims[3];
  // Pour la phase (r,s) d'une sortie sans recadrage, les coefficients de position
  // (r+2i,s+2j) lisent l'entrée (y-i,x-j). Un seul noyau dense produit les quatre phases.
  auto phases = h.reshape(w, {in, out, 2, 2, 2, 2});
  auto ordered = h.transpose(phases, {1, 3, 5, 0, 2, 4});
  auto joined = h.reshape(ordered, {out * 4, in, 2, 2});
  auto kernel = h.op("reverse", {joined}, joined.dims, joined.element, "dimensions = array<i64: 2, 3>");
  auto low = h.op("convolution", {x, kernel}, {n, out * 4, height + 1, width + 1}, x.element,
                  "window_strides = array<i64: 1, 1>, padding = dense<1> : tensor<2x2xi64>, "
                  "dimension_numbers = #stablehlo.conv<[b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1]>, "
                  "feature_group_count = 1 : i64, batch_group_count = 1 : i64");
  auto separate = h.reshape(low, {n, out, 2, 2, height + 1, width + 1});
  auto interleaved =
      h.reshape(h.transpose(separate, {0, 1, 4, 2, 5, 3}), {n, out, 2 * height + 2, 2 * width + 2});
  return h.slice(interleaved, {0, 0, 1, 1}, {n, out, 2 * height + 1, 2 * width + 1}, {1, 1, 1, 1});
}
#endif
}  // namespace kirari::gpu::zml
