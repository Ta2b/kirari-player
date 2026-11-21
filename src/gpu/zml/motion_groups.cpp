#include "gpu/zml/motion_groups.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <optional>

#include "gpu/ir/graph.h"
#include "gpu/zml/rife_sampling_plan.h"
#ifdef KIRARI_ZML_FOLD_RESCONV
#include "kernels/ref/half.h"
#endif

namespace kirari::gpu::zml {
namespace {
using ir::Node;
using ir::OpKind;
using ir::TensorId;
using Affine = std::array<TensorId, 2>;

class Match {
 public:
  Match(const ir::Graph& graph, ir::GraphIndex& index) : g_(graph), index_(index) {}

  std::optional<MotionGroup> block(TensorId output) {
    auto x = output;
    const auto* shuffle = take(x, OpKind::DepthToSpace);
    if (!shuffle) return {};
    const auto& a = std::get<ir::BlockAttrs>(shuffle->attrs);
    if (a.blocksize != 2 || a.mode != ir::BlockMode::Crd) return {};
    auto head = transposed(x);
    if (!head) return {};
    const auto c = g_.tensors[(*head)[0]].dims[1];
    if (c != 20 && c != 24 && c != 52) return {};

    std::array<Unit, 8> body{};
    std::int64_t scales = 0;
    for (std::size_t i = body.size(); i-- > 0;) {
      if (!residual(x, body[i])) return {};
      if (body[i].beta != ir::kNoTensor) scales |= std::int64_t{1} << i;
    }
    std::array<Affine, 2> down{};
    for (std::size_t i = down.size(); i-- > 0;) {
      if (!leaky(x)) return {};
      auto conv = direct(x, 2);
      if (!conv) return {};
      down[i] = *conv;
    }
    MotionGroup group{.op = "KirariIFBlock", .inputs = {x}, .outputs = {output}, .integers = {scales, 0}};
    for (const auto& layer : down) append(group, layer);
    for (const auto& unit : body) {
      append(group, unit.conv);
      if (unit.beta != ir::kNoTensor) group.inputs.push_back(unit.beta);
    }
    append(group, *head);
    return finish(std::move(group));
  }

  std::optional<MotionGroup> encoder(TensorId output) {
    auto x = output;
    auto head = transposed(x);
    if (!head) return {};
    const auto c = g_.tensors[(*head)[0]].dims[1];
    if (c != 4 && c != 8 && c != 16) return {};
    std::array<Affine, 3> layers{};
    for (std::size_t i = layers.size(); i-- > 0;) {
      if (!leaky(x)) return {};
      auto conv = direct(x, i == 0 ? 2 : 1);
      if (!conv) return {};
      layers[i] = *conv;
    }
    if (g_.tensors[x].dims[1] != 3) return {};
    MotionGroup group{.op = "KirariEncoder", .inputs = {x}, .outputs = {output}, .integers = {0}};
    for (const auto& layer : layers) append(group, layer);
    append(group, *head);
    return finish(std::move(group));
  }

 private:
  struct Unit {
    Affine conv;
    TensorId beta = ir::kNoTensor;
  };
  bool residual(TensorId& x, Unit& unit) {
    if (!leaky(x)) return false;
    TensorId skip = ir::kNoTensor;
    if (const auto* add = take(x, OpKind::Add)) {
      const auto first = add->inputs[0], second = add->inputs[1];
      // La branche résiduelle peut elle-même produire un Mul ; on vérifie la connexion
      // explicite à l'entrée de la convolution avant d'accepter le motif.
      const bool left = producer_is(first, OpKind::Mul) && scaled_conv_input(first) == second;
      x = left ? first : second;
      skip = left ? second : first;
      const auto* mul = take(x, OpKind::Mul);
      if (!mul) return false;
      const bool first_conv = producer_is(mul->inputs[0], OpKind::Conv);
      unit.beta = mul->inputs[first_conv ? 1 : 0];
      x = mul->inputs[first_conv ? 0 : 1];
      if (!producer_is(x, OpKind::Conv)) return false;
      const auto& beta = g_.tensors[unit.beta];
      if (beta.kind != ir::TensorKind::Constant || beta.dims != ir::Dims{1, g_.tensors[x].dims[1], 1, 1})
        return false;
    }
    auto conv = direct(x, 1);
    if (!conv || (skip != ir::kNoTensor && skip != x)) return false;
    unit.conv = *conv;
    return true;
  }
  static void append(MotionGroup& group, const Affine& layer) {
    group.inputs.insert(group.inputs.end(), layer.begin(), layer.end());
  }
  bool producer_is(TensorId id, OpKind op) {
    const auto i = index_.producer(id);
    return i != ir::kNoNode && g_.nodes[i].op == op;
  }
  TensorId scaled_conv_input(TensorId id) {
    const auto& mul = g_.nodes[index_.producer(id)];
    for (auto v : mul.inputs)
      if (producer_is(v, OpKind::Conv)) return g_.nodes[index_.producer(v)].inputs[0];
    return ir::kNoTensor;
  }
  const Node* take(TensorId& id, OpKind op) {
    const auto i = index_.producer(id);
    if (i == ir::kNoNode || g_.nodes[i].op != op) return nullptr;
    members_.push_back(i);
    const auto& n = g_.nodes[i];
    id = n.inputs.front();
    return &n;
  }
  bool leaky(TensorId& x) {
    const auto* activation = take(x, OpKind::LeakyRelu);
    return activation && std::get<ir::LeakyReluAttrs>(activation->attrs).alpha == 0.2F;
  }
  bool parameters(const Node& n, std::int64_t side) const {
    if (n.inputs.size() != 3 || n.inputs[2] == ir::kNoTensor) return false;
    const auto* w = ir::constant_operand(g_, n, 1);
    const auto* b = ir::constant_operand(g_, n, 2);
    return w && b && w->dims.size() == 4 && b->dims.size() == 1 && w->dims[2] == side && w->dims[3] == side;
  }
  std::optional<Affine> direct(TensorId& x, std::int64_t stride) {
    const auto* n = take(x, OpKind::Conv);
    if (!n || !parameters(*n, 3)) return {};
    const auto& a = std::get<ir::ConvAttrs>(n->attrs);
    if (a.auto_pad != ir::AutoPad::NotSet || a.group != 1 || a.strides != ir::Pair{stride, stride} ||
        a.dilations != ir::Pair{1, 1} || a.pads != ir::Quad{1, 1, 1, 1} ||
        a.epilogue.act != ir::Activation::None || a.epilogue.residual || a.epilogue.gate ||
        a.epilogue.space || a.in_quant != 0.0F || a.out_quant != 0.0F)
      return {};
    return Affine{n->inputs[1], n->inputs[2]};
  }
  std::optional<Affine> transposed(TensorId& x) {
    const auto* n = take(x, OpKind::ConvTranspose);
    if (!n || !parameters(*n, 4)) return {};
    const auto& a = std::get<ir::ConvTransposeAttrs>(n->attrs);
    if (a.auto_pad != ir::AutoPad::NotSet || a.group != 1 || a.strides != ir::Pair{2, 2} ||
        a.dilations != ir::Pair{1, 1} || a.pads != ir::Quad{1, 1, 1, 1} || !a.output_shape.empty() ||
        a.output_padding != ir::Pair{})
      return {};
    return Affine{n->inputs[1], n->inputs[2]};
  }
  std::optional<MotionGroup> finish(MotionGroup group) {
    std::ranges::sort(members_);
    for (auto i : members_) {
      const auto id = g_.nodes[i].outputs[0];
      if (id == group.outputs[0]) continue;
      if (ir::is_graph_output(g_, id)) return {};
      for (auto reader : index_.readers(id))
        if (!std::ranges::binary_search(members_, reader)) return {};
    }
    group.members = std::move(members_);
    return group;
  }
  const ir::Graph& g_;
  ir::GraphIndex& index_;
  std::vector<std::size_t> members_;
};
enum class RifeVersion { Other, Lite422, Full425 };

// Les deux RIFE livrés ont un encodeur 16→4 commun et des largeurs d'IFBlock distinctes.
// La signature porte sur les groupes reconnus et la vraie entrée, y compris les ONNX dérivés.
RifeVersion delivered_rife(const ir::Graph& graph, const std::vector<MotionGroup>& groups) {
  if (graph.inputs.size() != 1) return RifeVersion::Other;
  const auto& input = graph.tensors[graph.inputs.front()];
  if (input.type != ir::DataType::F16 || input.dims.size() != 4 || input.dims[0] != 1 ||
      (input.dims[1] != 11 && compact_rife_grid(graph).empty()))
    return RifeVersion::Other;
  ir::Dims widths;
  std::size_t encoders = 0;
  for (const auto& group : groups) {
    if (group.op == "KirariEncoder") {
      if (graph.tensors[group.inputs[group.inputs.size() - 2]].dims != ir::Dims{16, 4, 4, 4})
        return RifeVersion::Other;
      ++encoders;
    } else if (group.op == "KirariIFBlock") {
      const auto& weight = graph.tensors[group.inputs[1]].dims;
      if (weight[1] != (widths.empty() ? 15 : 28)) return RifeVersion::Other;
      widths.push_back(weight[0]);
    }
  }
  if (encoders == 1 && widths == ir::Dims{96, 64, 32, 16}) return RifeVersion::Lite422;
  if (encoders == 1 && widths == ir::Dims{96, 64, 48, 32, 16}) return RifeVersion::Full425;
  return RifeVersion::Other;
}
struct D2Graph {
  const ir::Graph& g;
  ir::GraphIndex& idx;

  bool fp16(TensorId id, const ir::Dims& dims) const {
    const auto& t = g.tensors[id];
    return t.type == ir::DataType::F16 && t.dims == dims;
  }
  const ir::Node* parent(TensorId id, OpKind op) const {
    const auto p = idx.producer(id);
    return p != ir::kNoNode && g.nodes[p].op == op ? &g.nodes[p] : nullptr;
  }
  TensorId uncast(TensorId id) const {
    const auto* cast = parent(id, OpKind::Cast);
    return cast ? cast->inputs[0] : id;
  }
  // Les poids distribués utilisent Resize FP16 ; certains exports l'entourent de deux
  // Cast. Dans les deux cas l'entrée et la sortie du calcul bilinéaire sont arrondies en FP16.
  const ir::Node* resize(TensorId out, float scale) const {
    const auto inner = uncast(out);
    const auto* n = parent(inner, OpKind::Resize);
    if (!n || g.tensors[out].type != ir::DataType::F16) return nullptr;
    const auto in = n->inputs.front();
    const bool promoted = inner != out;
    if (promoted && (g.tensors[inner].type != ir::DataType::F32 || g.tensors[in].type != ir::DataType::F32 ||
                     !parent(in, OpKind::Cast) || g.tensors[uncast(in)].type != ir::DataType::F16))
      return nullptr;
    if (!promoted && g.tensors[in].type != ir::DataType::F16) return nullptr;
    const auto& a = std::get<ir::ResizeAttrs>(n->attrs);
    const auto* factors = ir::constant_operand(g, *n, 2);
    if (a.mode != ir::ResizeMode::Linear || a.coord != ir::CoordMode::HalfPixel ||
        (!a.axes.empty() && a.axes != ir::Dims{0, 1, 2, 3}) || !factors ||
        factors->f != std::vector<float>{1.F, 1.F, scale, scale})
      return nullptr;
    return n;
  }
  bool coarse_planes(const std::vector<TensorId>& parts, TensorId coarse, std::int64_t h,
                     std::int64_t w) const {
    constexpr std::array<int, 6> channels{6, 8, 1, 1, 8, 4};
    if (parts.size() != channels.size()) return false;
    for (std::size_t k = 0; k < parts.size(); ++k)
      if (!fp16(parts[k], {1, channels[k], h, w})) return false;
    const auto* split = parent(parts[3], OpKind::Split);
    if (!split || std::get<ir::SplitAttrs>(split->attrs).axis != 1 || split->outputs.size() != 3 ||
        split->outputs[1] != parts[3] || split->outputs[2] != parts[4] ||
        !fp16(split->outputs[0], {1, 4, h, w}) || !fp16(split->inputs[0], {1, 13, h, w}))
      return false;
    const auto* up = resize(split->inputs[0], 4.F);
    if (!up || (up->inputs[0] != coarse && uncast(up->inputs[0]) != coarse)) return false;

    return true;
  }
};

}  // namespace

std::optional<MotionGroup> rife_d2_centre(const ir::Graph& g, const std::vector<MotionGroup>& groups,
                                          bool resize_phases) {
  if (delivered_rife(g, groups) == RifeVersion::Other) return {};
  const auto& frame = g.tensors[g.inputs.front()].dims;
  const auto h = frame[2], w = frame[3];
  if (frame[1] != 11 || h < 8 || w < 8 || h % 8 || w % 8) return {};
  std::vector<const MotionGroup*> blocks;
  for (const auto& group : groups)
    if (group.op == "KirariIFBlock") blocks.push_back(&group);
  const auto& d2 = *blocks[blocks.size() - 2];
  const auto coarse = blocks[blocks.size() - 3]->outputs.front();
  const auto target = d2.inputs.front();
  ir::GraphIndex idx(g);
  const D2Graph view{g, idx};
  if (!view.fp16(coarse, {1, 13, h / 4, w / 4}) || !view.fp16(target, {1, 28, h / 2, w / 2}) ||
      !view.fp16(d2.inputs[1], {32, 28, 3, 3}))
    return {};
  const auto* down = view.resize(target, 0.5F);
  if (!down) return {};
  const auto warp = compact_warp_plan(g, *down, 0.5F, 0.5F, true);
  if (!warp || warp->factor != 2 || !view.fp16(view.uncast(warp->warp.inputs[0]), {2, 7, h, w})) return {};
  const auto& join = warp->join;
  const auto& parts = join.inputs;
  if (!view.coarse_planes(parts, coarse, h, w)) return {};

  // N'absorber que la chaîne privée Concat → [Cast] → Resize → [Cast] du bloc D2.
  // Le Resize ×4 en amont et le warp gardent leurs autres lecteurs, en particulier le flot.
  if (idx.sole_reader(target) != d2.members.front()) return {};
  MotionGroup result{.op = "KirariRifeD2Centre",
                     .inputs = {parts[0], parts[1], parts[2], coarse, parts[5]},
                     .outputs = {target},
                     .integers = {resize_phases}};
  for (auto out = target;;) {
    const auto p = idx.producer(out);
    result.members.push_back(p);
    if (out == join.outputs[0]) break;
    out = g.nodes[p].inputs[0];
    if (idx.sole_reader(out) != p) return {};
  }
  std::ranges::reverse(result.members);
  return result;
}

#ifdef KIRARI_ZML_RIFE422_WARP_XY
ir::TensorId rife422_half_warp(const ir::Graph& graph, const std::vector<MotionGroup>& groups) {
  if (delivered_rife(graph, groups) != RifeVersion::Lite422) return ir::kNoTensor;
  for (const auto& n : graph.nodes) {
    if (n.op != OpKind::Resize) continue;
    const auto plan = compact_warp_plan(graph, n, 0.5F, 0.5F, true);
    if (!plan) continue;
    const auto& warp = plan->warp;
    const auto& attrs = std::get<ir::GridSampleAttrs>(warp.attrs);
    const auto& src = graph.tensors[warp.inputs.front()];
    const auto& planes = graph.tensors[plan->join.outputs.front()].dims;
    if (src.type == ir::DataType::F32 && src.dims.size() == 4 && src.dims[0] == 2 && src.dims[1] == 7 &&
        graph.tensors[warp.outputs.front()].dims == src.dims &&
        planes == ir::Dims{1, 28, src.dims[2], src.dims[3]} && attrs.mode == ir::GridMode::Bilinear &&
        attrs.padding == ir::GridPadding::Border && attrs.align_corners)
      return warp.outputs.front();
  }
  return ir::kNoTensor;
}
#endif

namespace {
std::vector<TensorId> expanded_grid_planes(const ir::Graph& graph, ir::GraphIndex& index,
                                           const ir::Node& concat, std::int64_t height, std::int64_t width) {
  const ir::Dims plane{1, 1, height, width};
  const std::array<ir::Dims, 4> expected{ir::Dims{width}, {1, 1, height, 1}, {}, {}};
  std::vector<ir::TensorId> planes;
  for (std::size_t k = 0; k < expected.size(); ++k) {
    const auto out = concat.inputs[k + 1];
    const auto at = index.producer(out);
    if (at == ir::kNoNode) return {};
    const auto& expand = graph.nodes[at];
    if (expand.op != OpKind::Expand || expand.inputs.size() != 2) return {};
    const auto& from = graph.tensors[expand.inputs[0]];
    const auto& shape = graph.tensors[expand.inputs[1]];
    const auto& dst = graph.tensors[out];
    if (from.type != ir::DataType::F16 || from.dims != expected[k] || dst.dims != plane ||
        dst.type != ir::DataType::F16 || shape.kind != ir::TensorKind::Constant || shape.value.i != plane ||
        ir::storage_bytes(dst) <= ir::storage_bytes(from) + ir::storage_bytes(shape))
      return {};
    if (from.kind != ir::TensorKind::Constant) {
      // L'axe Y de rang 1 est orienté par Reshape, plié ensuite en [1,1,H,1].
      const auto vertical = index.producer(expand.inputs[0]);
      if (k != 1 || vertical == ir::kNoNode) return {};
      const auto& reshape = graph.nodes[vertical];
      if (reshape.op != OpKind::Reshape || !ir::constant_operand(graph, reshape, 0) ||
          graph.tensors[reshape.inputs[0]].dims != ir::Dims{height})
        return {};
    }
    planes.push_back(out);
  }
  return planes;
}
}  // namespace

std::vector<ir::TensorId> compact_rife_grid(const ir::Graph& graph) {
  if (graph.inputs.size() != 1) return {};
  const auto input = graph.inputs.front();
  const auto& tensor = graph.tensors[input];
  if (tensor.type != ir::DataType::F16 || tensor.dims.size() != 4 || tensor.dims[0] != 1 ||
      tensor.dims[1] != 7)
    return {};
  const auto eleven = ir::find_tensor(graph, "kirari_rife_grid/eleven_planes");
  if (!eleven) return {};
  ir::GraphIndex index(graph);
  const auto producer = index.producer(*eleven);
  const auto reader = index.sole_reader(*eleven);
  if (producer == ir::kNoNode || !reader) return {};
  const auto& concat = graph.nodes[producer];
  const auto& split = graph.nodes[*reader];
  if (concat.op != OpKind::Concat || std::get<ir::AxisAttrs>(concat.attrs).axis != 1 ||
      concat.inputs.size() != 5 || concat.inputs.front() != input || split.op != OpKind::Split ||
      std::get<ir::SplitAttrs>(split.attrs).axis != 1 || split.outputs.size() != 4)
    return {};
  const auto height = tensor.dims[2], width = tensor.dims[3];
  constexpr std::array<std::int64_t, 4> channels{6, 1, 2, 2};
  for (std::size_t k = 0; k < channels.size(); ++k) {
    const auto& part = graph.tensors[split.outputs[k]];
    if (part.type != ir::DataType::F16 || part.dims != ir::Dims{1, channels[k], height, width}) return {};
  }
  return expanded_grid_planes(graph, index, concat, height, width);
}

std::vector<MotionGroup> motion_groups(const ir::Graph& graph) {
  ir::GraphIndex index(graph);
  std::vector<MotionGroup> groups;
  for (const auto& n : graph.nodes) {
    Match match(graph, index);
    if (n.op == OpKind::ConvTranspose) {
      if (auto encoder = match.encoder(n.outputs[0])) groups.push_back(std::move(*encoder));
    } else if (n.op == OpKind::DepthToSpace) {
      if (auto block = match.block(n.outputs[0])) groups.push_back(std::move(*block));
    }
  }
  return groups;
}

#ifdef KIRARI_ZML_FOLD_RESCONV
namespace {
struct Folded {
  TensorId w, b;
  std::vector<float> weights, bias;
};
std::optional<Folded> folded_residual(const ir::Graph& graph, ir::GraphIndex& index, const MotionGroup& group,
                                      std::size_t i) {
  const auto wid = group.inputs[i], bid = group.inputs[i + 1];
  const auto& w = graph.tensors[wid];
  const auto& b = graph.tensors[bid];
  const auto& beta = graph.tensors[group.inputs[i + 2]];
  const auto c = w.dims[0];
  const auto reader = index.sole_reader(wid);
  if (!reader || index.sole_reader(bid) != reader || w.type != ir::DataType::F16 || b.type != w.type ||
      beta.type != w.type || w.dims != ir::Dims{c, c, 3, 3} || b.dims != ir::Dims{c} ||
      beta.dims != ir::Dims{1, c, 1, 1})
    return {};
  const auto& conv = graph.nodes[*reader];
  const auto& x = graph.tensors[conv.inputs.front()];
  const auto& y = graph.tensors[conv.outputs.front()];
  if (x.type != w.type || y.type != w.type || x.dims != y.dims) return {};
  const auto channels = static_cast<std::size_t>(c);
  const auto row = channels * 9;
  Folded p{wid, bid, w.value.f, b.value.f};
  for (std::size_t o = 0; o < channels; ++o) {
    for (std::size_t k = 0; k < row; ++k) {
      const auto scaled = kernels::ref::round_to_half(w.value.f[o * row + k] * beta.value.f[o]);
      // Le HLO témoin contient déjà βW arrondi en FP16. Seul le centre diagonal change.
      p.weights[o * row + k] = k == o * 9 + 4 ? kernels::ref::round_to_half(scaled + 1.0F) : scaled;
    }
    p.bias[o] = kernels::ref::round_to_half(b.value.f[o] * beta.value.f[o]);
  }
  const auto finite = [](float v) { return std::isfinite(v); };
  if (!std::ranges::all_of(p.weights, finite) || !std::ranges::all_of(p.bias, finite)) return {};
  return p;
}
}  // namespace

// x + β(W*x + b) = (I + βW)*x + βb. Le centre OIHW est aussi le pixel courant aux
// bords : Match::direct exige pas/dilatation 1, bord 1 et groupe 1 pour ces huit couches.
// Les arrondis FP16 changent ; les sorties de lecture entière qualifient cette recette.
void fold_motion_residuals(ir::Graph& graph, std::vector<MotionGroup>& groups) {
  if (delivered_rife(graph, groups) == RifeVersion::Other) return;
  ir::GraphIndex index(graph);
  std::vector<Folded> pending;
  for (const auto& group : groups) {
    if (group.op != "KirariIFBlock") continue;
    if (group.integers[0] != 255 || group.inputs.size() != 31) return;
    for (std::size_t i = 5; i < 29; i += 3) {
      auto layer = folded_residual(graph, index, group, i);
      if (!layer) return;
      pending.push_back(std::move(*layer));
    }
  }
  // Mutation atomique à l'échelle du modèle. Chaque poids remplacé est lu uniquement
  // dans son groupe. Les huit paires remplacent les huit triplets (poids, biais, bêta).
  for (auto& p : pending) {
    graph.tensors[p.w].value.f = std::move(p.weights);
    graph.tensors[p.b].value.f = std::move(p.bias);
  }
  for (auto& group : groups) {
    if (group.op != "KirariIFBlock") continue;
    auto& args = group.inputs;
    for (std::size_t i = 0; i < 8; ++i) {
      args[5 + 2 * i] = args[5 + 3 * i];
      args[6 + 2 * i] = args[6 + 3 * i];
    }
    args[21] = args[29];
    args[22] = args[30];
    args.resize(23);
    group.integers[0] = 0;
    if (group.integers.size() == 2) group.integers.push_back(2);  // Select seulement dans le corps.
  }
}
#endif

MotionRecipe configure_motion_groups(const ir::Graph& graph, std::vector<MotionGroup>& groups,
                                     RifeRecipes recipes) {
  const auto version = delivered_rife(graph, groups);
  const bool lite422 = recipes.lite422 && version == RifeVersion::Lite422;
  const bool full425 = recipes.full425 && version == RifeVersion::Full425;
  MotionRecipe phases{.transposed = lite422 || full425, .resize = lite422};
  if (recipes.d2 != RifeD2Centre::Disabled && version != RifeVersion::Other) {
    const auto& dims = graph.tensors[graph.inputs.front()].dims;
    // Les deux tailles de lecture qualifiées incluent le padding du réseau.
    phases.d2_centre =
        recipes.d2 == RifeD2Centre::Experimental
            ? dims[1] == 11 && dims[2] >= 8 && dims[3] >= 8 && dims[2] % 8 == 0 && dims[3] % 8 == 0
            : version == RifeVersion::Lite422 &&
                  (dims == ir::Dims{1, 11, 768, 1280} || dims == ir::Dims{1, 11, 1088, 1920});
  }
#ifdef KIRARI_ZML_POLYPHASE
  phases.transposed = true;
#endif
#ifdef KIRARI_ZML_RESIZE_PHASES
  phases.resize = true;
#endif
  for (auto& group : groups) {
    // Encodeur : [polyphase, select?]. IFBlock : [masque des résiduels, polyphase, select?].
    const std::size_t index = group.op == "KirariEncoder" ? 0 : 1;
    group.integers.resize(index + 1);
    group.integers[index] = phases.transposed;
#ifdef KIRARI_ZML_RIFE_SELECT_ACTIVATION
    if (version != RifeVersion::Other) group.integers.push_back(1);
#endif
  }
  return phases;
}
}  // namespace kirari::gpu::zml
