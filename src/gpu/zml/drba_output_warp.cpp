#include "gpu/zml/drba_output_warp.h"

#include <algorithm>
#include <array>
#include <limits>

#include "gpu/ir/graph.h"
#include "gpu/zml/drba_resize.h"

namespace kirari::gpu::zml {
namespace {
using ir::DataType;
using ir::Dims;
using ir::OpKind;
using ir::TensorId;
using enum DataType;
constexpr TensorId missing = ir::kNoTensor;

class Finish {
 public:
  explicit Finish(const ir::Graph& g) : g_(g), index_(g), member_(g.nodes.size()) {}

  std::optional<MotionGroup> match() {
    if (!read_geometry()) return {};
    const auto out = g_.outputs.front();
    const auto sum = node(out, OpKind::Add, F16, 2);
    if (!sum) return {};
    const auto first = node(sum->inputs[0], OpKind::Mul, F16, 2);
    const auto last = node(sum->inputs[1], OpKind::Mul, F16, 2);
    if (!first || !last) return {};
    const auto mask = first->inputs[1];
    const auto inverse = node(last->inputs[1], OpKind::Sub, F16, 2);
    if (!inverse || inverse->inputs[1] != mask || !scalar(inverse->inputs[0], F16, 1)) return {};
    const auto logits = unary(mask, OpKind::Sigmoid, F16);
    const auto head = resize_head(logits, 4, 5);
    if (!tensor(head, F16, {1, 5, h_ / 4, w_ / 4})) return {};
    MotionGroup group{.op = "KirariDrbaOutputWarp", .inputs = {head}, .outputs = {out}};
    for (unsigned side = 0; side < 2; ++side) {
      const auto rounded = (side == 0 ? first : last)->inputs[0];
      if (!append_warp(group, head, side, rounded)) return {};
    }
    collect_members(group);
    return group;
  }

  TensorId projection(const MotionGroup& finish) {
    // Mesure CUDA à 720p, après centrage dans 1280 × 768. L'ancre de sortie a déjà reconnu
    // le mélange RGB des deux GridSample et toute la chaîne des cinq plans du dernier bloc.
    const Dims rgb{1, 3, 768, 1280};
    if (finish.op != "KirariDrbaOutputWarp" || finish.inputs.size() != 11 || g_.inputs.size() != 2 ||
        !tensor(g_.inputs[0], F16, {1, 7, 768, 1280}) || !tensor(g_.inputs[1], F16, {1, 7, 768, 1280}) ||
        !tensor(finish.inputs[1], F16, rgb) || slice(finish.inputs[1], 3, 6) != g_.inputs[0])
      return missing;
    const auto id = finish.inputs[6];
    const auto sum = node(id, OpKind::Add, F16, 2);
    if (!tensor(id, F16, rgb) || !sum) return missing;
    const auto first = node(sum->inputs[0], OpKind::Mul, F16, 2);
    const auto last = node(sum->inputs[1], OpKind::Mul, F16, 2);
    if (!first || !last || index_.readers(sum->inputs[0]).size() != 1 ||
        index_.readers(sum->inputs[1]).size() != 1 || slice(first->inputs[1], 0, 3) != g_.inputs[0] ||
        slice(last->inputs[1], 3, 6) != g_.inputs[1])
      return missing;
    const auto alpha = first->inputs[0];
    const auto complement = node(last->inputs[0], OpKind::Sub, F16, 2);
    if (!complement || !scalar(complement->inputs[0], F16, 1) || complement->inputs[1] != alpha ||
        !projection_alpha(alpha) || !projection_readers(id, finish))
      return missing;
    return id;
  }

 private:
  const ir::Graph& g_;
  ir::GraphIndex index_;
  std::vector<bool> member_;
  std::int64_t h_ = 0, w_ = 0;

  bool read_geometry() {
    if (g_.inputs.size() != 2 || g_.outputs.size() != 1 || g_.tensors[g_.inputs[0]].name != "pair01" ||
        g_.tensors[g_.inputs[1]].name != "pair12" || !ir::find_tensor(g_, "net.encode.cnn0.weight"))
      return false;
    const auto out = g_.outputs.front();
    const auto& d = g_.tensors[out].dims;
    if (d.size() != 4 || d[0] != 1 || d[1] != 3 || d[2] < 4 || d[3] < 4 || d[2] % 4 || d[3] % 4 ||
        d[2] > (1 << 23) || d[3] > (1 << 23) || d[2] > std::numeric_limits<std::int32_t>::max() / d[3])
      return false;
    h_ = d[2];
    w_ = d[3];
    for (const auto id : g_.inputs)
      if (g_.tensors[id].dims != Dims{1, 7, h_, w_}) return false;
    return true;
  }
  bool append_warp(MotionGroup& group, TensorId head, unsigned side, TensorId rounded) {
    const auto warp = node(unary(rounded, OpKind::Cast, F16), OpKind::GridSample, F32, 2);
    if (!warp) return false;
    const auto& attrs = std::get<ir::GridSampleAttrs>(warp->attrs);
    if (attrs.mode != ir::GridMode::Bilinear || attrs.padding != ir::GridPadding::Border ||
        !attrs.align_corners || attrs.grid_nchw)
      return false;
    const auto pixels = unary(warp->inputs[0], OpKind::Cast, F32);
    if (!tensor(pixels, F16, g_.tensors[group.outputs.front()].dims)) return false;
    group.inputs.push_back(pixels);
    auto xy = warp->inputs[1];
    while (const auto cast = node(xy, OpKind::Cast, F32, 1)) {
      xy = cast->inputs[0];
      if (xy == missing || g_.tensors[xy].type != F32) return false;
    }
    const auto trans = node(xy, OpKind::Transpose, F32, 1);
    if (!trans || std::get<ir::TransposeAttrs>(trans->attrs).perm != Dims{0, 2, 3, 1}) return false;
    const auto cat = node(trans->inputs[0], OpKind::Concat, F32, 2);
    if (!cat || std::get<ir::AxisAttrs>(cat->attrs).axis != 1) return false;
    for (unsigned axis = 0; axis < 2; ++axis)
      if (!append_coordinate(group, head, side, axis, cat->inputs[axis])) return false;
    return true;
  }
  bool append_coordinate(MotionGroup& group, TensorId head, unsigned side, unsigned axis, TensorId id) {
    const auto coord = node(id, OpKind::Add, F32, 2);
    if (!coord) return false;
    const auto disp = node(coord->inputs[1], OpKind::Mul, F32, 2);
    if (!disp || !tensor(disp->inputs[1], F32, {}) ||
        !tensor(coord->inputs[0], F32, axis == 0 ? Dims{1, 1, 1, w_} : Dims{1, 1, h_, 1}))
      return false;
    const auto plane = slice(unary(disp->inputs[0], OpKind::Cast, F32), axis, axis + 1);
    const auto begin = std::int64_t{2} * side;
    const auto scaled = node(slice(plane, begin, begin + 2), OpKind::Mul, F16, 2);
    if (!scaled || !scalar(scaled->inputs[1], F16, 4) || resize_head(scaled->inputs[0], 0, 4) != head)
      return false;
    group.inputs.push_back(coord->inputs[0]);
    group.inputs.push_back(disp->inputs[1]);
    return true;
  }
  void collect_members(MotionGroup& group) {
    // Une conversion de l'image peut aussi alimenter un étage précédent. Conserver cette
    // conversion et ses parents si un autre lecteur les utilise, même après l'ancre.
    for (const auto id : group.inputs) {
      const auto producer = index_.producer(id);
      if (producer != ir::kNoNode) member_[producer] = false;
    }
    for (std::size_t k = g_.nodes.size(); k-- > 0;) {
      if (!member_[k] || g_.nodes[k].outputs.front() == group.outputs.front()) continue;
      for (const auto reader : index_.readers(g_.nodes[k].outputs.front()))
        if (!member_[reader]) member_[k] = false;
    }
    for (std::size_t k = 0; k < member_.size(); ++k)
      if (member_[k]) group.members.push_back(k);
  }
  bool tensor(TensorId id, DataType ty, const Dims& dims) const {
    return id != missing && g_.tensors[id].type == ty && g_.tensors[id].dims == dims;
  }
  bool scalar(TensorId id, DataType ty, float value) const {
    return tensor(id, ty, {}) && g_.tensors[id].kind == ir::TensorKind::Constant &&
           g_.tensors[id].value.f.size() == 1 && g_.tensors[id].value.f[0] == value;
  }
  const ir::Node* node(TensorId id, OpKind op, DataType ty, std::size_t inputs) {
    if (id == missing || g_.tensors[id].type != ty) return nullptr;
    const auto k = index_.producer(id);
    if (k == ir::kNoNode) return nullptr;
    const auto& n = g_.nodes[k];
    if (n.op != op || n.outputs.size() != 1 || n.inputs.size() != inputs) return nullptr;
    member_[k] = true;
    return &n;
  }
  TensorId unary(TensorId id, OpKind op, DataType type) {
    const auto n = node(id, op, type, 1);
    return n ? n->inputs[0] : missing;
  }
  TensorId slice(TensorId id, std::int64_t begin, std::int64_t end, std::int64_t axis = 1) {
    const auto n = node(id, OpKind::Slice, F16, 5);
    if (!n) return missing;
    const std::array<std::int64_t, 4> expected{begin, end, axis, 1};
    for (std::size_t k = 0; k < expected.size(); ++k) {
      const auto value = ir::constant_operand(g_, *n, 1 + k);
      if (!value || value->type != I64 || value->i.size() != 1 || value->i[0] != expected[k]) return missing;
    }
    return n->inputs[0];
  }
  bool projection_alpha(TensorId id) {
    const auto clip = node(id, OpKind::Clip, F16, 3);
    if (!tensor(id, F16, {1, 1, 1, 1}) || !clip || !scalar(clip->inputs[1], F16, 0) ||
        !scalar(clip->inputs[2], F16, 1))
      return false;
    const auto scale = node(clip->inputs[0], OpKind::Mul, F16, 2);
    if (!scale || !scalar(scale->inputs[1], F16, 4096)) return false;
    const auto sub = node(scale->inputs[0], OpKind::Sub, F16, 2);
    if (!sub || !scalar(sub->inputs[0], F16, 1)) return false;
    const auto tau = node(sub->inputs[1], OpKind::Mul, F16, 2);
    if (!tau || !scalar(tau->inputs[1], F16, 2)) return false;
    // Le graphe original lit le premier pixel du plan temps de pair01.
    return slice(slice(slice(tau->inputs[0], 0, 1, 3), 0, 1, 2), 6, 7) == g_.inputs[0];
  }

  bool projection_readers(TensorId id, const MotionGroup& finish) {
    // Le Cast FP32 est partagé par les réductions /8 et /4 et par le GridSample du deuxième côté.
    if (index_.readers(id).size() != 1) return false;
    const auto& cast = g_.nodes[index_.readers(id).front()];
    if (cast.op != OpKind::Cast || cast.outputs.size() != 1 ||
        !tensor(cast.outputs[0], F32, {1, 3, 768, 1280}))
      return false;
    const auto& readers = index_.readers(cast.outputs[0]);
    if (readers.size() != 3) return false;
    int mask = 0;
    for (const auto k : readers) {
      const auto& n = g_.nodes[k];
      if (n.inputs.empty() || n.inputs[0] != cast.outputs[0] || n.outputs.size() != 1) return false;
      if (n.op == OpKind::GridSample && std::ranges::binary_search(finish.members, k)) {
        mask |= 1;
      } else if (n.op == OpKind::Resize) {
        const auto scales = ir::constant_operand(g_, n, 2);
        if (!scales || scales->f.size() != 4) return false;
        const int factor = drba_resize_factor(g_, n, scales->f);
        if (factor == 4) mask |= 2;
        if (factor == 8) mask |= 4;
      } else {
        return false;
      }
    }
    return mask == 7;
  }
  TensorId resize_head(TensorId rounded, std::int64_t begin, std::int64_t end) {
    const auto up = node(unary(rounded, OpKind::Cast, F16), OpKind::Resize, F32, 3);
    if (!up || up->inputs[1] != missing) return missing;
    const auto& attrs = std::get<ir::ResizeAttrs>(up->attrs);
    const auto scales = ir::constant_operand(g_, *up, 2);
    if (attrs.mode != ir::ResizeMode::Linear || attrs.coord != ir::CoordMode::HalfPixel ||
        !attrs.axes.empty() || !scales || scales->f.size() != 4 ||
        !std::ranges::equal(scales->f, std::array{1.F, 1.F, 4.F, 4.F}))
      return missing;
    return slice(unary(up->inputs[0], OpKind::Cast, F32), begin, end);
  }
};
}  // namespace

std::optional<MotionGroup> drba_output_warp(const ir::Graph& graph) {
  return Finish(graph).match();
}
ir::TensorId drba_shared_projection(const ir::Graph& graph, const MotionGroup& finish) {
  return Finish(graph).projection(finish);
}
}  // namespace kirari::gpu::zml
