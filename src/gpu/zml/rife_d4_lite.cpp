#include <algorithm>
#include <array>

#include "gpu/ir/graph.h"
#include "gpu/zml/motion_groups.h"

namespace kirari::gpu::zml {
namespace {
using ir::DataType;
using ir::Dims;
using ir::OpKind;
using ir::TensorId;

constexpr int h = 768, w = 1280;

bool shaped(const ir::Graph& g, TensorId id, DataType type, const Dims& dims) {
  return g.tensors[id].type == type && g.tensors[id].dims == dims;
}

const MotionGroup* lite_stage(const ir::Graph& g, const std::vector<MotionGroup>& groups) {
  if (g.inputs.size() != 1 || g.outputs.size() != 2 || groups.size() != 6 ||
      !shaped(g, g.inputs[0], DataType::F16, {1, 11, h, w}) ||
      !shaped(g, g.outputs[0], DataType::F16, {1, 4, h, w}) ||
      !shaped(g, g.outputs[1], DataType::F16, {1, 1, h, w}))
    return nullptr;
  const auto& enc = groups[0];
  if (enc.op != "KirariEncoder" || enc.outputs.size() != 1 ||
      !shaped(g, enc.outputs.front(), DataType::F16, {2, 4, h, w}))
    return nullptr;
  constexpr std::array<int, 5> widths{96, 64, 48, 32, 12}, divisors{32, 16, 8, 4, 1};
  for (std::size_t k = 0; k < widths.size(); ++k) {
    const auto& b = groups[k + 1];
    const int c = k ? 28 : 15;
    if (b.op != "KirariIFBlock" || b.inputs.size() < 2 || b.members.empty() ||
        !shaped(g, b.inputs[1], DataType::F16, {widths[k], c, 3, 3}) ||
        !shaped(g, b.inputs[0], DataType::F16, {1, c, h / divisors[k], w / divisors[k]}))
      return nullptr;
  }
  return &groups[4];
}

// Chaque valeur intermédiaire de cette chaîne privée a un seul lecteur. Les
// autres branches, notamment le flot cumulé réutilisé au D1, restent vivantes.
class D4Chain {
 public:
  D4Chain(const ir::Graph& graph, const MotionGroup& block)
      : g_(graph), idx_(graph), value_(block.inputs.front()), reader_(block.members.front()) {}

  const ir::Node* take(OpKind op, DataType type, const Dims& dims) {
    if (!shaped(g_, value_, type, dims) || idx_.sole_reader(value_) != reader_) return nullptr;
    const auto i = idx_.producer(value_);
    if (i == ir::kNoNode) return nullptr;
    const auto& n = g_.nodes[i];
    if (n.op != op || n.inputs.empty() || n.outputs.size() != 1) return nullptr;
    nodes.push_back(i);
    value_ = n.inputs.front();
    reader_ = i;
    return &n;
  }
  std::vector<std::size_t> nodes;

 private:
  const ir::Graph& g_;
  ir::GraphIndex idx_;
  TensorId value_;
  std::size_t reader_;
};

bool quarter(const ir::Graph& graph, const ir::Node& resize) {
  const auto& a = std::get<ir::ResizeAttrs>(resize.attrs);
  const auto* scales = ir::constant_operand(graph, resize, 2);
  return a.mode == ir::ResizeMode::Linear && a.coord == ir::CoordMode::HalfPixel &&
         (a.axes.empty() || a.axes == Dims{0, 1, 2, 3}) && scales &&
         scales->f == std::vector<float>{1.F, 1.F, .25F, .25F};
}

bool packed_channels(const ir::Graph& g, const ir::Node& join) {
  constexpr std::array<int, 4> channels{14, 1, 9, 4};
  if (join.inputs.size() != channels.size() || std::get<ir::AxisAttrs>(join.attrs).axis != 1) return false;
  for (std::size_t k = 0; k < channels.size(); ++k)
    if (!shaped(g, join.inputs[k], DataType::F16, {1, channels[k], h, w})) return false;
  return true;
}
}  // namespace

std::optional<MotionGroup> rife425_lite_d4(const ir::Graph& g, const std::vector<MotionGroup>& groups) {
  const auto* block = lite_stage(g, groups);
  if (!block) return {};
  D4Chain path(g, *block);
  using enum ir::DataType;
  if (!path.take(OpKind::Cast, F16, {1, 28, h / 4, w / 4})) return {};
  const auto* down = path.take(OpKind::Resize, F32, {1, 28, h / 4, w / 4});
  if (!down || !quarter(g, *down) || !path.take(OpKind::Cast, F32, {1, 28, h, w})) return {};
  const auto* join = path.take(OpKind::Concat, F16, {1, 28, h, w});
  if (!join || !packed_channels(g, *join) || !path.take(OpKind::Reshape, F16, {1, 14, h, w}) ||
      !path.take(OpKind::Cast, F16, {2, 7, h, w}))
    return {};
  const auto* warp = path.take(OpKind::GridSample, F32, {2, 7, h, w});
  if (!warp || warp->inputs.size() != 2 || !shaped(g, warp->inputs[0], F32, {2, 7, h, w}) ||
      !shaped(g, warp->inputs[1], F32, {2, h, w, 2}))
    return {};
  const auto& a = std::get<ir::GridSampleAttrs>(warp->attrs);
  if (a.mode != ir::GridMode::Bilinear || a.padding != ir::GridPadding::Border || !a.align_corners ||
      a.grid_nchw)
    return {};
  std::ranges::reverse(path.nodes);
  return MotionGroup{
      .op = "KirariRife425LiteD4",
      .inputs = {warp->inputs[0], warp->inputs[1], join->inputs[1], join->inputs[2], join->inputs[3]},
      .outputs = {block->inputs[0]},
      .members = std::move(path.nodes)};
}
}  // namespace kirari::gpu::zml
