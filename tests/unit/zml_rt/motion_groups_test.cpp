#include "gpu/zml/motion_groups.h"

#include <array>

#include "../gpu_ir/test_util.h"
#include "gpu/ir/graph.h"

using namespace kirari::gpu::ir;
using namespace kirari::gpu::ir::test;
using kirari::gpu::zml::motion_groups;
using kirari::gpu::zml::MotionGroup;
using kirari::gpu::zml::MotionRecipe;
using kirari::gpu::zml::RifeD2Centre;
using kirari::gpu::zml::RifeRecipes;

namespace {
ModelBuilder ifblock(bool scaled) {
  ModelBuilder m;
  m.input("x", {1, 2, 8, 8})
      .weight("w", {2, 2, 3, 3}, sequence(36))
      .weight("b", {2}, sequence(2))
      .weight("beta", {1, 2, 1, 1}, {0.5F, -0.75F})
      .weight("head", {2, 24, 4, 4}, sequence(768))
      .weight("headb", {24}, sequence(24));
  std::string x = "x";
  for (int i = 0; i < 10; ++i) {
    const auto tag = std::to_string(i);
    const auto stride = i < 2 ? 2 : 1;
    auto pre = "conv" + tag;
    m.node("Conv", {x, "w", "b"}, {pre},
           {attr_ints("pads", {1, 1, 1, 1}), attr_ints("strides", {stride, stride})});
    if (scaled && i >= 2) {
      m.node("Mul", {"beta", pre}, {"multiply" + tag}).node("Add", {x, "multiply" + tag}, {"sum" + tag});
      pre = "sum" + tag;
    }
    x = "out" + tag;
    m.node("LeakyRelu", {pre}, {x}, {attr_float("alpha", 0.2F)});
  }
  m.node("ConvTranspose", {x, "head", "headb"}, {"up"},
         {attr_ints("pads", {1, 1, 1, 1}), attr_ints("strides", {2, 2})})
      .node("DepthToSpace", {"up"}, {"result"}, {attr_int("blocksize", 2), attr_string("mode", "CRD")})
      .output("result");
  return m;
}

void add_compact_grid(Graph& graph, TensorId input, const Dims& split_channels) {
  const auto height = graph.tensors[input].dims[2], width = graph.tensors[input].dims[3];
  const Dims plane{1, 1, height, width};
  const auto shape = add_tensor(graph, {.kind = TensorKind::Constant,
                                        .type = DataType::I64,
                                        .dims = {4},
                                        .known = true,
                                        .value = {.type = DataType::I64, .dims = {4}, .i = plane}});
  std::vector<TensorId> joined{input};
  for (const auto& dims : std::array<Dims, 4>{Dims{width}, {1, 1, height, 1}, {}, {}}) {
    const auto src =
        add_tensor(graph, {.kind = TensorKind::Constant, .type = DataType::F16, .dims = dims, .known = true});
    const auto dst = add_tensor(graph, {.type = DataType::F16, .dims = plane, .known = true});
    graph.nodes.push_back({.op = OpKind::Expand, .inputs = {src, shape}, .outputs = {dst}});
    joined.push_back(dst);
  }
  const auto eleven = add_tensor(graph, {.name = "kirari_rife_grid/eleven_planes",
                                         .type = DataType::F16,
                                         .dims = {1, 11, height, width},
                                         .known = true});
  graph.nodes.push_back(
      {.op = OpKind::Concat, .attrs = AxisAttrs{.axis = 1}, .inputs = joined, .outputs = {eleven}});
  std::vector<TensorId> split;
  for (auto c : split_channels)
    split.push_back(add_tensor(graph, {.type = DataType::F16, .dims = {1, c, height, width}, .known = true}));
  graph.nodes.push_back(
      {.op = OpKind::Split, .attrs = SplitAttrs{.axis = 1}, .inputs = {eleven}, .outputs = split});
}

// Attributs du manifeste après reconnaissance des convolutions et des huit unités de chaque bloc.
void phases_for(Dims widths, int planes, RifeRecipes recipes, MotionRecipe expected, const Dims& grid = {},
                DataType input_type = DataType::F16, Pair spatial = {192, 320}) {
  Graph graph;
  const auto input = add_tensor(
      graph, {.kind = TensorKind::Input, .type = input_type, .dims = {1, planes, spatial[0], spatial[1]}});
  graph.inputs = {input};
  if (!grid.empty()) add_compact_grid(graph, input, grid);
  const auto weight = [&](Dims shape) {
    return add_tensor(graph, {.kind = TensorKind::Constant, .type = DataType::F16, .dims = std::move(shape)});
  };
  const auto head = weight({16, 4, 4, 4});
  std::vector<MotionGroup> groups{{.op = "KirariEncoder", .inputs = {input, head, head}, .integers = {0}}};
  for (auto width : widths) {
    const auto w = weight({width, groups.size() == 1 ? 15 : 28, 3, 3});
    groups.push_back({.op = "KirariIFBlock", .inputs = {input, w}, .integers = {255, 0}});
  }
  const auto phases = configure_motion_groups(graph, groups, recipes);
#ifdef KIRARI_ZML_POLYPHASE
  expected.transposed = true;
#endif
#ifdef KIRARI_ZML_RESIZE_PHASES
  expected.resize = true;
#endif
  CHECK(phases.transposed == expected.transposed);
  CHECK(phases.resize == expected.resize);
  CHECK(phases.d2_centre == expected.d2_centre);
  CHECK(groups.front().integers.at(0) == static_cast<int>(expected.transposed));
  for (std::size_t k = 1; k < groups.size(); ++k) {
    CHECK(groups[k].integers.at(0) == 255);
    CHECK(groups[k].integers.at(1) == static_cast<int>(expected.transposed));
  }
}
}  // namespace

TEST_CASE("ZML : la reconnaissance des blocs RIFE vérifie le résiduel et les sorties observables",
          "[zml][rife]") {
  for (bool residual : {false, true}) {
    auto m = ifblock(residual);
    const auto g = build(m);
    const auto groups = motion_groups(g);
    REQUIRE(groups.size() == 1);
    const auto& group = groups.front();
    CHECK(group.op == "KirariIFBlock");
    CHECK(group.inputs.size() == (residual ? 31 : 23));
    CHECK(group.integers[0] == (residual ? 255 : 0));
    CHECK(group.members.size() == g.nodes.size());
    CHECK(group.outputs == g.outputs);
    CHECK(motion_groups(build(m.output("out5"))).empty());
    auto different = g;
    for (auto& n : different.nodes)
      if (n.op == OpKind::LeakyRelu) {
        std::get<LeakyReluAttrs>(n.attrs).alpha = 0.1F;
        break;
      }
    CHECK(motion_groups(different).empty());
  }
}

TEST_CASE("ZML : le manifeste sélectionne séparément les phases des deux RIFE reconnus en FP16",
          "[zml][rife]") {
  const Dims lite{96, 64, 32, 16};
  const Dims full{96, 64, 48, 32, 16};
  const RifeRecipes both{.lite422 = true, .full425 = true};
  const MotionRecipe up_and_transpose{.transposed = true, .resize = true};
  const MotionRecipe transpose_only{.transposed = true};
  phases_for(lite, 11, both, up_and_transpose);
  phases_for(full, 11, both, transpose_only);
  phases_for(lite, 11, {.lite422 = true}, up_and_transpose);
  phases_for(full, 11, {.lite422 = true}, {});
  phases_for(lite, 11, {.full425 = true}, {});
  phases_for(full, 11, {.full425 = true}, transpose_only);
  phases_for(lite, 11, {}, {});
  phases_for(full, 11, {}, {});
  phases_for(full, 11, both, {}, {}, DataType::F32);
  phases_for({96, 64, 32}, 11, both, {});
  phases_for(lite, 7, both, {});  // sans maillage compact reconnu
  phases_for(lite, 7, both, up_and_transpose, {6, 1, 2, 2});
  phases_for(lite, 7, both, {}, {6, 2, 2, 1});
  phases_for(full, 7, both, transpose_only, {6, 1, 2, 2});
  phases_for(full, 7, both, {}, {6, 2, 2, 1});
}

TEST_CASE("ZML : le centre D2 automatique cible le RIFE 4.22 lite FP16 à onze plans en 720p et 1080p",
          "[zml][rife][d2]") {
  const Dims lite{96, 64, 32, 16}, full{96, 64, 48, 32, 16};
  const RifeRecipes automatic{.d2 = RifeD2Centre::Auto422};
  const MotionRecipe d2{.d2_centre = true};
  for (const Pair measured : {Pair{768, 1280}, Pair{1088, 1920}}) {
    phases_for(lite, 11, automatic, d2, {}, DataType::F16, measured);
    phases_for(lite, 11, {}, {}, {}, DataType::F16, measured);
    phases_for(full, 11, automatic, {}, {}, DataType::F16, measured);
    phases_for({96, 64, 32}, 11, automatic, {}, {}, DataType::F16, measured);
    phases_for(lite, 11, automatic, {}, {}, DataType::F32, measured);
    phases_for(lite, 11, automatic, {}, {}, DataType::F16, {measured[0], measured[1] - 8});
    phases_for(lite, 11, automatic, {}, {}, DataType::F16, {measured[0] - 8, measured[1]});
    phases_for(lite, 7, automatic, {}, {6, 1, 2, 2}, DataType::F16, measured);
  }
  const RifeRecipes manual{.d2 = RifeD2Centre::Experimental};
  phases_for(lite, 11, manual, d2);
  phases_for(full, 11, manual, d2);
}
