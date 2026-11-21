#include "gpu/zml/drba_graph.h"

#include "../gpu_ir/test_util.h"
#include "gpu/zml/drba_warps.h"

using namespace kirari::gpu::ir;
using namespace kirari::gpu::ir::test;
using namespace kirari::gpu::zml;
namespace {
ModelBuilder encoders() {
  ModelBuilder m;
  m.input("pair01", {1, 3, 4, 6}).input("pair12", {1, 3, 4, 6}).node("Add", {"pair01", "pair12"}, {"sum"});
  const std::string first = "net.encode.cnn0.weight";
  m.weight(first, {3, 3, 1, 1}, sequence(9))
      .weight("middle", {3, 3, 1, 1}, sequence(9, -1.0F, 1.0F, 7))
      .weight("last", {3, 3, 1, 1}, sequence(9, -1.0F, 1.0F, 16));
  for (const std::string image : {"pair01", "pair12", "sum"}) {
    std::string x = image;
    for (const std::string& w : {first, std::string("middle"), std::string("last")}) {
      m.node("Conv", {x, w}, {image + w})
          .node("LeakyRelu", {image + w}, {image + w + "act"}, {attr_float("alpha", 0.1F)});
      x = image + w + "act";
    }
    m.node("ConvTranspose", {x, "middle"}, {image + "_out"}).output(image + "_out");
  }
  return m;
}

ModelBuilder warps(float second_step, bool second_aligned) {
  ModelBuilder m;
  m.input("rgb", {1, 3, 4, 6})
      .input("features", {1, 16, 4, 6})
      .input("flow", {1, 3, 5, 2})
      .weight("step0", {}, {0.5F})
      .weight("step1", {}, {second_step})
      .node("Mul", {"flow", "step0"}, {"grid0"})
      .node("Mul", {"flow", "step1"}, {"grid1"})
      .node("GridSample", {"rgb", "grid0"}, {"rgb_warp"}, {attr_int("align_corners", 1)})
      .node("GridSample", {"features", "grid1"}, {"features_warp"},
            {attr_int("align_corners", second_aligned)})
      .output("rgb_warp")
      .output("features_warp");
  return m;
}
void check_outputs(const Graph& before, const Graph& after, const std::vector<NamedValue>& inputs) {
  REQUIRE(validate(after));
  const auto baseline = execute(before, inputs);
  const auto results = execute(after, inputs);
  REQUIRE(results.size() == baseline.size());
  for (std::size_t i = 0; i < baseline.size(); ++i) {
    CHECK(results[i].name == baseline[i].name);
    CHECK(results[i].value.dims == baseline[i].value.dims);
    CHECK(results[i].value.f == baseline[i].value.f);
  }
}
}  // namespace

TEST_CASE("ZML : les trois encodages DistilDRBA gardent leurs sorties individuelles", "[zml][drba]") {
  const auto original = build(encoders());
  auto graph = original;
  REQUIRE(optimize_drba_graph(graph));
  REQUIRE(infer_shapes(graph));
  CHECK(count_ops(graph, OpKind::Conv) == 3);
  CHECK(count_ops(graph, OpKind::ConvTranspose) == 1);
  CHECK_FALSE(optimize_drba_graph(graph));
  check_outputs(original, graph,
                {{"pair01", random_floats({1, 3, 4, 6}, 4)}, {"pair12", random_floats({1, 3, 4, 6}, 9)}});
  auto distinct = original;
  for (auto& n : distinct.nodes)
    if (n.op == OpKind::LeakyRelu) {
      std::get<LeakyReluAttrs>(n.attrs).alpha = 0.125F;
      break;
    }
  CHECK_FALSE(optimize_drba_graph(distinct));
}

TEST_CASE("ZML : seuls les warps DistilDRBA de coordonnées identiques se regroupent", "[zml][drba]") {
  const auto original = build(warps(0.5F, true));
  auto graph = original;
  REQUIRE(combine_drba_warps(graph) == 1);
  REQUIRE(infer_shapes(graph));
  CHECK(count_ops(graph, OpKind::GridSample) == 1);
  CHECK(combine_drba_warps(graph) == 0);
  check_outputs(original, graph,
                {{"rgb", random_floats({1, 3, 4, 6}, 7)},
                 {"features", random_floats({1, 16, 4, 6}, 13)},
                 {"flow", random_floats({1, 3, 5, 2}, 3, -4.0F, 4.0F)}});
  auto other_grid = build(warps(0.25F, true));
  CHECK(combine_drba_warps(other_grid) == 0);
  auto other_alignment = build(warps(0.5F, false));
  CHECK(combine_drba_warps(other_alignment) == 0);
}
