#include "gpu/zml/rife_conv.h"

#include "../gpu_ir/test_util.h"

using namespace kirari::gpu::ir;
using namespace kirari::gpu::ir::test;

namespace {
ModelBuilder motion_head(std::string_view mode) {
  ModelBuilder m;
  m.input("x", {1, 2, 3, 5})
      .weight("w", {2, 12, 4, 4}, sequence(2 * 12 * 4 * 4))
      .weight("b", {12}, sequence(12))
      .node("ConvTranspose", {"x", "w", "b"}, {"up"},
            {attr_ints("strides", {2, 2}), attr_ints("pads", {1, 1, 1, 1})})
      .node("DepthToSpace", {"up"}, {"channels"}, {attr_int("blocksize", 2), attr_string("mode", mode)});
  return m;
}

void check_pruning(const ModelBuilder& model, std::int64_t expected_filters) {
  const auto original = build(model);
  auto graph = original;
  kirari::gpu::zml::prune_motion_heads(graph);
  REQUIRE(validate(graph));
  REQUIRE(infer_shapes(graph));
  const auto& conv = graph.nodes.front();
  REQUIRE(conv.op == OpKind::ConvTranspose);
  CHECK(graph.tensors[conv.inputs[1]].dims == Dims{2, expected_filters, 4, 4});
  CHECK(graph.tensors[conv.inputs[2]].dims == Dims{expected_filters});
  const std::vector<NamedValue> input{{"x", random_floats({1, 2, 3, 5}, 11)}};
  const auto before = execute(original, input), after = execute(graph, input);
  REQUIRE(before.size() == after.size());
  for (std::size_t i = 0; i < before.size(); ++i) {
    CHECK(after[i].name == before[i].name);
    CHECK(after[i].value.dims == before[i].value.dims);
    CHECK(after[i].value.f == before[i].value.f);
  }
}
}  // namespace

TEST_CASE("ZML : les filtres inutilisés après le PixelShuffle de mouvement", "[zml][rife]") {
  SECTION("La sortie Split morte peut disparaître ; les poids partagés gardent leurs canaux") {
    auto m = motion_head("CRD");
    m.ints("sizes", {1, 1, 1})
        .node("Split", {"channels", "sizes"}, {"xflow", "yflow", "unused"}, {attr_int("axis", 1)})
        .node("ConvTranspose", {"x", "w", "b"}, {"shared"})
        .output("xflow")
        .output("yflow")
        .output("shared");
    check_pruning(m, 8);
  }
  SECTION("Une sortie intermédiaire publique conserve tous les filtres") {
    auto m = motion_head("CRD");
    m.ints("sizes", {1, 1, 1})
        .node("Split", {"channels", "sizes"}, {"xflow", "yflow", "tail"}, {attr_int("axis", 1)})
        .output("xflow")
        .output("up");
    check_pruning(m, 12);
  }
  SECTION("Deux tranches de canal à bornes positives se partagent le même préfixe") {
    auto m = motion_head("CRD");
    m.ints("axis", {-3})
        .ints("zero", {0})
        .ints("one", {1})
        .ints("two", {2})
        .node("Slice", {"channels", "zero", "one", "axis", "one"}, {"xflow"})
        .node("Slice", {"channels", "one", "two", "axis", "one"}, {"yflow"})
        .output("xflow")
        .output("yflow");
    check_pruning(m, 8);
  }
  SECTION("L'ordre DCR disperse les filtres d'un canal") {
    auto m = motion_head("DCR");
    m.ints("sizes", {1, 1, 1})
        .node("Split", {"channels", "sizes"}, {"xflow", "yflow", "unused"}, {attr_int("axis", 1)})
        .output("xflow");
    check_pruning(m, 12);
  }
  SECTION("Le dernier canal à index négatif dépend de la taille initiale") {
    auto m = motion_head("CRD");
    m.ints("axis", {1})
        .ints("start", {-2})
        .ints("end", {2})
        .node("Slice", {"channels", "start", "end", "axis"}, {"yflow"})
        .output("yflow");
    check_pruning(m, 12);
  }
}
