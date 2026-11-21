#include "gpu/zml/gmfss_mlp.h"

#include <cmath>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/zml/gmflow_global.h"
#include "gpu/zml/gmfss_graph.h"

using namespace kirari::gpu;
namespace {
ir::Graph load_pair(const char* name) {
  const auto path = std::filesystem::path(KIRARI_GMFSS_MODEL_DIR) / name;
  if (!std::filesystem::is_regular_file(path)) SKIP("Le modèle GMFSS du catalogue est absent");
  auto graph = ir::load_onnx_file(path);
  REQUIRE(graph.has_value());
  const std::array<ir::InputShape, 1> fixed{{{"frames", {1, 6, 1152, 1920}}}};
  REQUIRE(ir::infer_shapes(*graph, fixed).has_value());
  ir::promote_grid_chains(*graph);
  REQUIRE(ir::fold_constants(*graph).has_value());
  ir::eliminate_dead_code(*graph);
  ir::fuse_pixel_shuffles(*graph);
  zml::widen_gmflow_intermediates(*graph);
  return std::move(*graph);
}
}  // namespace

TEST_CASE("ZML : le MLP par bandes garde sa topologie et ses poids réels", "[.gmflow_mlp][zml][gmfss]") {
  auto graph = load_pair("gmfss_fortuna_union_s05_pair.onnx");
  const auto group = zml::gmflow_quarter_mlp(graph);
  REQUIRE(group.has_value());
  REQUIRE(group->inputs.size() == 9);
  REQUIRE(group->outputs.size() == 1);
  REQUIRE(group->members.size() == 10);
  SECTION("poids FP32 de même arrondi FP16 mais de valeur différente") {
    auto& w = graph.tensors[group->inputs[2]].value.f.front();
    w = std::nextafter(w, 1.0F);
    CHECK_FALSE(zml::gmflow_quarter_mlp(graph));
  }
  SECTION("poids FP16 modifié") {
    graph.tensors[group->inputs[6]].value.f.front() = 32.0F;
    CHECK_FALSE(zml::gmflow_quarter_mlp(graph));
  }
  SECTION("concaténation permutée") {
    auto& ins = graph.nodes[group->members.front()].inputs;
    std::swap(ins[0], ins[1]);
    CHECK_FALSE(zml::gmflow_quarter_mlp(graph));
  }
  SECTION("carte cachée lue par une autre branche") {
    graph.nodes.push_back(
        {.name = "branche", .op = ir::OpKind::Identity, .inputs = graph.nodes[group->members[5]].outputs});
    CHECK_FALSE(zml::gmflow_quarter_mlp(graph));
  }
  SECTION("autre taille de lecture") {
    graph.tensors[graph.inputs.front()].dims[2] = 1088;
    CHECK_FALSE(zml::gmflow_quarter_mlp(graph));
  }
}

TEST_CASE("ZML : le MLP de s05q conserve ses poids FP32", "[.gmflow_mlp][zml][gmfss]") {
  const auto graph = load_pair("gmfss_fortuna_union_s05q_pair.onnx");
  CHECK_FALSE(zml::gmflow_quarter_mlp(graph));
}

TEST_CASE("ZML : le matcher global en bandes exige les deux directions et les coordonnées réelles",
          "[.gmflow_global][zml][gmfss]") {
  auto graph = load_pair("gmfss_fortuna_union_s05_pair.onnx");
  const auto group = zml::gmflow_global_match(graph);
  REQUIRE(group.has_value());
  REQUIRE(group->inputs.size() == 4);
  REQUIRE(group->members.size() == 8);
  REQUIRE(group->outputs.size() == 1);
  SECTION("un autre consommateur a besoin des probabilités") {
    graph.outputs[0] = graph.nodes[group->members[6]].outputs.front();
    CHECK_FALSE(zml::gmflow_global_match(graph));
  }
  SECTION("les deux directions du score ont été échangées") {
    auto& cat = graph.nodes[group->members[5]];
    std::swap(cat.inputs[0], cat.inputs[1]);
    CHECK_FALSE(zml::gmflow_global_match(graph));
  }
  SECTION("la seconde grille n'a pas les mêmes coordonnées") {
    graph.tensors[group->inputs[3]].value.f.back() += 1.0F;
    CHECK_FALSE(zml::gmflow_global_match(graph));
  }
  SECTION("le softmax est appliqué sur un autre axe") {
    std::get<ir::NormAttrs>(graph.nodes[group->members[6]].attrs).axis = 1;
    CHECK_FALSE(zml::gmflow_global_match(graph));
  }
}
