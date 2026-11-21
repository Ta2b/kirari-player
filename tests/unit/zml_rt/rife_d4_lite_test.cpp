#include <array>
#include <filesystem>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "gpu/zml/drba_graph.h"
#include "gpu/zml/motion_groups.h"
#include "gpu/zml/rife_conv.h"

using namespace kirari::gpu;
namespace {
ir::Graph prepared(const char* name, ir::Dims input = {1, 11, 768, 1280}) {
  const auto file = std::filesystem::path(KIRARI_RIFE_MODEL_DIR) / name;
  if (!std::filesystem::is_regular_file(file)) SKIP("Le modèle RIFE du catalogue est absent");
  auto graph = ir::load_onnx_file(file);
  REQUIRE(graph.has_value());
  // Le lecteur ouvre la dérivation mouvement de ces exports du catalogue.
  const std::string_view flow =
      std::string_view(name) == "rife_v4.22_lite.onnx" ? "/Add_5_output_0" : "/Add_7_output_0";
  const auto flow_id = ir::find_tensor(*graph, flow), mask_id = ir::find_tensor(*graph, "/Sigmoid_output_0");
  REQUIRE(flow_id.has_value());
  REQUIRE(mask_id.has_value());
  graph->outputs = {*flow_id, *mask_id};
  REQUIRE(graph->inputs.size() == 1);
  const std::array<ir::InputShape, 1> fixed{{{graph->tensors[graph->inputs[0]].name, std::move(input)}}};
  REQUIRE(ir::infer_shapes(*graph, fixed).has_value());
  ir::promote_grid_chains(*graph);
  REQUIRE(ir::fold_constants(*graph).has_value());
  ir::eliminate_dead_code(*graph);
  ir::fuse_pixel_shuffles(*graph);
  zml::optimize_drba_graph(*graph);
  zml::prune_motion_heads(*graph);
  return std::move(*graph);
}
}  // namespace

TEST_CASE("ZML : le D4 du RIFE 4.25 lite garde les arrondis et les usages du graphe",
          "[zml][rife][d4_lite]") {
  auto g = prepared("rife_v4.25_lite.onnx");
  const auto groups = zml::motion_groups(g);
  const auto d4 = zml::rife425_lite_d4(g, groups);
  REQUIRE(d4.has_value());
  REQUIRE(d4->inputs.size() == 5);
  REQUIRE(d4->outputs == std::vector<ir::TensorId>{groups[4].inputs.front()});
  REQUIRE(d4->members.size() == 7);
  SECTION("une autre branche demande le warp complet") {
    g.nodes.push_back(
        {.name = "branche", .op = ir::OpKind::Identity, .inputs = g.nodes[d4->members[0]].outputs});
    CHECK_FALSE(zml::rife425_lite_d4(g, groups));
  }
  SECTION("l’échantillonnage emploie des zéros hors image") {
    auto& a = std::get<ir::GridSampleAttrs>(g.nodes[d4->members[0]].attrs);
    a.padding = ir::GridPadding::Zeros;
    CHECK_FALSE(zml::rife425_lite_d4(g, groups));
  }
  SECTION("le masque et les neuf attributs changent de place") {
    auto& join = g.nodes[d4->members[3]];
    REQUIRE(join.op == ir::OpKind::Concat);
    std::swap(join.inputs[1], join.inputs[2]);
    CHECK_FALSE(zml::rife425_lite_d4(g, groups));
  }
  SECTION("le premier arrondi FP16 du warp a disparu") {
    g.tensors[g.nodes[d4->members[1]].outputs[0]].type = ir::DataType::F32;
    CHECK_FALSE(zml::rife425_lite_d4(g, groups));
  }
  SECTION("le contrat d’entrée du modèle est en FP32") {
    g.tensors[g.inputs[0]].type = ir::DataType::F32;
    CHECK_FALSE(zml::rife425_lite_d4(g, groups));
  }
}

TEST_CASE("ZML : la réduction D4 garde le graphe des RIFE voisins et des autres formats",
          "[zml][rife][d4_lite]") {
  for (const char* model : {"rife_v4.22_lite.onnx", "rife_v4.25.onnx", "rife_v4.26.onnx"}) {
    CAPTURE(model);
    auto graph = prepared(model);
    CHECK_FALSE(zml::rife425_lite_d4(graph, zml::motion_groups(graph)));
  }
  auto graph = prepared("rife_v4.25_lite.onnx", {1, 11, 1152, 1920});
  CHECK_FALSE(zml::rife425_lite_d4(graph, zml::motion_groups(graph)));
}
