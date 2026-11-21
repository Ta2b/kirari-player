#include "gpu/zml/artcnn_body.h"

#include <array>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/ir/import.h"
#include "gpu/ir/passes.h"
#include "gpu/ir/shape.h"
#include "prepare/sha256.h"

using namespace kirari::gpu;
namespace {
ir::Graph loaded(const ModelSpec& model) {
  auto g = ir::load_onnx_file(model.onnx_path);
  REQUIRE(g.has_value());
  const std::array<ir::InputShape, 1> fixed{{{"input", {1, 3, 1080, 1920}}}};
  REQUIRE(ir::infer_shapes(*g, fixed).has_value());
  REQUIRE(ir::fold_constants(*g).has_value());
  ir::eliminate_dead_code(*g);
  ir::fuse_pixel_shuffles(*g);
  return std::move(*g);
}

ModelSpec official(std::string_view name) {
  ModelSpec m{.onnx_path = std::filesystem::path(KIRARI_ARTCNN_MODEL_DIR) / name,
              .precision = TensorType::F16};
  auto sha = kirari::prepare::sha256_file(m.onnx_path);
  REQUIRE(sha.has_value());
  m.sha256 = *sha;
  return m;
}
}  // namespace

TEST_CASE("ZML : ArtCNN conserve l’enveloppe couleur et les poids source", "[zml][artcnn]") {
  for (const auto* name : {"2x_ArtCNN_C4F16_rgb_fp16.onnx", "2x_ArtCNN_C4F32_rgb_fp16.onnx"}) {
    CAPTURE(name);
    auto model = official(name);
    const auto source = loaded(model);
    auto g = source;
    REQUIRE(zml::use_artcnn_half_body(g, model));
    REQUIRE(ir::validate(g).has_value());
    CHECK(g.inputs == source.inputs);
    CHECK(g.outputs == source.outputs);
    for (std::size_t i = 0; i < source.tensors.size(); ++i) {
      if (source.tensors[i].kind == ir::TensorKind::Constant) {
        CHECK(g.tensors[i].type == source.tensors[i].type);
        CHECK(g.tensors[i].value.f == source.tensors[i].value.f);
      }
    }
    std::size_t convolutions = 0;
    for (const auto& n : g.nodes) {
      if (n.op == ir::OpKind::Conv) {
        const auto dtype = convolutions++ == 0 ? ir::DataType::F32 : ir::DataType::F16;
        for (auto id : n.inputs) CHECK(g.tensors[id].type == dtype);
        CHECK(g.tensors[n.outputs[0]].type == dtype);
      }
      if (n.op == ir::OpKind::Resize || n.op == ir::OpKind::DepthToSpace || n.op == ir::OpKind::Clip ||
          n.op == ir::OpKind::Concat || n.op == ir::OpKind::Sub) {
        CHECK(g.tensors[n.inputs[0]].type == ir::DataType::F32);
        CHECK(g.tensors[n.outputs[0]].type == ir::DataType::F32);
      }
      if (n.op == ir::OpKind::Add) {
        const auto dtype = n.name.starts_with("luma/") ? ir::DataType::F16 : ir::DataType::F32;
        for (auto id : n.inputs) CHECK(g.tensors[id].type == dtype);
      }
    }
    CHECK(convolutions == 8);
    CHECK_FALSE(zml::use_artcnn_half_body(g, model));
  }
}

TEST_CASE("ZML : ArtCNN exige le fichier, le mode FP16 et la géométrie qualifiés", "[zml][artcnn]") {
  auto model = official("2x_ArtCNN_C4F16_rgb_fp16.onnx");
  auto g = loaded(model);
  SECTION("demande FP32") {
    model.precision = TensorType::F32;
  }
  SECTION("autre empreinte") {
    model.sha256[0] = 'a';
  }
  SECTION("empreinte déclarée valide avec un autre fichier") {
    model.onnx_path = official("2x_ArtCNN_C4F32_rgb_fp16.onnx").onnx_path;
  }
  SECTION("autre hauteur") {
    g.tensors[g.inputs[0]].dims[2] = 1088;
  }
  SECTION("branche couleur consommant la tête apprise") {
    g.nodes[3].inputs[1] = g.nodes[5].outputs[0];
  }
  const auto before = g.nodes.size();
  CHECK_FALSE(zml::use_artcnn_half_body(g, model));
  CHECK(g.nodes.size() == before);
}
