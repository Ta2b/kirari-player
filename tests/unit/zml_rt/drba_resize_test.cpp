#include "gpu/zml/drba_resize.h"

#include <array>
#include <format>

#include <catch2/catch_test_macros.hpp>

#include "gpu/zml/hlo.h"

using namespace kirari::gpu;
namespace {
ir::Graph geometry(int height, int width, int factor) {
  ir::Graph g;
  for (const std::string name : {"pair01", "pair12", "out", "net.encode.cnn0.weight", "rgb", "small"}) {
    ir::Tensor t;
    t.name = name;
    t.dims = {1, name.starts_with("pair") ? 7 : 3, height, width};
    t.type = ir::DataType::F16;
    g.tensors.push_back(std::move(t));
  }
  g.tensors.back().dims = {1, 3, height / factor, width / factor};
  g.inputs = {0, 1};
  g.outputs = {2};
  g.nodes.push_back({.name = "Resize",
                     .op = ir::OpKind::Resize,
                     .attrs = ir::ResizeAttrs{.mode = ir::ResizeMode::Linear},
                     .inputs = {4},
                     .outputs = {5}});
  return g;
}
}  // namespace

TEST_CASE("ZML : les quatre pieds FP32 des réductions /4 et /8 de DistilDRBA", "[zml][drba]") {
  for (int r : {4, 8}) {
    auto g = geometry(3 * r, 5 * r, r);
    auto& n = g.nodes.front();
    const auto scale = 1.0F / static_cast<float>(r);
    const std::array<float, 4> scales{1, 1, scale, scale};
    CHECK(zml::drba_resize_factor(g, n, scales) == r);
    zml::Hlo hlo;
    std::vector<zml::Value> tensors(g.tensors.size());
    tensors[4] = {"%rgb", g.tensors[4].dims, "f16"};
    auto sampled = zml::sample(hlo, g, n, tensors);
    REQUIRE(sampled.has_value());
    const auto text = hlo.module({tensors[4]}, {*sampled});
    const auto first = r / 2 - 1;
    for (auto y : {first, first + 1})
      for (auto x : {first, first + 1})
        CHECK(text.contains(std::format("start_indices = array<i64: 0, 0, {}, {}>", y, x)));
    CHECK(text.contains(std::format("strides = array<i64: 1, 1, {}, {}>", r, r)));
    CHECK(text.contains("tensor<1x3x3x5xf32>"));
    CHECK(sampled->element == "f16");
    g.tensors[4].type = g.tensors[5].type = ir::DataType::F32;
    CHECK(zml::drba_resize_factor(g, n, scales) == r);
  }
}

TEST_CASE("ZML : géométrie et signature Distil requises pour le Resize direct", "[zml][drba]") {
  auto g = geometry(12, 20, 4);
  auto& n = g.nodes.front();
  std::array<float, 4> scales{1, 1, 0.25F, 0.25F};
  auto& attrs = std::get<ir::ResizeAttrs>(n.attrs);
  SECTION("dimensions arrondies") {
    g.tensors[4].dims[3] = 21;
  }
  SECTION("axe non divisible") {
    g.tensors[4].dims[2] = 15;
  }
  SECTION("taille de sortie erronée") {
    g.tensors[5].dims[2] = 2;
  }
  SECTION("échelle donnée proche") {
    scales[2] = 0.2499F;
  }
  SECTION("échelle d'un autre axe") {
    scales[1] = 1.01F;
  }
  SECTION("rapport anisotrope") {
    scales[3] = 0.125F;
  }
  SECTION("align_corners") {
    attrs.coord = ir::CoordMode::AlignCorners;
  }
  SECTION("pytorch_half_pixel avec sortie unitaire") {
    attrs.coord = ir::CoordMode::PytorchHalfPixel;
    g.tensors[4].dims[2] = 4;
    g.tensors[5].dims[2] = 1;
  }
  SECTION("voisin le plus proche") {
    attrs.mode = ir::ResizeMode::Nearest;
  }
  SECTION("type entier") {
    g.tensors[4].type = g.tensors[5].type = ir::DataType::I64;
  }
  SECTION("autre modèle") {
    g.tensors[1].name = "pair02";
  }
  SECTION("autre encodeur") {
    g.tensors[3].name = "weights";
  }
  CHECK(zml::drba_resize_factor(g, n, scales) == 0);
}
