#include <iterator>
#include <optional>

#include "../gpu_ir/test_util.h"
#include "gpu/zml/hlo.h"

using namespace kirari::gpu;
using namespace kirari::gpu::ir::test;

namespace {
// Isole le nœud demandé, ou le dernier. Producteurs et lecteurs gardent leurs formes et types inférés.
std::string sampling_module(const ModelBuilder& model, std::optional<ir::OpKind> target = {}) {
  const auto graph = build(model);
  zml::Hlo hlo;
  std::vector<zml::Value> tensors;
  for (const auto& t : graph.tensors)
    tensors.push_back(
        {"%input" + std::to_string(tensors.size()), t.dims, std::string(zml::element_type(t.type))});
  const auto iter =
      target ? std::ranges::find(graph.nodes, *target, &ir::Node::op) : std::prev(graph.nodes.end());
  REQUIRE(iter != graph.nodes.end());
  const auto y = zml::sample(hlo, graph, *iter, tensors);
  REQUIRE(y);
  REQUIRE(y->dims == graph.tensors[iter->outputs.front()].dims);
  return hlo.module(tensors, {*y});
}

ModelBuilder resized(float scale) {
  ModelBuilder m;
  m.input("x", {2, 7, 32, 48}).weight("scale", {4}, {1, 1, scale, scale});
  m.node("Resize", {"x", "", "scale"}, {"output"}, {attr_string("mode", "linear")}).output("output");
  return m;
}
}  // namespace

TEST_CASE("ZML : les warps des deux images utilisent une fenêtre bordée par coordonnée", "[zml][rife]") {
  for (auto height : {1, 6}) {
    ModelBuilder m;
    m.input("rgb", {2, 7, height, 9})
        .input("grid", {2, 3, 5, 2})
        .node("GridSample", {"rgb", "grid"}, {"output"}, {attr_string("padding_mode", "border")})
        .output("output");
    const auto text = sampling_module(m);
    CHECK(text.contains("operand_batching_dims = [0]") == (height > 1));
    CHECK(text.contains("2x7x3x5x2x2xf32") == (height > 1));
  }
}

TEST_CASE("ZML : le warp RIFE final lit des pixels FP16 linéaires sur un rectangle impair", "[zml][rife]") {
  for (const int height : {1, 3}) {
    constexpr int width = 5;
    constexpr auto fp16 = ir::onnx::kFloat16;
    ModelBuilder m;
    m.input("pair", {2, 7, height, width}, fp16)
        .input("grid", {2, height, width, 2})
        .input("state", {1, 14, height, width}, fp16)
        .input("filter", {1, 28, 3, 3}, fp16)
        .ints("channels", {3, 4})
        .ints("rgb_shape", {1, 6, height, width})
        .ints("feature_shape", {1, 8, height, width})
        .node("Cast", {"pair"}, {"float_pair"}, {attr_int("to", ir::onnx::kFloat)})
        .node("GridSample", {"float_pair", "grid"}, {"warp"},
              {attr_string("padding_mode", "border"), attr_int("align_corners", 1)})
        .node("Cast", {"warp"}, {"rounded"}, {attr_int("to", fp16)})
        .node("Split", {"rounded", "channels"}, {"colors", "features"}, {attr_int("axis", 1)})
        .node("Reshape", {"colors", "rgb_shape"}, {"rgb"})
        .node("Reshape", {"features", "feature_shape"}, {"feats"})
        .node("Concat", {"rgb", "feats", "state"}, {"all"}, {attr_int("axis", 1)})
        .node("Conv", {"all", "filter"}, {"output"},
              {attr_ints("strides", {2, 2}), attr_ints("pads", {1, 1, 1, 1})})
        .output("output", fp16);
    const auto text = sampling_module(m, ir::OpKind::GridSample);
#ifdef KIRARI_ZML_LINEAR_WARP
    CHECK(text.contains("tensor<30x7xf16>") == (height == 3));
    CHECK(text.contains("index_vector_dim = 4") == (height == 3));
#else
    CHECK_FALSE(text.contains("index_vector_dim = 4"));
#endif
  }
}

TEST_CASE("ZML : les réductions à pas constant se lisent par tranches", "[zml][rife]") {
  CHECK_FALSE(sampling_module(resized(0.125F)).contains("\"stablehlo.gather\""));
  CHECK(sampling_module(resized(0.75F)).contains("\"stablehlo.gather\""));
}

TEST_CASE("ZML : quatre gathers de Resize remplacés par des phases sur la recette candidate", "[zml][rife]") {
  for (const float scale : {2.0F, 4.0F, 8.0F, 16.0F}) {
    const auto text = sampling_module(resized(scale));
    constexpr std::string_view gather = "\"stablehlo.gather\"";
    std::size_t count = 0;
    for (auto pos = text.find(gather); pos != std::string::npos; pos = text.find(gather, pos + gather.size()))
      ++count;
#ifdef KIRARI_ZML_RESIZE_PHASES
    CHECK(count == 0);
#else
    CHECK(count == 4);
#endif
  }
}

TEST_CASE("ZML : une réduction RIFE ne déforme que les taps de ses cellules", "[zml][rife]") {
  ModelBuilder m;
  m.input("pair", {2, 7, 32, 48})
      .input("grid", {2, 32, 48, 2})
      .input("timestep", {1, 1, 32, 48})
      .ints("channels", {3, 4})
      .ints("both_rgb", {1, 6, 32, 48})
      .ints("both_features", {1, 8, 32, 48})
      .weight("scale", {4}, {1, 1, 0.125F, 0.125F})
      .node("GridSample", {"pair", "grid"}, {"warp"}, {attr_string("padding_mode", "border")})
      .node("Split", {"warp", "channels"}, {"colors", "features"}, {attr_int("axis", 1)})
      .node("Reshape", {"colors", "both_rgb"}, {"rgb"})
      .node("Reshape", {"features", "both_features"}, {"encoded"})
      .node("Concat", {"rgb", "encoded", "timestep"}, {"planes"}, {attr_int("axis", 1)})
      .node("Resize", {"planes", "", "scale"}, {"output"}, {attr_string("mode", "linear")})
      .output("output");
  const auto text = sampling_module(m);
  CHECK(text.contains("2x7x8x12x2x2xf32"));
  CHECK_FALSE(text.contains("2x7x32x48x2x2xf32"));
}
