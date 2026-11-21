#include "gpu/zml/gmfss_graph.h"

#include <array>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "gpu/ir/graph.h"
#include "gpu/ir/interpreter.h"

using namespace kirari::gpu;

TEST_CASE("ZML : GMFlow élargit ses intermédiaires et respecte les arrondis FP16 explicites",
          "[zml][gmfss]") {
  using namespace ir;
  Graph graph;
  auto scalar = [&](std::string name, TensorKind kind, DataType type) {
    return add_tensor(graph,
                      {.name = std::move(name), .kind = kind, .type = type, .dims = {1}, .known = true});
  };
  const auto x = scalar("frames", TensorKind::Input, DataType::F16);
  const auto w = scalar("w", TensorKind::Constant, DataType::F16);
  // Une constante calculée par le pliage, à arrondir comme le littéral FP16 du graphe source.
  graph.tensors[w].value = {.type = DataType::F16, .dims = {1}, .f = {1.0001F}};
  const auto sum = scalar("sum", TensorKind::Activation, DataType::F16);
  const auto half = scalar("rounded", TensorKind::Activation, DataType::F16);
  graph.inputs = {x};
  graph.outputs = {sum, half};
  graph.nodes = {{.name = "add", .op = OpKind::Add, .inputs = {x, w}, .outputs = {sum}},
                 {.name = "half",
                  .op = OpKind::Cast,
                  .attrs = CastAttrs{.to = DataType::F16},
                  .inputs = {sum},
                  .outputs = {half}}};
  zml::widen_gmflow_intermediates(graph);
  REQUIRE(validate(graph).has_value());
  CHECK(graph.tensors[x].type == DataType::F16);
  CHECK(graph.tensors[w].value.f.front() == 1.0F);
  CHECK(graph.tensors[sum].type == DataType::F32);
  CHECK(graph.tensors[half].type == DataType::F32);
  const std::array<NamedValue, 1> inputs{
      NamedValue{"frames", {.type = DataType::F16, .dims = {1}, .f = {0.00048828125F}}}};
  auto result = run(graph, inputs);
  REQUIRE(result.has_value());
  REQUIRE(result->size() == 2);
  CHECK((*result)[0].value.f.front() == 1.00048828125F);
  CHECK((*result)[1].value.f.front() == 1.0F);
}
