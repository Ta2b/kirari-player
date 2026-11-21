#include "gpu/zml/span_strips.h"

#include <catch2/catch_test_macros.hpp>

using kirari::gpu::zml::span_strips_plan;

TEST_CASE("SPAN specialise les quatre bandes sans decaler le degroupage", "[span_strips]") {
  const auto balanced = span_strips_plan({1, 3, 1080, 1920}, 8);
  REQUIRE(balanced);
  CHECK(balanced->core == 270);
  CHECK(balanced->halo == 52);
  CHECK(balanced->window == 374);
  const auto performance = span_strips_plan({1, 3, 1080, 1920}, 5);
  REQUIRE(performance);
  CHECK(performance->halo == 34);
  CHECK(performance->window == 338);
  CHECK_FALSE(span_strips_plan({1, 3, 1078, 1920}, 8));
  CHECK_FALSE(span_strips_plan({1, 3, 1080, 1919}, 8));
  CHECK_FALSE(span_strips_plan({1, 3, 128, 128}, 8));
  CHECK_FALSE(span_strips_plan({1, 3, 1080, 1920}, 6));
}
