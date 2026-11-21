#include "gpu/zml/span_frontend.h"

#include <catch2/catch_test_macros.hpp>

namespace kirari::gpu::zml {
TEST_CASE("SPAN sélectionne les bandes de projection par famille et géométrie", "[zml][span]") {
  const SpanSdk portable{.available = true};
  const SpanSdk balanced{.available = true, .balanced_strips = true};
  const SpanSdk performance{.available = true, .performance_strips = true};
  const SpanSdk both{.available = true, .balanced_strips = true, .performance_strips = true};

  CHECK(portable.projection_option(48, 5, 1080, 1920).empty());
  CHECK(portable.projection_option(64, 8, 1080, 1920).empty());
  CHECK(balanced.projection_option(64, 8, 1080, 1920) == "--projection-strips");
  CHECK(balanced.projection_option(48, 5, 1080, 1920).empty());
  CHECK(performance.projection_option(48, 5, 1080, 1920) == "--projection-strips");
  CHECK(performance.projection_option(64, 8, 1080, 1920).empty());
  CHECK(both.projection_option(64, 8, 1080, 1920) == "--projection-strips");
  CHECK(both.projection_option(48, 5, 1080, 1920) == "--projection-strips");
  CHECK(both.projection_option(48, 8, 1080, 1920).empty());
  CHECK(both.projection_option(64, 5, 1080, 1920).empty());
  CHECK(both.projection_option(48, 5, 720, 1280).empty());
  CHECK(both.projection_option(48, 5, 1920, 1080).empty());
  CHECK(both.projection_option(48, 5, 1080, 1280).empty());
  const SpanSdk tiled{
      .available = true, .balanced_strips = true, .performance_strips = true, .pixel_strips = true};
  CHECK(tiled.projection_option(48, 5, 1080, 1920) == "--projection-pixel-strips");
  CHECK(tiled.projection_option(64, 8, 1080, 1920) == "--projection-strips");
  CHECK(tiled.projection_option(48, 5, 720, 1280).empty());
  const SpanSdk pixel_only{.available = true, .pixel_strips = true};
  CHECK(pixel_only.projection_option(48, 5, 1080, 1920).empty());
  const SpanSdk unavailable{.performance_strips = true};
  CHECK(unavailable.projection_option(48, 5, 1080, 1920).empty());
}
}  // namespace kirari::gpu::zml
