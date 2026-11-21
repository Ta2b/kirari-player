#pragma once
#include "gpu/ir/ir.h"

namespace kirari::gpu::ir::detail {
float original_coordinate(CoordMode mode, std::int64_t x, float scale, std::int64_t in, std::int64_t out);
std::int64_t nearest_index(NearestMode mode, float x, std::int64_t in);
struct Linear {
  std::int64_t lo = 0;
  std::int64_t hi = 0;
  float t = 0.0F;
};
Linear linear_taps(float x, std::int64_t in);
}  // namespace kirari::gpu::ir::detail
