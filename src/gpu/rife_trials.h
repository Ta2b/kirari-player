#pragma once

#include <cstdint>

namespace kirari::gpu {

// Range les sources du warp cubique 4K par pixel sur les RTX 40 compatibles.
inline constexpr std::uint32_t kTrialPackedWarp = 1U << 18U;

}  // namespace kirari::gpu
