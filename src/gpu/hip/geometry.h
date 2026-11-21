#pragma once

namespace kirari::gpu::hip_backend {
inline constexpr int kChannels = 64;
inline constexpr int kWave = 32;
inline constexpr int kRows = 4;
inline constexpr int kWorkers = kWave * kRows;
inline constexpr int kPixels = 16;
inline constexpr int kPointPixels = kRows * kPixels;
}  // namespace kirari::gpu::hip_backend
