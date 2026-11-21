#include <hip/hip_runtime.h>

#include "geometry.h"
#include "matrix.h"

namespace {
using namespace kirari::gpu::hip_backend;
using matrix::Half;
using Half8 = uint4;

__device__ float bounded(float value) {
  return isnan(value) ? 0.F : fmaxf(-65504.F, fminf(value, 65504.F));
}

template <int Inputs, int Outputs, bool Relu, bool Pixel>
__device__ void stencil(const Half* input, const Half* weights, const float* bias, Half* output, int width,
                        int height, Half* halo) {
  constexpr int pitch = Inputs + 8;
  constexpr int halo_width = kPixels + 2;
  constexpr int chunks = Inputs / 8;
  const int worker = threadIdx.x, wave = worker / kWave, lane = worker % kWave;
  const int origin_x = blockIdx.x * kPixels, origin_y = blockIdx.y * kRows;
  for (int slot = worker; slot < (kRows + 2) * halo_width * chunks; slot += kWorkers) {
    const int pixel = slot / chunks, c = slot % chunks * 8;
    const int x = origin_x + pixel % halo_width - 1, y = origin_y + pixel / halo_width - 1;
    Half8 value{};
    if (x >= 0 && x < width && y >= 0 && y < height)
      value = *reinterpret_cast<const Half8*>(input + (y * width + x) * Inputs + c);
    *reinterpret_cast<Half8*>(halo + pixel * pitch + c) = value;
  }
  __syncthreads();

  matrix::Accumulator acc[Outputs / 16];
#pragma unroll
  for (int n = 0; n < Outputs / 16; ++n) acc[n] = matrix::zero();
  const int local = lane % 16;
#pragma unroll 1
  for (int tap = 0; tap < 9; ++tap) {
    const int at = (wave + tap / 3) * halo_width + tap % 3;
#pragma unroll 2
    for (int k = 0; k < Inputs; k += 16) {
      const auto a = matrix::left(halo + at * pitch + k, pitch, local);
#pragma unroll
      for (int n = 0; n < Outputs / 16; ++n) {
        const auto b = matrix::right(weights + (n * 16 * 9 + tap) * Inputs + k, 9 * Inputs, local);
        acc[n] = matrix::add(a, b, acc[n]);
      }
    }
  }

  const int y = origin_y + wave;
  if (y >= height) return;
#pragma unroll
  for (int n = 0; n < Outputs / 16; ++n) {
    const int c = n * 16 + local;
    const float offset = bias[c];
    const auto output_values = matrix::elements(acc[n], worker / kWave, lane);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      // Les deux moitiés du wave écrivent les pixels pairs et impairs.
      const int x = origin_x + 2 * i + lane / 16;
      if (x >= width) continue;
      const float v = bounded(output_values.values[i]) + offset;
      const Half code = static_cast<Half>(bounded(Relu ? fmaxf(0.F, v) : v));
      if constexpr (Pixel) {
        const int dest_y = y * 4 + c / 4 % 4, dest_x = x * 4 + c % 4;
        output[((c / 16) * (height * 4) + dest_y) * (width * 4) + dest_x] = code;
      } else {
        output[(y * width + x) * Outputs + c] = code;
      }
    }
  }
}

template <int Sources>
__device__ void point(const Half* const (&inputs)[Sources], const Half* skip, const Half* weights,
                      const float* bias, Half* output, int pixels) {
  const int worker = threadIdx.x, lane = worker % kWave, local = lane % 16;
  const int base = blockIdx.x * kPointPixels + (worker / kWave) * 16;
  matrix::Accumulator acc[kChannels / 16];
#pragma unroll
  for (int n = 0; n < kChannels / 16; ++n) acc[n] = matrix::zero();
#pragma unroll 1
  for (int source = 0; source < Sources; ++source) {
#pragma unroll 2
    for (int k = 0; k < kChannels; k += 16) {
      const auto a = matrix::point_left(inputs[source], base, pixels, kChannels, k, local);
#pragma unroll
      for (int n = 0; n < kChannels / 16; ++n) {
        const auto b =
            matrix::right(weights + (n * 16 * Sources + source) * kChannels + k, Sources * kChannels, local);
        acc[n] = matrix::add(a, b, acc[n]);
      }
    }
  }
#pragma unroll
  for (int n = 0; n < kChannels / 16; ++n) {
    const int c = n * 16 + local;
    const float offset = bias[c];
    const auto output_values = matrix::elements(acc[n], worker / kWave, lane);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int p = base + 2 * i + lane / 16;
      if (p >= pixels) continue;
      const int pos = p * kChannels + c;
      float value = output_values.values[i] + offset;
      if constexpr (Sources == 1)
        value = (static_cast<float>(skip[pos]) + static_cast<float>(inputs[0][pos])) * tanhf(value);
      output[pos] = static_cast<Half>(value);
    }
  }
}
}  // namespace

extern "C" __global__ __launch_bounds__(kWorkers) void unshuffle(const Half* input, Half* output, int width,
                                                                 int height) {
  const int i = blockIdx.x * kWorkers + threadIdx.x, p = i / 16, c = i % 16;
  if (p >= width * height) return;
  const int y = 2 * (p / width) + c / 2 % 2, x = 2 * (p % width) + c % 2;
  output[i] = c < 12 ? input[(c / 4 * (2 * height) + y) * (2 * width) + x] : Half(0);
}

// Ces points d'entrée partagent le même emplacement de halo et la même séquence tap, K16.
#define CONVOLUTION(name, ci, co, relu, pixel)                                                          \
  extern "C" __global__ __launch_bounds__(kWorkers) void name(                                          \
      const Half* input, const Half* weights, const float* bias, Half* output, int width, int height) { \
    __shared__ __align__(16) Half halo[(kRows + 2) * (kPixels + 2) * (ci + 8)];                         \
    stencil<ci, co, relu, pixel>(input, weights, bias, output, width, height, halo);                    \
  }
CONVOLUTION(stem, 16, 64, false, false)
CONVOLUTION(relu, 64, 64, true, false)
CONVOLUTION(plain, 64, 64, false, false)
CONVOLUTION(pixel_shuffle, 64, 48, false, true)
#undef CONVOLUTION

extern "C" __global__ __launch_bounds__(kWorkers) void gate(const Half* skip, const Half* residual,
                                                            const Half* weights, const float* bias,
                                                            Half* output, int pixels) {
  const Half* const inputs[]{residual};
  point(inputs, skip, weights, bias, output, pixels);
}

extern "C" __global__ __launch_bounds__(kWorkers) void project(const Half* stem, const Half* last,
                                                               const Half* first, const Half* relu,
                                                               const Half* weights, const float* bias,
                                                               Half* output, int pixels) {
  const Half* const inputs[]{stem, last, first, relu};
  point(inputs, nullptr, weights, bias, output, pixels);
}
