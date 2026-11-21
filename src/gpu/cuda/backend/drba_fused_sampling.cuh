// Arrondis et échantillonnage partagés par les deux voies exactes de DistilDRBA.
#pragma once

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace kirari::gpu::cuda::drba_sample {

__device__ __forceinline__ float rounded(float v) {
  return __half2float(__float2half_rn(v));
}

// Le côté choisi du graphe : vrai si tau < 1 (image 0), faux sinon (image 2). tau = 2 × plan 6 de pair01 ;
// le Clip du graphe, ((1 - tau) × 4096) borné à [0, 1], vaut 0 ou 1 pour un tau FP16 (jamais tau = 1 envoyé).
__device__ __forceinline__ bool before_side(float tau) {
  return fminf(fmaxf((1.0F - tau) * 4096.0F, 0.0F), 1.0F) >= 0.5F;
}

// Sortie 52 canaux d'un bloc ; Z = DepthToSpace(CRD, 2)(Y) de zh × zw.
struct SourceZ {
  const __half* y;
  int zh, zw;
  int pitch = 0;
};

// DepthToSpace lu à la volée : Z[c][zy][zx] = Y[4c + 2 (zy & 1) + (zx & 1)][zy >> 1][zx >> 1].
__device__ __forceinline__ float read_z(const SourceZ& z, int c, int zy, int zx) {
  const auto channel = static_cast<std::size_t>(4 * c + 2 * (zy & 1) + (zx & 1));
  if (z.pitch != 0) {
    const auto pixel = static_cast<std::size_t>(zy >> 1) * (z.zw / 2) + (zx >> 1);
    return __half2float(z.y[pixel * z.pitch + channel]);
  }
  return __half2float(z.y[(channel * static_cast<std::size_t>(z.zh / 2) + static_cast<std::size_t>(zy >> 1)) *
                              static_cast<std::size_t>(z.zw / 2) +
                          static_cast<std::size_t>(zx >> 1)]);
}

template <int Width, int Height>
struct TileZ {
  const float* values;
  int zh, zw, left, top;
};

template <int Width, int Height>
__device__ __forceinline__ float read_z(const TileZ<Width, Height>& z, int c, int zy, int zx) {
  return z.values[(c * Height + zy - z.top) * Width + zx - z.left];
}

// Resize bilinéaire demi-pixel de facteur s de Z (canaux c0 .. c0 + N - 1) au pixel (px, py) du plan agrandi,
// arrondi FP16 comme la sortie du Cast qui suit le Resize FP32 du graphe.
template <int N, class Z>
__device__ __forceinline__ void enlarge(const Z& z, int c0, int s, int px, int py, float (&r)[N]) {
  const float inv = 1.0F / static_cast<float>(s);
  float sy = (static_cast<float>(py) + 0.5F) * inv - 0.5F;
  float sx = (static_cast<float>(px) + 0.5F) * inv - 0.5F;
  sy = fminf(fmaxf(sy, 0.0F), static_cast<float>(z.zh - 1));
  sx = fminf(fmaxf(sx, 0.0F), static_cast<float>(z.zw - 1));
  const int y0 = static_cast<int>(sy);
  const int x0 = static_cast<int>(sx);
  const int y1 = min(y0 + 1, z.zh - 1);
  const int x1 = min(x0 + 1, z.zw - 1);
  const float wy = sy - static_cast<float>(y0);
  const float wx = sx - static_cast<float>(x0);
#pragma unroll
  for (int i = 0; i < N; ++i) {
    const float a = read_z(z, c0 + i, y0, x0);
    const float b = read_z(z, c0 + i, y0, x1);
    const float c = read_z(z, c0 + i, y1, x0);
    const float e = read_z(z, c0 + i, y1, x1);
    r[i] = rounded((a * (1.0F - wx) + b * wx) * (1.0F - wy) + (c * (1.0F - wx) + e * wx) * wy);
  }
}

// GridSample (bilinéaire, bords répétés, coins alignés) : indices et poids depuis la position en pixels. La
// grille normalisée du graphe, base + flot × pas ramenée par (g + 1) / 2 × (n - 1), vaut exactement x + flot.
// Le graphe FP16 livré la calcule en deux sommes FP16, d'un pas de 0,31 px à 1280 de large et de 0,47 px à
// 1920 (A03-18, A46-04) ; ici en FP32, comme le réseau d'origine. Le fusionné s'écarte donc exprès du graphe
// FP16 tant que celui-ci n'est pas réexporté avec sa grille en FP32 (tools/rife/convert_rife_fp16.py).
struct Tap {
  int i00, i01, i10, i11;  // indices dans un plan h × w
  float wx, wy;
};

__device__ __forceinline__ Tap tap_of(int x, int y, float flow_x, float flow_y, int w, int h) {
  float ix = __fadd_rn(static_cast<float>(x), flow_x);
  float iy = __fadd_rn(static_cast<float>(y), flow_y);
  ix = fminf(fmaxf(ix, 0.0F), static_cast<float>(w - 1));
  iy = fminf(fmaxf(iy, 0.0F), static_cast<float>(h - 1));
  const int x0 = static_cast<int>(ix);
  const int y0 = static_cast<int>(iy);
  const int x1 = min(x0 + 1, w - 1);
  const int y1 = min(y0 + 1, h - 1);
  Tap t;
  t.i00 = y0 * w + x0;
  t.i01 = y0 * w + x1;
  t.i10 = y1 * w + x0;
  t.i11 = y1 * w + x1;
  t.wx = ix - static_cast<float>(x0);
  t.wy = iy - static_cast<float>(y0);
  return t;
}

__device__ __forceinline__ float sample(const __half* plane, const Tap& t) {
  const float a = __half2float(plane[t.i00]);
  const float b = __half2float(plane[t.i01]);
  const float d = __half2float(plane[t.i10]);
  const float e = __half2float(plane[t.i11]);
  return (a * (1.0F - t.wx) + b * t.wx) * (1.0F - t.wy) + (d * (1.0F - t.wx) + e * t.wx) * t.wy;
}

}  // namespace kirari::gpu::cuda::drba_sample
