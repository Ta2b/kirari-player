#include "gpu/zml/span_strips.h"

#include <format>
#include <string_view>

#include "gpu/span_fp8/package.h"

namespace kirari::gpu::zml {
std::optional<SpanStrips> span_strips_plan(const ir::Dims& dims, std::size_t blocks) {
  if ((blocks != 5 && blocks != 8) || dims.size() != 4 || dims[0] != 1 || dims[1] != 3) return {};
  const auto height = dims[2], width = dims[3];
  if (height <= 0 || width <= 0 || height > span_fp8::kMaxSide || width > span_fp8::kMaxSide ||
      height % 8 != 0 || width % 2 != 0)
    return {};
  // Le dégroupage 2×2 double le rayon des 3×3 : entrée, trois par bloc, puis sortie du corps.
  const auto core = height / 4;
  const auto halo = 2 * (2 + 3 * static_cast<std::int64_t>(blocks));
  const auto window = core + 2 * halo;
  if (window >= height) return {};
  return SpanStrips{height, width, core, halo, window};
}

std::optional<std::string> wrap_span_strips(std::string tile_module, const SpanStrips& p) {
  using namespace std::string_view_literals;
  auto decl = "func.func @main("sv;
  auto pos = tile_module.find(decl);
  if (pos == std::string::npos) {
    decl = "func.func public @main("sv;
    pos = tile_module.find(decl);
  }
  if (pos == std::string::npos || tile_module.contains("@kirari_span_bande")) return {};
  const auto f16 = [](std::int64_t h, std::int64_t w) { return std::format("tensor<1x3x{}x{}xf16>", h, w); };
  const auto input = f16(p.height, p.width), output = f16(2 * p.height, 2 * p.width);
  const auto tile = f16(p.window, p.width), pixels = f16(2 * p.window, 2 * p.width);
  const auto piece = f16(2 * p.core, 2 * p.width);
  // Certains frontaux ajoutent des attributs aux arguments/résultats. Ne modifier que le symbole,
  // et vérifier les types réellement émis avant de raccorder cet appel.
  const auto body = tile_module.find('\n', pos);
  const std::string_view signature = std::string_view(tile_module).substr(pos, body - pos);
  if (!signature.contains(tile) || !signature.contains(pixels)) return {};
  tile_module.replace(pos, decl.size(), "func.func private @kirari_span_bande(");
  // Les fenêtres extrêmes sont déplacées vers l'intérieur. Ainsi le bourrage des convolutions
  // est appliqué au vrai bord de l'image, et le dégroupage 2×2 reste aligné pour les quatre tours.
  tile_module.insert(pos, std::format(R"(func.func @main(%image: {0}) -> {1} {{
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %empty = stablehlo.constant dense<0.000000e+00> : {1}
    %index, %output, %source_out = stablehlo.while(%i = %zero, %old = %empty, %source = %image) : tensor<i32>, {1}, {0}
    cond {{
      %four = stablehlo.constant dense<4> : tensor<i32>
      %continue = stablehlo.compare LT, %i, %four : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %continue : tensor<i1>
    }} do {{
      %c0 = stablehlo.constant dense<0> : tensor<i32>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %two = stablehlo.constant dense<2> : tensor<i32>
      %core = stablehlo.constant dense<{5}> : tensor<i32>
      %halo = stablehlo.constant dense<{6}> : tensor<i32>
      %limit = stablehlo.constant dense<{7}> : tensor<i32>
      %base = stablehlo.multiply %i, %core : tensor<i32>
      %wanted = stablehlo.subtract %base, %halo : tensor<i32>
      %positive = stablehlo.maximum %wanted, %c0 : tensor<i32>
      %start = stablehlo.minimum %positive, %limit : tensor<i32>
      %tile = "stablehlo.dynamic_slice"(%source, %c0, %c0, %start, %c0) {{slice_sizes = array<i64: 1, 3, {8}, {9}>}} : ({0}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> {2}
      %pixels = func.call @kirari_span_bande(%tile) : ({2}) -> {3}
      %offset = stablehlo.subtract %base, %start : tensor<i32>
      %local = stablehlo.multiply %offset, %two : tensor<i32>
      %dst = stablehlo.multiply %base, %two : tensor<i32>
      %piece = "stablehlo.dynamic_slice"(%pixels, %c0, %c0, %local, %c0) {{slice_sizes = array<i64: 1, 3, {10}, {11}>}} : ({3}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> {4}
      %updated = "stablehlo.dynamic_update_slice"(%old, %piece, %c0, %c0, %dst, %c0) : ({1}, {4}, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> {1}
      %next = stablehlo.add %i, %one : tensor<i32>
      stablehlo.return %next, %updated, %source : tensor<i32>, {1}, {0}
    }}
    return %output : {1}
  }}
  )",
                                      input, output, tile, pixels, piece, p.core, p.halo, p.height - p.window,
                                      p.window, p.width, 2 * p.core, 2 * p.width));
  return tile_module;
}
}  // namespace kirari::gpu::zml
