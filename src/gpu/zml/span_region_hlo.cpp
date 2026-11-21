// Boucle de fenêtres mobiles du SPAN Balanced. Le SDK ZML émet la fonction privée du réseau
// sur 200 × 360 pixels. Le petit argument de contrôle porte le masque FP16 (0 ou 1), les 24
// couples (y,x) et leur effectif, ces derniers en mots u16 lus sans conversion flottante.
#include "gpu/zml/span_region_hlo.h"

#include <format>

namespace kirari::gpu::zml {
std::optional<std::string> wrap_span_regions(std::string tile, int h, int w) {
  constexpr std::string_view decl = "  func.func @main(";
  const auto pos = tile.find(decl);
  if (pos == std::string::npos) return {};
  const std::string_view signature = std::string_view(tile).substr(pos, tile.find('\n', pos) - pos);
  if (!signature.contains("tensor<1x3x200x360xf16>") || !signature.contains(" -> tensor<1x3x400x720xf16>"))
    return {};
  const int rows = (h + 15) / 16, cols = (w + 63) / 64, count = rows * cols;
  tile.replace(pos, decl.size(), "  func.func private @span(");
  tile.insert(
      pos,
      std::format(
          R"mlir(  func.func @main(%image: tensor<1x3x{0}x{1}xf16>, %previous: tensor<1x3x{2}x{3}xf16>, %metadata: tensor<1x1x1x{8}xf16>) -> tensor<1x3x{2}x{3}xf16> {{
    %maskflat = stablehlo.slice %metadata [0:1, 0:1, 0:1, 0:{6}] : (tensor<1x1x1x{8}xf16>) -> tensor<1x1x1x{6}xf16>
    %mask = stablehlo.reshape %maskflat : (tensor<1x1x1x{6}xf16>) -> tensor<1x1x{4}x{5}xf16>
    %positionsflat = stablehlo.slice %metadata [0:1, 0:1, 0:1, {6}:{7}] : (tensor<1x1x1x{8}xf16>) -> tensor<1x1x1x48xf16>
    %positions = stablehlo.reshape %positionsflat : (tensor<1x1x1x48xf16>) -> tensor<1x1x24x2xf16>
    %countwordflat = stablehlo.slice %metadata [0:1, 0:1, 0:1, {7}:{8}] : (tensor<1x1x1x{8}xf16>) -> tensor<1x1x1x1xf16>
    %countword = stablehlo.reshape %countwordflat : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %countbits = stablehlo.bitcast_convert %countword : (tensor<f16>) -> tensor<ui16>
    %count = stablehlo.convert %countbits : (tensor<ui16>) -> tensor<i32>
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %end, %result = stablehlo.while(%i = %zero, %acc = %previous) : tensor<i32>, tensor<1x3x{2}x{3}xf16>
    cond {{
      %more = stablehlo.compare LT, %i, %count : (tensor<i32>, tensor<i32>) -> tensor<i1>
      stablehlo.return %more : tensor<i1>
    }} do {{
      %next = func.call @region(%image, %acc, %mask, %positions, %i) : (tensor<1x3x{0}x{1}xf16>, tensor<1x3x{2}x{3}xf16>, tensor<1x1x{4}x{5}xf16>, tensor<1x1x24x2xf16>, tensor<i32>) -> tensor<1x3x{2}x{3}xf16>
      %one = stablehlo.constant dense<1> : tensor<i32>
      %step = stablehlo.add %i, %one : tensor<i32>
      stablehlo.return %step, %next : tensor<i32>, tensor<1x3x{2}x{3}xf16>
    }}
    return %result : tensor<1x3x{2}x{3}xf16>
  }}
  func.func private @region(%image: tensor<1x3x{0}x{1}xf16>, %old: tensor<1x3x{2}x{3}xf16>, %mask: tensor<1x1x{4}x{5}xf16>, %positions: tensor<1x1x24x2xf16>, %i: tensor<i32>) -> tensor<1x3x{2}x{3}xf16> {{
    %zero = stablehlo.constant dense<0> : tensor<i32>
    %one = stablehlo.constant dense<1> : tensor<i32>
    %two = stablehlo.constant dense<2> : tensor<i32>
    %halo = stablehlo.constant dense<52> : tensor<i32>
    %yword = "stablehlo.dynamic_slice"(%positions, %zero, %zero, %i, %zero) {{slice_sizes = array<i64: 1, 1, 1, 1>}} : (tensor<1x1x24x2xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x1xf16>
    %yscalar = stablehlo.reshape %yword : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %ybits = stablehlo.bitcast_convert %yscalar : (tensor<f16>) -> tensor<ui16>
    %y = stablehlo.convert %ybits : (tensor<ui16>) -> tensor<i32>
    %ylimit = stablehlo.constant dense<{9}> : tensor<i32>
    %ywanted = stablehlo.subtract %y, %halo : tensor<i32>
    %ystart = stablehlo.clamp %zero, %ywanted, %ylimit : tensor<i32>
    %yoffset = stablehlo.subtract %y, %ystart : tensor<i32>
    %ylocal = stablehlo.multiply %yoffset, %two : tensor<i32>
    %ydest = stablehlo.multiply %y, %two : tensor<i32>
    %xword = "stablehlo.dynamic_slice"(%positions, %zero, %zero, %i, %one) {{slice_sizes = array<i64: 1, 1, 1, 1>}} : (tensor<1x1x24x2xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x1x1xf16>
    %xscalar = stablehlo.reshape %xword : (tensor<1x1x1x1xf16>) -> tensor<f16>
    %xbits = stablehlo.bitcast_convert %xscalar : (tensor<f16>) -> tensor<ui16>
    %x = stablehlo.convert %xbits : (tensor<ui16>) -> tensor<i32>
    %xlimit = stablehlo.constant dense<{10}> : tensor<i32>
    %xwanted = stablehlo.subtract %x, %halo : tensor<i32>
    %xstart = stablehlo.clamp %zero, %xwanted, %xlimit : tensor<i32>
    %xoffset = stablehlo.subtract %x, %xstart : tensor<i32>
    %xlocal = stablehlo.multiply %xoffset, %two : tensor<i32>
    %xdest = stablehlo.multiply %x, %two : tensor<i32>
    %crop = "stablehlo.dynamic_slice"(%image, %zero, %zero, %ystart, %xstart) {{slice_sizes = array<i64: 1, 3, 200, 360>}} : (tensor<1x3x{0}x{1}xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x3x200x360xf16>
    %net = func.call @span(%crop) : (tensor<1x3x200x360xf16>) -> tensor<1x3x400x720xf16>
    %fresh = "stablehlo.dynamic_slice"(%net, %zero, %zero, %ylocal, %xlocal) {{slice_sizes = array<i64: 1, 3, 192, 512>}} : (tensor<1x3x400x720xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x3x192x512xf16>
    %repeat = stablehlo.broadcast_in_dim %mask, dims = [0, 1, 2, 4] : (tensor<1x1x{4}x{5}xf16>) -> tensor<1x1x{4}x32x{5}x128xf16>
    %raster = stablehlo.reshape %repeat : (tensor<1x1x{4}x32x{5}x128xf16>) -> tensor<1x1x{11}x{12}xf16>
    %core = "stablehlo.dynamic_slice"(%raster, %zero, %zero, %ydest, %xdest) {{slice_sizes = array<i64: 1, 1, 192, 512>}} : (tensor<1x1x{11}x{12}xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x1x192x512xf16>
    %off = stablehlo.constant dense<0.000000e+00> : tensor<1x1x192x512xf16>
    %on = stablehlo.compare NE, %core, %off, FLOAT : (tensor<1x1x192x512xf16>, tensor<1x1x192x512xf16>) -> tensor<1x1x192x512xi1>
    %write = stablehlo.broadcast_in_dim %on, dims = [0, 1, 2, 3] : (tensor<1x1x192x512xi1>) -> tensor<1x3x192x512xi1>
    %kept = "stablehlo.dynamic_slice"(%old, %zero, %zero, %ydest, %xdest) {{slice_sizes = array<i64: 1, 3, 192, 512>}} : (tensor<1x3x{2}x{3}xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x3x192x512xf16>
    %pixels = stablehlo.select %write, %fresh, %kept : tensor<1x3x192x512xi1>, tensor<1x3x192x512xf16>
    %next = "stablehlo.dynamic_update_slice"(%old, %pixels, %zero, %zero, %ydest, %xdest) : (tensor<1x3x{2}x{3}xf16>, tensor<1x3x192x512xf16>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<1x3x{2}x{3}xf16>
    return %next : tensor<1x3x{2}x{3}xf16>
  }}
)mlir",
          h, w, 2 * h, 2 * w, rows, cols, count, count + 48, count + 49, h - 200, w - 360, rows * 32,
          cols * 128));
  return tile;
}
}  // namespace kirari::gpu::zml
