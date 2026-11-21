//! Échantillonnage ONNX de RIFE et DistilDRBA, dans le graphe du compilateur ZML.
const std = @import("std");
const zml = @import("zml/core");
const Tensor = zml.Tensor;

pub const GridOptions = struct {
    mode: enum { bilinear, nearest } = .bilinear,
    padding: enum { border, zeros, reflection } = .border,
    align_corners: bool = true,
    grid_nchw: bool = false,
    interleaved_xy: bool = false,
    integer_zero_margin: bool = false,
};

// Références aux pixels avant leur conversion. Le manifeste ne sérialise que les indices des tenseurs.
pub const GridSources = struct {
    linear: ?Tensor = null,
    central: ?Tensor = null,
};

pub const ResizeOptions = struct {
    mode: enum { linear, nearest } = .linear,
    coordinate: enum { half_pixel, pytorch_half_pixel, align_corners, asymmetric } = .half_pixel,
    nearest: enum { floor, ceil, round_prefer_floor, round_prefer_ceil } = .floor,
    scale_h: ?f32 = null,
    scale_w: ?f32 = null,
    upsample_phases: bool = false,
    drba_direct: bool = false,
};

fn real(value: i64) f32 {
    return @floatFromInt(value);
}

fn clamp(x: Tensor, low: f32, high: f32) Tensor {
    return x.clamp(.scalar(low, .f32), .scalar(high, .f32));
}

// Arrondir dans le repère d'origine. Une marge d'un pixel déplacerait les égalités de nearest-even.
fn sourcePixel(normalized: Tensor, size: i64, options: GridOptions) Tensor {
    const side = real(size);
    var pixel = normalized.addConstant(1);
    pixel = if (options.align_corners) pixel.scale(side - 1).scale(0.5) else pixel.scale(side).subConstant(1).scale(0.5);
    if (options.mode == .nearest) pixel = pixel.roundNearestEven();
    switch (options.padding) {
        .border => return clamp(pixel, 0, side - 1),
        .zeros => {
            const clipped = clamp(pixel, -1, side);
            return if (options.integer_zero_margin and options.mode == .bilinear) clipped else clipped.addConstant(1);
        },
        .reflection => {
            if (size == 1) return Tensor.zeroes(pixel.shape());
            const low: f32 = if (options.align_corners) 0 else -0.5;
            const span = if (options.align_corners) side - 1 else side;
            const period = 2 * span;
            const phase = pixel.subConstant(low).abs().remainderConst(period);
            return clamp(phase.minimum(Tensor.scalar(period, .f32).sub(phase)).addConstant(low), 0, side - 1);
        },
    }
}

fn mix(a: Tensor, b: Tensor, weight: Tensor) Tensor {
    return a.mul(Tensor.scalar(1, .f32).sub(weight)).add(b.mul(weight));
}

// Une fenêtre contient tous les canaux et jusqu'à 2×2 voisins du même lot.
fn corners(image: Tensor, indices: Tensor, central: ?Tensor) [4]Tensor {
    const rows = @min(2, image.dim(.h));
    const cols = @min(2, image.dim(.w));
    // Distil prélève les voisins de la tranche centrale FP16 puis les élargit avant la même interpolation FP32.
    const tile = if (central) |source|
        source.gatherSlices(.{ .h = rows, .w = cols }, indices, .{}).convert(.f32)
    else
        image.gatherSlices(.{ .h = rows, .w = cols }, indices, .{});
    // [N,OH,OW,C,h,w] -> [N,C,OH,OW,h,w]
    const window = tile.transpose(.{ 0, 3, 1, 2, 4, 5 }).convert(.f32);
    const nw = window.slice(5, .single(0)).slice(4, .single(0));
    const ne = window.slice(5, .single(cols - 1)).slice(4, .single(0));
    const sw = window.slice(5, .single(0)).slice(4, .single(rows - 1));
    const se = window.slice(5, .single(cols - 1)).slice(4, .single(rows - 1));
    return .{ nw, ne, sw, se };
}

fn bilinear(image: Tensor, py: Tensor, px: Tensor, central: ?Tensor, margin: i64) Tensor {
    // Au bord la fenêtre se décale d'un cran et le poids du dernier pixel vaut 1.
    const top = py.floor().minimum(.scalar(real(image.dim(.h) - @min(2, image.dim(.h)) - margin), .f32));
    const left = px.floor().minimum(.scalar(real(image.dim(.w) - @min(2, image.dim(.w)) - margin), .f32));
    const origin = Tensor.stack(&.{ top.convert(.i32), left.convert(.i32) }, .last, .coord);
    // GMFlow conserve la fraction dans le repère source. L’ajout de la marge
    // avant floor() peut arrondir un point intérieur jusque sur le pixel suivant.
    const indices = if (margin != 0) origin.addConstant(1) else origin;
    const nw, const ne, const sw, const se = corners(image, indices, central);
    const wx = px.sub(left).broadcast(nw.shape(), &.{ 0, 2, 3 });
    const wy = py.sub(top).broadcast(nw.shape(), &.{ 0, 2, 3 });
    return mix(mix(nw, ne, wx), mix(sw, se, wx), wy).convert(image.dtype());
}

pub fn gridSampleD2(image: Tensor, grid: Tensor) !Tensor {
    if (image.dtype() != .f32 or grid.dtype() != .f32 or
        !@import("std").mem.eql(i64, image.dims(), &.{ 2, 7, 768, 1280 }) or
        !@import("std").mem.eql(i64, grid.dims(), &.{ 2, 768, 1280, 2 })) return error.InvalidRifeD2Grid;
    const xy = grid.withTags(.{ .n, .oh, .ow, .coord });
    const gx = xy.slice(.coord, .single(0));
    const gy = xy.slice(.coord, .single(1));
    const foot = @import("rife_warp_d2.zig").footprint(sourcePixel(gx, 1280, .{}), sourcePixel(gy, 768, .{}), gx.isFinite().logical(.AND, gy.isFinite()));
    const nw, const ne, const sw, const se = corners(image.withTags(.{ .n, .c, .h, .w }), foot.indices, null);
    const wx = foot.wx.broadcast(nw.shape(), &.{ 0, 2, 3 });
    const wy = foot.wy.broadcast(nw.shape(), &.{ 0, 2, 3 });
    const value = mix(mix(nw, ne, wx), mix(sw, se, wx), wy);
    return foot.valid.broadcast(value.shape(), &.{ 0, 2, 3 }).select(value, .scalar(0, .f32)).withTags(image.shape().tags());
}

fn linearBorder(raw: Tensor, py: Tensor, px: Tensor) Tensor {
    const n = raw.dim(0);
    const h = raw.dim(2);
    const w = raw.dim(3);
    const top = py.floor().minimum(.scalar(real(h - 2), .f32));
    const left = px.floor().minimum(.scalar(real(w - 2), .f32));
    const batch = Tensor.arange(.{ .end = n }, .i32).broadcast(top.shape(), &.{0});
    const base = top.convert(.i32).scale(w).add(left.convert(.i32)).scale(n).add(batch).transpose(.{ 1, 2, 0 });
    const row = w * n;
    const ids = Tensor.stack(&.{ base, base.addConstant(n), base.addConstant(row), base.addConstant(row + n) }, .last, .corner);
    const table = raw.transpose(.{ 2, 3, 0, 1 }).reshape(.{ .pixel = h * w * n, .c = raw.dim(1) });
    const tile = table.gather(.{ .pixel = ids }, .{}).convert(.f32);
    const nw = tile.slice(.corner, .single(0));
    const ne = tile.slice(.corner, .single(1));
    const sw = tile.slice(.corner, .single(2));
    const se = tile.slice(.corner, .single(3));
    const wx = px.sub(left).transpose(.{ 1, 2, 0 }).broadcast(nw.shape(), &.{ 0, 1, 2 });
    const wy = py.sub(top).transpose(.{ 1, 2, 0 }).broadcast(nw.shape(), &.{ 0, 1, 2 });
    return mix(mix(nw, ne, wx), mix(sw, se, wx), wy).transpose(.{ 2, 3, 0, 1 });
}

pub fn gridSample(image: Tensor, grid: Tensor, options: GridOptions) Tensor {
    return gridSampleFrom(image, grid, options, .{});
}

// Une seule grille XY FP32 entre la construction des coordonnées et le warp du Resize ×½.
// Les coordonnées valides sont dans [0,taille−1], donc −1 représente exactement un axe non fini.
fn interleavedBorder(image: Tensor, xy: Tensor) Tensor {
    const sizes = Tensor.stack(&.{
        Tensor.scalar(real(image.dim(.w)) - 1, .f32),
        Tensor.scalar(real(image.dim(.h)) - 1, .f32),
    }, .last, .coord).broadcast(xy.shape(), &.{3});
    const zero = Tensor.scalar(0, .f32);
    const pixels = xy.addConstant(1).mul(sizes).scale(0.5).clamp(zero, sizes);
    const coords = xy.isFinite().select(pixels, .scalar(-1, .f32)).optimizationBarrier();
    const px = coords.slice(.coord, .single(0));
    const py = coords.slice(.coord, .single(1));
    const samples = bilinear(image, py.maximum(zero), px.maximum(zero), null, 0);
    const valid = px.cmp(.GE, zero).logical(.AND, py.cmp(.GE, zero)).broadcast(samples.shape(), &.{ 0, 2, 3 });
    return valid.select(samples, .scalar(0, samples.dtype()));
}

// Le manifeste ne propose raw que pour le warp RGB + traits reconnu avant le dernier IFBlock.
// raw précède la conversion FP16 → FP32 de l’image ; l'interface et l'arrondi du résultat restent ceux de image.
pub fn gridSampleFrom(image: Tensor, grid: Tensor, options: GridOptions, sources: GridSources) Tensor {
    var x = image.withTags(.{ .n, .c, .h, .w });
    const xy = (if (options.grid_nchw) grid.transpose(.{ 0, 2, 3, 1 }) else grid)
        .withTags(.{ .n, .oh, .ow, .coord }).convert(.f32);
    if (options.interleaved_xy and options.mode == .bilinear and options.padding == .border and options.align_corners and
        x.dim(.n) == 2 and x.dim(.c) == 7 and x.dtype() == .f32 and xy.dim(.oh) == x.dim(.h) and xy.dim(.ow) == x.dim(.w))
        return interleavedBorder(x, xy).withTags(image.shape().tags());
    const gx = xy.slice(.coord, .single(0));
    const gy = xy.slice(.coord, .single(1));
    const px = sourcePixel(gx, x.dim(.w), options);
    const py = sourcePixel(gy, x.dim(.h), options);
    if (options.padding == .zeros) {
        const margin = Tensor.Pad{ .low = 1, .high = 1 };
        x = x.pad(0, .{ .h = margin, .w = margin });
    }
    const sampled = switch (options.mode) {
        .bilinear => blk: {
            if (sources.linear) |pixels| {
                const h = x.dim(.h);
                const w = x.dim(.w);
                if (options.padding == .border and options.align_corners and pixels.dtype() == .f16 and
                    x.dim(.n) == 2 and x.dim(.c) == 7 and h >= 2 and w >= 2 and h <= 1 << 24 and w <= 1 << 24 and
                    h <= @divTrunc(@as(i64, 0x7fffffff), 2 * w) and py.dim(1) == h and py.dim(2) == w and
                    pixels.dim(0) == 2 and pixels.dim(1) == 7 and pixels.dim(2) == h and pixels.dim(3) == w)
                    break :blk linearBorder(pixels, py, px).convert(x.dtype());
            }
            break :blk bilinear(x, py, px, sources.central, @intFromBool(options.padding == .zeros and options.integer_zero_margin));
        },
        .nearest => x.gather(.{ .h = py.convert(.i32), .w = px.convert(.i32) }, .{}),
    };
    const finite = gx.isFinite().logical(.AND, gy.isFinite()).broadcast(sampled.shape(), &.{ 0, 2, 3 });
    return finite.select(sampled, .scalar(0, sampled.dtype())).withTags(image.shape().tags());
}

fn take(image: Tensor, indices: Tensor, comptime axis: u3) Tensor {
    const i = indices.convert(.i32).withTags(.{.destination});
    const gathered = if (axis == 2)
        image.gather(.{ .h = i }, .{ .indices_are_sorted = true })
    else
        image.gather(.{ .w = i }, .{ .indices_are_sorted = true });
    return gathered.withTags(image.shape().tags());
}

// Les cartes de flot utilisent align_corners. Les coordonnées sont fixes pour chaque
// programme ; construire les fractions FP32 au traçage préserve la division ONNX.
fn alignedAxis(x: Tensor, comptime axis: u3, dest: i64, alloc: std.mem.Allocator) !Tensor {
    const source = x.dim(axis);
    if (source == dest) return x;
    const count: usize = @intCast(dest);
    const lo = try alloc.alloc(i32, count);
    const hi = try alloc.alloc(i32, count);
    const fraction = try alloc.alloc(f32, count);
    const end: f32 = @floatFromInt(source - 1);
    const denominator: f32 = @floatFromInt(@max(1, dest - 1));
    for (lo, hi, fraction, 0..) |*a, *b, *f, k| {
        const pos = @min(@as(f32, @floatFromInt(k)) * end / denominator, end);
        a.* = @intFromFloat(pos);
        b.* = @min(a.* + 1, @as(i32, @intCast(source - 1)));
        f.* = pos - @as(f32, @floatFromInt(a.*));
    }
    const sh = zml.Shape.init(.{dest}, .i32);
    const il = Tensor.constantTensor(sh, std.mem.sliceAsBytes(lo));
    const ir = Tensor.constantTensor(sh, std.mem.sliceAsBytes(hi));
    const left = take(x, il, axis);
    const right = take(x, ir, axis);
    const w = Tensor.constantTensor(.init(.{dest}, .f32), std.mem.sliceAsBytes(fraction)).broadcast(left.shape(), &.{axis});
    return Tensor.scalar(1, .f32).sub(w).mul(left).add(w.mul(right));
}

pub fn resizeAligned(image: Tensor, h: i64, w: i64, alloc: std.mem.Allocator) !Tensor {
    const input = image.convert(.f32).withTags(.{ .n, .c, .h, .w });
    const horizontal = try alignedAxis(input, 3, w, alloc);
    return (try alignedAxis(horizontal, 2, h, alloc)).convert(image.dtype()).withTags(image.shape().tags());
}

// Axe de phases inséré après l'axe spatial : x0, puis les r positions
// entre x0 et x1, entre x1 et x2, etc. Les bords ONNX restent exacts.
fn upsamplePhases(x: Tensor, comptime axis: u3, factor: i64) Tensor {
    const size = x.dim(axis);
    const axes: []const i64 = &.{ 0, 1, 2, 3 };
    if (size == 1) return x.broadcast(x.shape().setDim(axis, factor), axes);
    const left = x.slice(axis, .{ .end = size - 1 });
    const right = x.slice(axis, .{ .start = 1, .end = size });
    const shape = left.shape().insert(axis + 1, .{factor});
    const source_axes: []const i64 = if (axis == 2) &.{ 0, 1, 2, 4 } else axes;
    const weight = Tensor.arange(.{ .end = factor }, .f32).addConstant(0.5).divByConst(real(factor));
    const mid = mix(left.broadcast(shape, source_axes), right.broadcast(shape, source_axes), weight.broadcast(shape, &.{axis + 1})).reshape(x.shape().setDim(axis, (size - 1) * factor));
    const edge = x.shape().setDim(axis, @divExact(factor, 2));
    const head = x.slice(axis, .{ .end = 1 }).broadcast(edge, axes);
    const tail = x.slice(axis, .{ .start = size - 1, .end = size }).broadcast(edge, axes);
    return Tensor.concatenate(&.{ head, mid, tail }, axis);
}

fn resizeAxis(image: Tensor, comptime axis: u3, dest: i64, scale: f32, options: ResizeOptions) Tensor {
    const source = image.dim(axis);
    if (source == dest and scale == 1) return image;
    const ratio = @divTrunc(source, dest);
    const half = options.coordinate == .half_pixel or (options.coordinate == .pytorch_half_pixel and dest > 1);
    const factor = @divTrunc(dest, source);
    if (options.upsample_phases and half and options.mode == .linear and
        (factor == 2 or factor == 4 or factor == 8 or factor == 16) and
        scale == real(factor) and dest == source * factor and dest <= 1 << 23)
        return upsamplePhases(image, axis, factor);
    if (half and options.mode == .linear and ratio >= 2 and @mod(ratio, 2) == 0 and
        source == dest * ratio and scale == 1 / real(ratio))
    {
        const start = @divExact(ratio, 2) - 1;
        const a = image.slice(axis, .{ .start = start, .step = @intCast(ratio) });
        const b = image.slice(axis, .{ .start = start + 1, .step = @intCast(ratio) });
        return mix(a, b, Tensor.scalar(0.5, .f32));
    }
    const dst = Tensor.arange(.{ .end = dest }, .f32);
    const pos = switch (options.coordinate) {
        .half_pixel => dst.addConstant(0.5).divByConst(scale).subConstant(0.5),
        .pytorch_half_pixel => if (dest > 1) dst.addConstant(0.5).divByConst(scale).subConstant(0.5) else dst,
        .align_corners => if (dest > 1) dst.scale(real(source - 1)).divByConst(real(dest - 1)) else dst,
        .asymmetric => dst.divByConst(scale),
    };
    const p = clamp(pos, 0, real(source - 1));
    const lo = p.floor();
    const fraction = p.sub(lo);
    if (options.mode == .nearest) {
        const index = switch (options.nearest) {
            .floor => lo,
            .ceil => p.ceil(),
            .round_prefer_floor => lo.add(fraction.cmp(.GT, .scalar(0.5, .f32)).convert(.f32)),
            .round_prefer_ceil => lo.add(fraction.cmp(.GE, .scalar(0.5, .f32)).convert(.f32)),
        };
        return take(image, index, axis);
    }
    const a = take(image, lo, axis);
    const b = take(image, lo.addConstant(1).minimum(.scalar(real(source - 1), .f32)), axis);
    return mix(a, b, fraction.broadcast(a.shape(), &.{axis}));
}

fn drbaDirectResize(image: Tensor, height: i64, width: i64, options: ResizeOptions) ?Tensor {
    const h = image.dim(2);
    const w = image.dim(3);
    if (options.mode != .linear or (image.dtype() != .f32 and image.dtype() != .f16) or
        h <= 0 or w <= 0 or h > 1 << 23 or w > 1 << 23 or height <= 0 or width <= 0) return null;
    if (options.coordinate != .half_pixel and
        !(options.coordinate == .pytorch_half_pixel and height > 1 and width > 1)) return null;
    const r = @divTrunc(h, height);
    if ((r != 4 and r != 8) or @rem(h, r) != 0 or @rem(w, r) != 0 or
        @divExact(h, r) != height or @divExact(w, r) != width) return null;
    const scale = 1 / real(r);
    if ((options.scale_h orelse scale) != scale or (options.scale_w orelse scale) != scale) return null;

    // Chacune des quatre slices a déjà la forme de sortie. L'arrondi FP32
    // intervient aux mêmes opérations que dans le filtre séparable horizontal puis vertical.
    const first = @divExact(r, 2) - 1;
    var rows: [2]Tensor = undefined;
    const half = Tensor.scalar(0.5, .f32);
    for (0..2) |row| {
        var taps: [2]Tensor = undefined;
        for (&taps, 0..) |*tap, col| {
            tap.* = image.slices(.{ 2, 3 }, &.{
                .{ .start = first + @as(i64, @intCast(row)), .end = h, .step = @intCast(r) },
                .{ .start = first + @as(i64, @intCast(col)), .end = w, .step = @intCast(r) },
            }).convert(.f32);
        }
        rows[row] = mix(taps[0], taps[1], half);
    }
    return mix(rows[0], rows[1], half).convert(image.dtype()).withTags(image.shape().tags());
}

pub fn resize(image: Tensor, height: i64, width: i64, options: ResizeOptions) Tensor {
    if (options.drba_direct) {
        if (drbaDirectResize(image, height, width, options)) |y| return y;
    }
    var x = image.withTags(.{ .n, .c, .h, .w });
    if (options.mode == .linear) x = x.convert(.f32);
    x = resizeAxis(x, 3, width, options.scale_w orelse real(width) / real(x.dim(3)), options);
    x = resizeAxis(x, 2, height, options.scale_h orelse real(height) / real(x.dim(2)), options);
    return x.convert(image.dtype()).withTags(image.shape().tags());
}

// Un Resize half_pixel d'un facteur pair lit seulement quatre pixels au centre de chaque cellule.
// RIFE peut appliquer ceci à la grille avant le warp pour ses étages au 1/4 et au 1/8.
pub fn compactCenters(image: Tensor, factor: i64, grid_nhw2: bool) Tensor {
    const x = if (grid_nhw2) image.transpose(.{ 0, 3, 1, 2 }) else image;
    const oh = @divExact(x.dim(2), factor);
    const ow = @divExact(x.dim(3), factor);
    const center = Tensor.Slice{ .start = @divExact(factor, 2) - 1, .end = @divExact(factor, 2) + 1 };
    const compact = x.reshape(.{ x.dim(0), x.dim(1), oh, factor, ow, factor })
        .slice(3, center).slice(5, center).reshape(.{ x.dim(0), x.dim(1), 2 * oh, 2 * ow });
    return (if (grid_nhw2) compact.transpose(.{ 0, 2, 3, 1 }) else compact).withTags(image.shape().tags());
}

// Le D4 de RIFE 4.25 lite place les sept canaux de A puis les sept de B dans le
// premier Reshape. Le Resize half_pixel ×¼ ne lit que les sites 4j+1 et 4j+2
// dans chaque axe. Leur moyenne FP32 suit le même arrondi FP16 après le warp.
pub fn rife425LiteD4(image: Tensor, grid: Tensor, time: Tensor, attributes: Tensor, flow: Tensor) !Tensor {
    const h = 768;
    const w = 1280;
    for ([_]Tensor{ image, grid, time, attributes, flow }, [_][4]i64{
        .{ 2, 7, h, w }, .{ 2, h, w, 2 }, .{ 1, 1, h, w }, .{ 1, 9, h, w }, .{ 1, 4, h, w },
    }, [_]zml.DataType{ .f32, .f32, .f16, .f16, .f16 }) |input, dims, dtype| {
        if (input.dtype() != dtype or !std.mem.eql(i64, input.dims(), &dims)) return error.InvalidRifeD4;
    }
    const coordinates = compactCenters(grid, 4, true);
    const warped = gridSample(image, coordinates, .{}).convert(.f16);
    const channels = Tensor.concatenate(&.{
        warped.reshape(.{ 1, 14, h / 2, w / 2 }),
        compactCenters(time, 4, false),
        compactCenters(attributes, 4, false),
        compactCenters(flow, 4, false),
    }, 1);
    return resize(channels.convert(.f32), h / 4, w / 4, .{ .scale_h = 0.5, .scale_w = 0.5 }).convert(.f16);
}

// Les deux Reshape du modèle placent les images successives sur les canaux. XLA ne garde le warp
// pleine taille que si une autre sortie vivante le lit ; la réduction ne lui demande que sa grille compacte.
pub fn reducedWarpPlanes(image: Tensor, grid: Tensor, options: GridOptions, factor: i64, first_channels: i64, parts: []Tensor, round_warp: bool) !Tensor {
    if (parts.len < 2 or first_channels <= 0 or first_channels >= image.dim(1) or factor < 4 or @mod(factor, 2) != 0)
        return error.InvalidWarpReduction;
    const source_h = grid.dim(if (options.grid_nchw) @as(u3, 2) else @as(u3, 1));
    const source_w = grid.dim(if (options.grid_nchw) @as(u3, 3) else @as(u3, 2));
    if (source_h <= 0 or source_w <= 0 or @mod(source_h, factor) != 0 or @mod(source_w, factor) != 0)
        return error.InvalidWarpReduction;
    for (parts) |x| {
        if (x.rank() != 4 or x.dim(2) != source_h or x.dim(3) != source_w) return error.InvalidWarpReduction;
    }
    const reduced = compactCenters(grid, factor, !options.grid_nchw);
    var warped = gridSample(image, reduced, options);
    if (round_warp) warped = warped.convert(.f16);
    warped = warped.convert(parts[0].dtype());
    for (parts, 0..) |*part, i| {
        if (i < 2) {
            const start = if (i == 0) 0 else first_channels;
            const end = if (i == 0) first_channels else image.dim(1);
            if (part.dim(0) * part.dim(1) != warped.dim(0) * (end - start)) return error.InvalidWarpReduction;
            part.* = warped.slice(1, .{ .start = start, .end = end })
                .reshape(.{ part.dim(0), part.dim(1), warped.dim(2), warped.dim(3) });
        } else {
            part.* = compactCenters(part.*, factor, false);
        }
    }
    return Tensor.concatenate(parts, 1);
}

// Les RGB restent dans deux tableaux plans FP16. Chaque gather lit les trois
// canaux du même pied au moyen d'un seul indice spatial.
pub const OutputSide = struct {
    rgb: Tensor,
    origin: [2]Tensor,
    unit: [2]Tensor,
};

const Footprint = struct {
    index: [4]Tensor,
    weight_x: Tensor,
    weight_y: Tensor,

    fn fromAxes(top: Tensor, bottom: Tensor, left: Tensor, right: Tensor, wx: Tensor, wy: Tensor, width: i64) Footprint {
        const first = top.convert(.i32).scale(width);
        const second = bottom.convert(.i32).scale(width);
        const l = left.convert(.i32);
        const r = right.convert(.i32);
        return .{
            .index = .{ first.add(l), first.add(r), second.add(l), second.add(r) },
            .weight_x = wx,
            .weight_y = wy,
        };
    }

    fn interpolate(f: Footprint, image: Tensor) Tensor {
        const table = image.reshape(.{ .c = image.dim(1), .pixel = image.dim(2) * image.dim(3) });
        var taps: [4]Tensor = undefined;
        for (&taps, f.index) |*tap, i| tap.* = table.gather(.{ .pixel = i }, .{}).convert(.f32);
        const wx = f.weight_x.broadcast(taps[0].shape(), &.{ 1, 2 });
        const wy = f.weight_y.broadcast(taps[0].shape(), &.{ 1, 2 });
        return mix(mix(taps[0], taps[1], wx), mix(taps[2], taps[3], wx), wy);
    }
};

// Indices de l'agrandissement half_pixel ×4. À la dernière ligne/colonne, les deux
// pieds peuvent être identiques (comme Resize), alors que GridSample décale sa fenêtre.
fn decoderAxis(source: i64, target: i64) Tensor {
    const center = Tensor.arange(.{ .end = target }, .f32).addConstant(0.5).divByConst(4).subConstant(0.5);
    return clamp(center, 0, real(source - 1));
}

pub fn drbaOutputWarp(head: Tensor, sides: [2]OutputSide) !Tensor {
    if (head.rank() != 4 or head.dtype() != .f16 or head.dim(0) != 1 or head.dim(1) != 5)
        return error.InvalidDrbaOutput;
    const h = head.dim(2) * 4;
    const w = head.dim(3) * 4;
    if (h <= 0 or w <= 0 or h > 1 << 23 or w > 1 << 23 or h > @divTrunc(@as(i64, 0x7fffffff), w)) return error.InvalidDrbaOutput;
    for (sides) |side| {
        if (side.rgb.rank() != 4 or side.rgb.dtype() != .f16 or side.rgb.dim(0) != 1 or side.rgb.dim(1) != 3 or side.rgb.dim(2) != h or side.rgb.dim(3) != w)
            return error.InvalidDrbaOutput;
        for (0..2) |axis| {
            const origin = side.origin[axis];
            if (origin.rank() != 4 or origin.dtype() != .f32 or origin.dim(0) != 1 or origin.dim(1) != 1 or
                origin.dim(2) != (if (axis == 0) @as(i64, 1) else h) or origin.dim(3) != (if (axis == 0) w else @as(i64, 1)) or
                side.unit[axis].rank() != 0 or side.unit[axis].dtype() != .f32)
                return error.InvalidDrbaOutput;
        }
    }
    const shape = zml.Shape.init(.{ .oh = h, .ow = w }, .f32);
    const x = decoderAxis(head.dim(3), w).broadcast(shape, &.{1});
    const y = decoderAxis(head.dim(2), h).broadcast(shape, &.{0});
    const left = x.floor();
    const top = y.floor();
    const right = left.addConstant(1).minimum(.scalar(real(head.dim(3) - 1), .f32));
    const bottom = top.addConstant(1).minimum(.scalar(real(head.dim(2) - 1), .f32));
    const resize_foot = Footprint.fromAxes(top, bottom, left, right, x.sub(left), y.sub(top), head.dim(3));
    // Calcul séparé par plan : aucune image du flot étendu ni concaténation des grilles.
    const decoded_mask = resize_foot.interpolate(head.slice(1, .{ .start = 4, .end = 5 })).convert(.f16);
    const mask = decoded_mask.sigmoid();
    var weighted: [2]Tensor = undefined;
    for (sides, 0..) |side, k| {
        var normalized: [2]Tensor = undefined;
        for (0..2) |axis| {
            const c: i64 = @intCast(k * 2 + axis);
            const flow = resize_foot.interpolate(head.slice(1, .{ .start = c, .end = c + 1 })).reshape(shape)
                .convert(.f16).scale(4).convert(.f32);
            const origin = side.origin[axis].reshape(.{if (axis == 0) w else h});
            const base = origin.broadcast(shape, if (axis == 0) &.{1} else &.{0});
            normalized[axis] = base.add(flow.mul(side.unit[axis]));
        }
        const px = sourcePixel(normalized[0], w, .{});
        const py = sourcePixel(normalized[1], h, .{});
        const l = px.floor().minimum(.scalar(real(w - 2), .f32));
        const t = py.floor().minimum(.scalar(real(h - 2), .f32));
        const warp_foot = Footprint.fromAxes(t, t.addConstant(1), l, l.addConstant(1), px.sub(l), py.sub(t), w);
        const sample = warp_foot.interpolate(side.rgb);
        const finite = normalized[0].isFinite().logical(.AND, normalized[1].isFinite()).broadcast(sample.shape(), &.{ 1, 2 });
        const rgb = finite.select(sample, .scalar(0, .f32)).convert(.f16);
        const alpha = if (k == 0) mask else Tensor.scalar(1, .f16).sub(mask);
        weighted[k] = rgb.mul(alpha.broadcast(rgb.shape(), &.{ 0, 1, 2 }));
    }
    return weighted[0].add(weighted[1]).reshape(sides[0].rgb.shape());
}
