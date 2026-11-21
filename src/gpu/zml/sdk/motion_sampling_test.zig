const std = @import("std");
const zml = @import("zml/core");
const ops = @import("motion_sampling.zig");
const Tensor = zml.Tensor;

fn compactDifference(image: Tensor, coords: Tensor) Tensor {
    const pair = Tensor.concatenate(&.{ image, image.addConstant(20) }, 1);
    const grid = coords.broadcast(.init(.{ 2, 8, 8, 2 }, .f32), &.{ 0, 1, 2, 3 });
    const full = ops.gridSample(pair, grid, .{}).convert(.f16).convert(.f32);
    var parts = [_]Tensor{
        full.slice(1, .single(0)).reshape(.{ 1, 2, 8, 8 }),
        full.slice(1, .single(1)).reshape(.{ 1, 2, 8, 8 }),
        Tensor.arange(.{ .end = 64 }, .f32).reshape(.{ 1, 1, 8, 8 }),
    };
    const dense = ops.resize(Tensor.concatenate(&parts, 1), 2, 2, .{});
    const compact = ops.reducedWarpPlanes(pair, grid, .{}, 4, 1, &parts, true) catch unreachable;
    return dense.sub(ops.resize(compact, 2, 2, .{})).abs().flatten().max(0).reshape(.{1});
}

fn phasedDifference(image: Tensor) Tensor {
    const two_rows = image.reshape(.{ 1, 1, 2, 3 });
    var error_max = Tensor.scalar(0, .f32);
    for ([_]i64{ 2, 4, 8, 16 }) |factor| {
        const height = 2 * factor;
        const width = 3 * factor;
        const reference = ops.resize(two_rows, height, width, .{});
        const phased = ops.resize(two_rows, height, width, .{ .upsample_phases = true });
        error_max = error_max.maximum(reference.sub(phased).abs().flatten().max(0));
    }
    const reference = ops.resize(image, 4, 6, .{});
    const phased = ops.resize(image, 4, 6, .{ .upsample_phases = true });
    return error_max.maximum(reference.sub(phased).abs().flatten().max(0)).reshape(.{1});
}

fn linearWarpDifference(probe: Tensor, height: i64, width: i64) Tensor {
    // Motifs distincts par image et par canal. À ce niveau FP16, arrondir un pied
    // de filtre au lieu du résultat logique FP32 produit un écart visible.
    const raw = Tensor.arange(.{ .end = 14 * height * width }, .f32).scale(3).addConstant(1500)
        .mul(probe.flatten().slice(0, .single(0))).reshape(.{ 2, 7, height, width }).convert(.f16);
    const at = Tensor.arange(.{ .end = 2 * height * width }, .f32).reshape(.{ 2, height, width });
    const gx = at.remainderConst(9).subConstant(4).scale(0.375);
    const gy = at.scale(5).remainderConst(7).subConstant(3).scale(0.5);
    const coords = Tensor.stack(&.{ gx, gy }, .last, .coord);
    const expected = ops.gridSample(raw.convert(.f32), coords, .{});
    const linear = ops.gridSampleFrom(raw.convert(.f32), coords, .{}, .{ .linear = raw });
    return expected.sub(linear).abs().flatten().max(0).reshape(.{1});
}

fn drbaResizeDifference(probe: Tensor) Tensor {
    var error_max = Tensor.scalar(0, .f32);
    // Motifs distincts entre canaux, y compris au dernier pied. Une géométrie
    // impaire ou non divisible garde le filtre séparable ordinaire.
    for ([_][4]i64{
        .{ 12, 20, 3, 5 },  .{ 24, 40, 3, 5 }, .{ 4, 8, 1, 2 }, .{ 13, 20, 3, 5 },
        .{ 24, 40, 5, 10 }, .{ 24, 40, 6, 5 },
    }) |dims| {
        const h, const w, const oh, const ow = dims;
        const x = Tensor.arange(.{ .end = 3 * h * w }, .f32).remainderConst(53).subConstant(26).scale(0.00273).addConstant(0.12531)
            .mul(probe.flatten().slice(0, .single(0))).reshape(.{ 1, 3, h, w });
        for ([_]zml.DataType{ .f16, .f32 }) |dtype| {
            const source = x.convert(dtype);
            const expected = ops.resize(source, oh, ow, .{});
            const direct = ops.resize(source, oh, ow, .{ .drba_direct = true });
            error_max = error_max.maximum(expected.sub(direct).convert(.f32).abs().flatten().max(0));
        }
    }
    return error_max.reshape(.{1});
}

fn outputWarpDifference(probe: Tensor, small_h: i64, small_w: i64) Tensor {
    const h = 4 * small_h;
    const w = 4 * small_w;
    const dynamic = probe.flatten().slice(0, .single(0));
    const head = Tensor.arange(.{ .end = 5 * small_h * small_w }, .f32).scale(13).remainderConst(29)
        .subConstant(14).scale(0.21973).mul(dynamic).reshape(.{ 1, 5, small_h, small_w }).convert(.f16);
    const decoded = ops.resize(head.convert(.f32), h, w, .{}).convert(.f16);
    const rgb = Tensor.arange(.{ .end = 3 * h * w }, .f32).scale(19).remainderConst(191).scale(0.00549)
        .mul(dynamic).reshape(.{ 1, 3, h, w }).convert(.f16);
    const mask = decoded.slice(1, .{ .start = 4, .end = 5 }).sigmoid();
    var sides: [2]ops.OutputSide = undefined;
    var parts: [2]Tensor = undefined;
    for (&sides, 0..) |*side, k| {
        var channels: [2]Tensor = undefined;
        side.rgb = if (k == 0) rgb else rgb.scale(0.68).addConstant(0.07123);
        for (0..2) |axis| {
            const size = if (axis == 0) w else h;
            const index = Tensor.arange(.{ .end = size }, .f32);
            const unit = Tensor.scalar(2 / @as(f32, @floatFromInt(size - 1)), .f32);
            const ramp = index.mul(unit).subConstant(1);
            // Inclure les deux bords exacts, les positions hors de l'image et le cas non fini.
            const nan_value = std.math.nan(f32);
            const nan_tensor = Tensor.constantTensor(.scalar(.f32), std.mem.asBytes(&nan_value));
            const base = if (k == 1 and axis == 0) index.cmp(.EQ, .scalar(2, .f32)).select(nan_tensor, ramp) else ramp;
            side.origin[axis] = if (axis == 0) base.reshape(.{ 1, 1, 1, w }) else base.reshape(.{ 1, 1, h, 1 });
            side.unit[axis] = unit;
            const c: i64 = @intCast(k * 2 + axis);
            const flow = decoded.slice(1, .{ .start = c, .end = c + 1 }).scale(4).convert(.f32);
            channels[axis] = side.origin[axis].broadcastLeft(flow.shape()).add(flow.mul(unit));
        }
        const grid = Tensor.concatenate(&channels, 1).transpose(.{ 0, 2, 3, 1 });
        const warped = ops.gridSample(side.rgb.convert(.f32), grid, .{}).convert(.f16);
        const alpha = if (k == 0) mask else Tensor.scalar(1, .f16).sub(mask);
        parts[k] = warped.mul(alpha.broadcastLeft(warped.shape()));
    }
    const candidate = ops.drbaOutputWarp(head, sides) catch unreachable;
    return parts[0].add(parts[1]).bitCast(.u16).cmp(.NE, candidate.bitCast(.u16)).convert(.f32).flatten().max(0).reshape(.{1});
}

fn packedD2Samples(image: Tensor) Tensor {
    const one = image.flatten().slice(0, .single(0));
    const px = Tensor.stack(&.{ one.scale(0x1p-22), one.scale(1279), one.scale(1024.5), one.scale(48.5) }, .last, .sample);
    const py = Tensor.stack(&.{ one.scale(0), one.scale(767), one.scale(683.25), one.scale(28.125) }, .last, .sample);
    const valid = Tensor.arange(.{ .end = 4 }, .i32).withTags(.{.sample}).cmp(.LT, .scalar(3, .i32));
    const foot = @import("rife_warp_d2.zig").footprint(px, py, valid);
    return Tensor.concatenate(&.{ foot.indices.slice(.coord, .single(0)).convert(.f32), foot.indices.slice(.coord, .single(1)).convert(.f32), foot.wx, foot.wy, foot.valid.convert(.f32) }, 0);
}

fn gmflowSamples(image: Tensor) Tensor {
    const one = image.flatten().slice(0, .single(0));
    const zero = one.scale(0);
    const edge = Tensor.stack(&.{ one, zero, zero }, .last, .w).reshape(.{ 1, 1, 1, 3 });
    const xy = Tensor.stack(&.{ one.scale(-0x1p-24), zero }, .last, .coord).reshape(.{ 1, 1, 1, 2 });
    const sample = ops.gridSample(edge, xy, .{ .padding = .zeros, .integer_zero_margin = true });
    const ramp = Tensor.stack(&.{ zero, zero, one }, .last, .w).reshape(.{ 1, 1, 1, 3 });
    var bytes: [256]u8 = undefined;
    var storage = std.heap.FixedBufferAllocator.init(&bytes);
    const aligned = ops.resizeAligned(ramp, 1, 7, storage.allocator()) catch unreachable;
    return Tensor.concatenate(&.{ sample.flatten(), aligned.flatten().slice(0, .{ .start = 5, .end = 6 }) }, 0);
}

fn exercise(image: Tensor, coords: Tensor) Tensor {
    const border = ops.gridSample(image, coords, .{});
    const zeros = ops.gridSample(image, coords, .{ .padding = .zeros });
    const reflected = ops.gridSample(image, coords.transpose(.{ 0, 3, 1, 2 }), .{ .padding = .reflection, .grid_nchw = true });
    const nearest = ops.gridSample(image, coords, .{ .padding = .zeros, .mode = .nearest });
    const half_reflected = ops.gridSample(image.convert(.f16), coords, .{ .padding = .reflection, .align_corners = false }).convert(.f32);
    const ramps = Tensor.arange(.{ .end = 64 }, .f32).reshape(.{ 1, 1, 8, 8 });
    const resized = ops.resize(ramps, 2, 2, .{});
    const compact = ops.resize(ops.compactCenters(ramps, 4, false), 2, 2, .{});
    const frac = ops.resize(image, 1, 2, .{});
    const nearest_resize = ops.resize(image, 1, 2, .{ .mode = .nearest, .nearest = .round_prefer_floor });
    return Tensor.concatenate(&.{
        border.flatten(),                  zeros.flatten(),                  reflected.flatten(),               nearest.flatten(),
        resized.flatten(),                 compact.flatten(),                frac.flatten(),                    nearest_resize.flatten(),
        half_reflected.flatten(),          compactDifference(image, coords), phasedDifference(image),           linearWarpDifference(image, 3, 5),
        linearWarpDifference(image, 1, 5), drbaResizeDifference(image),      outputWarpDifference(image, 3, 5), outputWarpDifference(image, 1, 1),
        packedD2Samples(image),            gmflowSamples(image),
    }, 0);
}

test "bords ONNX, arrondi pair, réductions fractionnaires et warps par lots" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .cpu, .{});
    defer platform.deinit(allocator, io);
    const x = Tensor.init(.{ 2, 1, 1, 3 }, .f32);
    const g = Tensor.init(.{ 2, 1, 8, 2 }, .f32);
    const values = [6]f32{ 1, 2, 3, 11, 12, 13 };
    const row = [16]f32{ -2, 0, -1, 0, -0.5, 0, 0, 0, 0.5, 0, 1, 0, 2, 0, std.math.nan(f32), 0 };
    const positions = row ++ row;
    const module = try zml.Compiler.compile(allocator, io, platform, exercise, .{ x, g }, .{});
    defer module.deinit();
    var image = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&values));
    defer image.deinit();
    var grid = try zml.Buffer.fromBytes(io, platform, g.shape(), std.mem.asBytes(&positions));
    defer grid.deinit();
    var output = try module.eval(allocator, io, .{ image, grid });
    defer output.deinit();
    const expected = [_]f32{
        1,    1,    1.5,  2,    2.5,  3,    3,    0,    11,   11,   11.5,  12,      12.5,          13,   13, 0,
        0,    1,    1.5,  2,    2.5,  3,    0,    0,    0,    11,   11.5,  12,      12.5,          13,   0,  0,
        2,    1,    1.5,  2,    2.5,  3,    2,    0,    12,   11,   11.5,  12,      12.5,          13,   12, 0,
        0,    1,    1,    2,    3,    3,    0,    0,    0,    11,   11,    12,      13,            13,   0,  0,
        13.5, 17.5, 45.5, 49.5, 13.5, 17.5, 45.5, 49.5, 1.25, 2.75, 11.25, 12.75,   1,             3,    11, 13,
        2,    1,    1.25, 2,    2.75, 3,    2,    0,    12,   11,   11.25, 12,      12.75,         13,   12, 0,
        0,    0,    0,    0,    0,    0,    0,    0,    766,  683,  0,     0,       1278,          1024, 0,  0,
        1,    0.5,  0,    0,    1,    0.25, 0,    1,    1,    1,    0,     0x1p-24, 0x1.555554p-1,
    };
    try std.testing.expectEqualDeep(expected, try output.getValue(@TypeOf(expected), io));
}
