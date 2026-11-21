const std = @import("std");
const zml = @import("zml/core");
const motion = @import("rife_blocks.zig");
const Tensor = zml.Tensor;

fn constant(dims: anytype, values: anytype) Tensor {
    return Tensor.constantTensor(.init(dims, .f32), std.mem.asBytes(&values));
}

fn exercise(input: Tensor) Tensor {
    // Impulsions aussi bien dans les poids de réduction que dans la tête transposée.
    // Les biais distincts rendent observable l'ordre CRD des quatre phases.
    const delta = constant(.{ 1, 1, 3, 3 }, [9]f32{ 0, 0, 0, 0, 1, 0, 0, 0, 0 });
    const bias = constant(.{1}, [1]f32{0});
    const beta = constant(.{ 1, 1, 1, 1 }, [1]f32{1});
    var head_values: [20 * 16]f32 = @splat(0);
    var head_bias: [20]f32 = undefined;
    for (0..20) |c| {
        head_values[c * 16 + 5] = 1;
        head_bias[c] = @floatFromInt(c);
    }
    const head = constant(.{ 1, 20, 4, 4 }, head_values);
    const headb = constant(.{20}, head_bias);
    var scaled_args: [31]Tensor = undefined;
    var plain_args: [23]Tensor = undefined;
    scaled_args[0] = input;
    plain_args[0] = input;
    for (0..2) |i| {
        scaled_args[1 + i * 2] = delta;
        scaled_args[2 + i * 2] = bias;
        plain_args[1 + i * 2] = delta;
        plain_args[2 + i * 2] = bias;
    }
    for (0..8) |i| {
        scaled_args[5 + i * 3] = delta;
        scaled_args[6 + i * 3] = bias;
        scaled_args[7 + i * 3] = beta;
        plain_args[5 + i * 2] = delta;
        plain_args[6 + i * 2] = bias;
    }
    scaled_args[29] = head;
    scaled_args[30] = headb;
    plain_args[21] = head;
    plain_args[22] = headb;
    const scaled = motion.fromGraph("KirariIFBlock", &scaled_args, &.{ 255, 0 }, &.{}) catch unreachable;
    const phase = motion.fromGraph("KirariIFBlock", &scaled_args, &.{ 255, 1 }, &.{}) catch unreachable;
    const plain = motion.fromGraph("KirariIFBlock", &plain_args, &.{ 0, 0, 1 }, &.{}) catch unreachable;
    const up = constant(.{ 1, 1, 4, 4 }, [16]f32{ 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 });
    const encoder = motion.fromGraph("KirariEncoder", &.{ input, delta, bias, delta, bias, delta, bias, up, bias }, &.{0}, &.{ "conv0", "conv1", "conv2", "conv3" }) catch unreachable;
    return Tensor.concatenate(&.{ scaled.flatten(), phase.flatten(), plain.flatten(), encoder.flatten() }, 0);
}

test "IFBlock résiduel, couches simples, phases transposées et encodeur" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .cpu, .{});
    defer platform.deinit(allocator, io);
    const x = Tensor.init(.{ 1, 1, 4, 8 }, .f32);
    const module = try zml.Compiler.compile(allocator, io, platform, exercise, .{x}, .{});
    defer module.deinit();
    const ones: [32]f32 = @splat(1);
    var input = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&ones));
    defer input.deinit();
    var output = try module.eval(allocator, io, .{input});
    defer output.deinit();
    var expected: [3 * 5 * 4 * 8 + 4 * 8]f32 = undefined;
    for (0..5) |c| for (0..4) |y| for (0..8) |col| {
        const pos = (c * 4 + y) * 8 + col;
        const b: f32 = @floatFromInt(c * 4 + 2 * (y % 2) + col % 2);
        const nonzero = y < 2 and col % 4 < 2;
        expected[pos] = b + @as(f32, if (nonzero) 256 else 0);
        expected[160 + pos] = expected[pos];
        expected[320 + pos] = b + @as(f32, if (nonzero) 1 else 0);
    };
    for (0..4) |y| for (0..8) |col| {
        expected[480 + y * 8 + col] = if (y % 2 == 0 and col % 2 == 0) 1 else 0;
    };
    try std.testing.expectEqualDeep(expected, try output.getValue(@TypeOf(expected), io));
}

test "LeakyRelu par selection reproduit les deux branches FP16" {
    const model = struct {
        pub fn forward(x: Tensor) Tensor {
            return @import("rife_activation.zig").leaky(x, true);
        }
    };
    const io = std.testing.io;
    const allocator = std.testing.allocator;
    const platform = try zml.Platform.init(allocator, io, .cpu, .{});
    defer platform.deinit(allocator, io);
    const x = Tensor.init(.{4}, .f16);
    const module = try zml.Compiler.compile(allocator, io, platform, model.forward, .{x}, .{});
    defer module.deinit();
    const samples = [4]f16{ -4, -1, 0, 2 };
    var input = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&samples));
    defer input.deinit();
    var output = try module.eval(allocator, io, .{input});
    defer output.deinit();
    const expected = [4]f16{ -0.8, -0.2, 0, 2 };
    try std.testing.expectEqualDeep(expected, try output.getValue(@TypeOf(expected), io));
}
