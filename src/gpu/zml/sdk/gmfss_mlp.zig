//! Premier MLP au quart de GMFlow s05. Quatre bandes couvrent tous les jetons
//! en conservant les deux contractions, la GELU, la LayerNorm et le résiduel FP32.
const std = @import("std");
const zml = @import("zml/core");
const Tensor = zml.Tensor;

const rows = 2160;
const width = 128;
const prefix = "/flownet/transformer/layers.0/";

fn scale(c: f32) Tensor {
    return .scalar(c, .f32);
}

fn project(x: Tensor, weight: Tensor, comptime name: []const u8) Tensor {
    const compiler = zml.Compiler.current();
    compiler.pushLocation(@src(), prefix ++ name);
    defer compiler.popLocation();
    const kernel = weight.broadcast(.init(.{ 4, weight.dim(0), weight.dim(1) }, .f32), &.{ 1, 2 });
    return x.dotGeneral(kernel, &.{.{ 2, 1 }}, &.{.{ 0, 0 }});
}

fn erf(x: Tensor) Tensor {
    const compiler = zml.Compiler.current();
    compiler.pushLocation(@src(), prefix ++ "cross_attn_ffn/mlp/mlp.1_1/Erf");
    defer compiler.popLocation();
    // Abramowitz–Stegun 7.1.26 dans le même ordre de calcul que le frontal stock.
    const ab = x.abs();
    const t = scale(1).div(scale(1).add(scale(0.3275911).mul(ab)));
    var poly = scale(1.061405429);
    for ([_]f32{ -1.453152027, 1.421413741, -0.284496736, 0.254829592 }) |c| poly = poly.mul(t).add(scale(c));
    return x.sign().mul(scale(1).sub(poly.mul(t).mul(ab.mul(ab).negate().exp())));
}

fn normalize(x: Tensor, weight: Tensor, bias: Tensor, epsilon: f32) Tensor {
    const compiler = zml.Compiler.current();
    compiler.pushLocation(@src(), prefix ++ "cross_attn_ffn/norm2_1/LayerNormalization");
    defer compiler.popLocation();
    const avg = x.sum(2).div(scale(width));
    const centered = x.sub(avg.broadcastLeft(x.shape()));
    const variance = centered.mul(centered).sum(2).div(scale(width));
    const inv = variance.add(scale(epsilon)).rsqrt();
    return centered.mul(inv.broadcastLeft(x.shape())).mul(weight.broadcast(x.shape(), &.{2})).add(bias.broadcast(x.shape(), &.{2}));
}

const Loop = struct {
    args: [9]Tensor,
    epsilon: f32,
    pub const State = struct { first: Tensor, output: Tensor };

    pub fn cond(_: Loop, state: State) Tensor {
        return state.first.cmp(.LT, .scalar(8640, .i32));
    }
    pub fn body(self: Loop, state: State) State {
        const a = self.args;
        const x = a[0].slice(1, .dyn(state.first, rows));
        const msg = a[1].slice(1, .dyn(state.first, rows));
        const joined = Tensor.concatenate(&.{ x, msg }, 2);
        const up = project(joined, a[2], "cross_attn_ffn/mlp/mlp.0_1/MatMul");
        const gate = erf(up.div(a[3])).add(a[4]);
        const hidden = up.mul(gate).mul(a[5]);
        const down = project(hidden, a[6], "cross_attn_ffn/mlp/mlp.2_1/MatMul");
        const result = x.add(normalize(down, a[7], a[8], self.epsilon));
        return .{ .first = state.first.addConstant(rows), .output = state.output.dynamicUpdateSlice1d(result, 1, state.first) };
    }
};

pub fn run(inputs: [9]Tensor, ints: []const i64) !Tensor {
    if (ints.len != 1 or ints[0] != @as(u32, @bitCast(@as(f32, 1e-5)))) return error.InvalidGmflowMlp;
    const shapes = [_][]const i64{ &.{ 4, 8640, width }, &.{ 4, 8640, width }, &.{ 256, 1024 }, &.{}, &.{}, &.{}, &.{ 1024, width }, &.{width}, &.{width} };
    var args: [9]Tensor = undefined;
    for (inputs, shapes, &args) |value, dims, *out| {
        if (!std.mem.eql(i64, value.dims(), dims) or value.dtype() != .f32) return error.InvalidGmflowMlp;
        out.* = value;
    }
    const ctx = Loop{ .args = args, .epsilon = @bitCast(@as(u32, @intCast(ints[0]))) };
    return zml.ops.@"while"(Loop, ctx, .{ .first = .scalar(0, .i32), .output = Tensor.zeroes(args[0].shape()) }).output;
}
