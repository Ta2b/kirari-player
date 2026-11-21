//! Porte 1×1 des modèles Balanced et Performance en 1080p, commune au témoin StableHLO et à la région de fusion XLA.
const std = @import("std");
const zml = @import("zml/core");
const mlir = @import("mlir");
const Tensor = zml.Tensor;

pub const Options = struct {
    family: enum { Balanced, Performance },
    region: bool,
    projection_region: bool = false,

    pub fn fromArgument(arg: []const u8) ?Options {
        const modes = std.StaticStringMap(Options).initComptime(.{
            .{ "--span-balanced-triton-gates", Options{ .family = .Balanced, .region = true } },
            .{ "--span-balanced-portable-gates", Options{ .family = .Balanced, .region = false } },
            .{ "--span-performance-triton-gates", Options{ .family = .Performance, .region = true } },
            .{ "--span-performance-portable-gates", Options{ .family = .Performance, .region = false } },
        });
        return modes.get(arg);
    }

    fn width(self: Options) i64 {
        return if (self.family == .Balanced) 64 else 48;
    }

    fn count(self: Options) usize {
        return if (self.family == .Balanced) 8 else 5;
    }

    pub fn supports(self: Options, blocks: u32, channels: i64, height: i64, width_px: i64) bool {
        return blocks == self.count() and channels == self.width() and height == 1080 and width_px == 1920;
    }

    pub fn moduleName(self: Options) []const u8 {
        if (self.projection_region) return "kirari_span_balanced_gate8_projection";
        return switch (self.family) {
            .Balanced => if (self.region) "kirari_span_balanced_gate8" else "kirari_span_balanced_gate8_portable",
            .Performance => if (self.region) "kirari_span_performance_gate5" else "kirari_span_performance_gate5_portable",
        };
    }
};

fn gate(arguments: [4]Tensor) Tensor {
    const residual, const weight, const bias, const skip = arguments;
    const product = residual.dotGeneral(weight, &.{.{ 1, 1 }}, &.{});
    const attention = product.add(bias.broadcast(product.shape(), &.{1})).tanh();
    return skip.add(residual).mul(attention);
}

fn fusion(arguments: anytype, comptime body: fn (@TypeOf(arguments)) Tensor, loc: *const mlir.Location) !Tensor {
    const c = zml.Compiler.current();
    const ctx = c.mlir_ctx;
    const arity = arguments.len;
    // Le SDK f430c3530 enregistre func, stablehlo et sdy. MLIR peut transporter la
    // région MHLO générique ; sa sémantique est ensuite vérifiée par l'importeur XLA.
    ctx.setAllowUnregisteredDialects(true);
    const kind = try mlir.Attribute.parse(ctx, "#mhlo<fusion_kind kCustom>");
    var operands: [arity]*const mlir.Value = undefined;
    var types: [arity]*const mlir.Type = undefined;
    for (arguments, &operands, &types) |arg, *val, *ty| {
        val.* = arg.value();
        ty.* = val.*.type_();
    }
    const block = mlir.Block.init(&types, &@as([arity]*const mlir.Location, @splat(loc)));
    {
        const scope = c.pushBlock(block);
        defer scope.pop();
        var args: [arity]Tensor = undefined;
        for (&args, 0..) |*arg, k| arg.* = Tensor.fromMlirValue(block.argument(k));
        const result = body(args);
        _ = (try mlir.Operation.try_make(ctx, "mhlo.return", .{
            .operands = .{ .flat = &.{result.value()} },
            .location = loc,
        })).appendTo(block);
    }
    const op = (try mlir.Operation.try_make(ctx, "mhlo.fusion", .{
        .operands = .{ .flat = &operands },
        .results = .{ .flat = &.{types[0]} },
        .blocks = &.{block},
        .attributes = &.{.named(ctx, "fusion_kind", kind)},
        .location = loc,
    })).appendTo(c.currentScope().block);
    return Tensor.fromMlirValue(op.result(0));
}

pub fn apply(options: Options, index: usize, skip: Tensor, residual: Tensor, weight: Tensor, bias: Tensor) !Tensor {
    if (index < 1 or index > options.count()) return error.InvalidGate;
    const channels = options.width();
    const expected = [_][4]i64{ .{ 1, 540, 960, channels }, .{ 1, 540, 960, channels }, .{ channels, 1, 1, channels } };
    for ([_]Tensor{ skip, residual, weight }, expected) |x, dims|
        if (x.dtype() != .f16 or !std.mem.eql(i64, x.dims(), &dims)) return error.InvalidGateShape;
    if (bias.dtype() != .f16 or !std.mem.eql(i64, bias.dims(), &.{channels})) return error.InvalidGateBias;
    const flat = residual.reshape(.{ 518400, channels });
    const arguments = [4]Tensor{ flat, weight.reshape(.{ channels, channels }), bias, skip.reshape(flat.shape()) };
    const x = if (options.region) region: {
        const c = zml.Compiler.current();
        const loc = c.unknown_location.namedFmt(c.mlir_ctx, "Kirari_SPAN_{s}_porte_{d}", .{ @tagName(options.family), index });
        break :region try fusion(arguments, gate, loc);
    } else gate(arguments);
    return x.reshape(residual.shape());
}

fn projection(arguments: [6]Tensor) Tensor {
    const channels = Tensor.concatenate(arguments[0..4], 1);
    const product = channels.dotGeneral(arguments[4], &.{.{ 1, 1 }}, &.{});
    return product.add(arguments[5].broadcast(product.shape(), &.{1}));
}

pub fn project(views: [4]Tensor, filter: Tensor, bias: Tensor) !Tensor {
    var args: [6]Tensor = undefined;
    for (views, args[0..4]) |view, *flat| {
        if (view.dtype() != .f16 or !std.mem.eql(i64, view.dims(), &.{ 1, 540, 960, 64 })) return error.InvalidProjectionShape;
        flat.* = view.reshape(.{ 518400, 64 });
    }
    if (filter.dtype() != .f16 or !std.mem.eql(i64, filter.dims(), &.{ 64, 1, 1, 256 }) or
        bias.dtype() != .f16 or !std.mem.eql(i64, bias.dims(), &.{64})) return error.InvalidProjectionWeight;
    args[4] = filter.reshape(.{ 64, 256 });
    args[5] = bias;
    const c = zml.Compiler.current();
    const loc = c.unknown_location.named(c.mlir_ctx, "Kirari_SPAN_Balanced_projection");
    return (try fusion(args, projection, loc)).reshape(views[0].shape());
}
