//! Traçage de l'IR de mouvement à formes fixées, avec les poids d'origine du modèle.
const std = @import("std");
const zml = @import("zml/core");
const mlir = @import("mlir");
const func = @import("mlir/dialects/func");
const stablehlo = @import("mlir/dialects/stablehlo");
const conv = @import("motion_conv.zig");
const sampling = @import("motion_sampling.zig");
const blocks = @import("rife_blocks.zig");
const features = @import("rife_features.zig");
const Tensor = zml.Tensor;
const missing = std.math.maxInt(u32);

const Type = enum {
    f16,
    f32,
    i64,
    fn zmlType(self: Type) zml.DataType {
        return switch (self) {
            .f16 => .f16,
            .f32 => .f32,
            .i64 => .i64,
        };
    }
};
const Spec = struct {
    dtype: Type,
    dims: []const i64,
    offset: ?usize = null,
    size: usize = 0,
    fn shape(self: Spec) !zml.Shape {
        if (self.dims.len > 8) return error.InvalidRank;
        var count: u64 = 1;
        for (self.dims) |d| {
            if (d < 0 or d > 1 << 24) return error.InvalidDimension;
            count *= @intCast(d);
            if (count > 1 << 32) return error.InvalidDimension;
        }
        return .init(self.dims, self.dtype.zmlType());
    }
};
const Node = struct {
    op: []const u8,
    name: []const u8 = "",
    layers: []const []const u8 = &.{},
    ins: []const u32,
    outs: []const u32,
    i: []const i64 = &.{},
    f: []const f32 = &.{},
    offset: ?usize = null,
    size: usize = 0,
    compact: ?CompactWarp = null,
    upsample_phases: bool = false,
    drba_direct_resize: bool = false,
    drba_shared_projection: bool = false,
    drba_central_pixels: ?u32 = null,
    linear_source: ?u32 = null,
    interleaved_xy: bool = false,
};
const CompactWarp = struct {
    image: u32,
    grid: u32,
    factor: i64,
    first_channels: i64,
    round_warp: bool,
    round_planes: bool,
    grid_options: sampling.GridOptions,
    planes: []const u32,
};
const Program = struct { drba_output_pixel: bool = false, family: enum { motion, upscale, temporal, gmfss }, io: Type, inputs: []const u32, outputs: []const u32, tensors: []const Spec, nodes: []const Node };
const Op = enum { KirariGmflowMatch, KirariGmflowMlp, MatMul, Conv, ConvTranspose, GridSample, Resize, DepthToSpace, KirariEncoder, KirariIFBlock, KirariRifeNextEncoder, KirariDrbaOutputWarp, KirariRifeD2Centre, KirariRife425LiteD4, Add, Sub, Mul, Div, Clip, Abs, Neg, Relu, LeakyRelu, PRelu, Sigmoid, Tanh, Cast, Identity, Reshape, Unsqueeze, Squeeze, Expand, Concat, Transpose, FakeQuantE4M3 };

fn bytesAt(data: []const u8, offset: usize, len: usize) ![]const u8 {
    if (offset > data.len or len > data.len - offset) return error.TruncatedGraph;
    return data[offset..][0..len];
}
fn tensorAt(values: []const ?Tensor, id: u32) !Tensor {
    if (id >= values.len) return error.InvalidOperand;
    return values[id] orelse error.UnboundOperand;
}

// La moyenne half_pixel 2×2 des quatre échantillons de la montée ×4 coïncide
// en valeur réelle avec son centre. Seul l'arrondi FP16 intermédiaire des neuf
// attributs change. Les 15 premiers plans et le flot suivent la réduction ONNX.
fn rifeD2Centre(n: Node, values: []const ?Tensor, dst: zml.Shape) !Tensor {
    if (n.ins.len != 5 or n.i.len != 1 or (n.i[0] != 0 and n.i[0] != 1) or
        dst.rank() != 4 or dst.dtype() != .f16 or dst.dim(0) != 1 or dst.dim(1) != 28)
        return error.InvalidRifeD2;
    const height = dst.dim(2);
    const width = dst.dim(3);
    if (height < 4 or width < 4 or @mod(height, 4) != 0 or @mod(width, 4) != 0) return error.InvalidRifeD2;
    var parts: [5]Tensor = undefined;
    for (&parts, n.ins, [_]i64{ 6, 8, 1, 13, 4 }, 0..) |*part, id, channels, k| {
        part.* = try tensorAt(values, id);
        const h = if (k == 3) @divExact(height, 2) else 2 * height;
        const w = if (k == 3) @divExact(width, 2) else 2 * width;
        if (part.rank() != 4 or part.dtype() != .f16 or part.dim(0) != 1 or
            part.dim(1) != channels or part.dim(2) != h or part.dim(3) != w) return error.InvalidRifeD2;
    }
    const down: sampling.ResizeOptions = .{ .scale_h = 0.5, .scale_w = 0.5, .upsample_phases = n.i[0] != 0 };
    const up: sampling.ResizeOptions = .{ .scale_h = 2, .scale_w = 2, .upsample_phases = n.i[0] != 0 };
    const first = Tensor.concatenate(parts[0..3], 1);
    const attributes = parts[3].slice(1, .{ .start = 4, .end = 13 });
    return Tensor.concatenate(&.{
        sampling.resize(first, height, width, down),
        sampling.resize(attributes, height, width, up),
        sampling.resize(parts[4], height, width, down),
    }, 1);
}

fn direct(n: Node, values: []const ?Tensor, dst: zml.Shape, arena: std.mem.Allocator, gmflow_pair: bool) !Tensor {
    const op = std.meta.stringToEnum(Op, n.op) orelse return error.UnsupportedOperation;
    if (n.ins.len == 0) return error.InvalidOperand;
    const x = try tensorAt(values, n.ins[0]);
    const b: ?Tensor = if (n.ins.len > 1 and n.ins[1] != missing) try tensorAt(values, n.ins[1]) else null;
    const c: ?Tensor = if (n.ins.len > 2 and n.ins[2] != missing) try tensorAt(values, n.ins[2]) else null;
    const i = n.i;
    return switch (op) {
        .KirariGmflowMatch => blk: {
            if (!gmflow_pair or n.ins.len != 4) return error.InvalidGlobalMatcher;
            break :blk try @import("gmflow_global.zig").streamed(x, b orelse return error.InvalidGlobalMatcher, c orelse return error.InvalidGlobalMatcher, try tensorAt(values, n.ins[3]));
        },
        .KirariGmflowMlp => blk: {
            if (n.ins.len != 9) return error.InvalidGmflowMlp;
            var args: [9]Tensor = undefined;
            for (&args, n.ins) |*arg, id| arg.* = try tensorAt(values, id);
            break :blk try @import("gmfss_mlp.zig").run(args, i);
        },
        .MatMul => try @import("gmfss.zig").matmul(x, b orelse return error.MissingWeights, dst),
        .KirariRife425LiteD4 => blk: {
            if (n.ins.len != 5) return error.InvalidRifeD4;
            break :blk try sampling.rife425LiteD4(x, b orelse return error.MissingOperand, c orelse return error.MissingOperand, try tensorAt(values, n.ins[3]), try tensorAt(values, n.ins[4]));
        },
        .KirariRifeD2Centre => try rifeD2Centre(n, values, dst),
        .Conv, .ConvTranspose => blk: {
            if (i.len != 12) return error.InvalidConvolution;
            const kernel = b orelse return error.MissingWeights;
            const opts: conv.ConvOptions = .{
                .transposed = op == .ConvTranspose,
                .strides = i[0..2].*,
                .dilations = i[2..4].*,
                .pads = i[4..8].*,
                .output_padding = i[8..10].*,
                .group = i[10],
                .polyphase = i[11] != 0,
            };
            if (gmflow_pair) {
                if (@import("gmfss.zig").reduceChannels(x, kernel, c, opts)) |y| break :blk y;
            }
            break :blk conv.convolution(x, kernel, c, opts);
        },
        .KirariDrbaOutputWarp => blk: {
            if (n.ins.len != 11) return error.InvalidDrbaOutput;
            var sides: [2]sampling.OutputSide = undefined;
            for (&sides, 0..) |*side, k| {
                const args = n.ins[1 + 5 * k ..][0..5];
                side.* = .{
                    .rgb = try tensorAt(values, args[0]),
                    .origin = .{ try tensorAt(values, args[1]), try tensorAt(values, args[3]) },
                    .unit = .{ try tensorAt(values, args[2]), try tensorAt(values, args[4]) },
                };
            }
            break :blk try sampling.drbaOutputWarp(x, sides);
        },
        .GridSample => blk: {
            if (i.len != 4) return error.InvalidGrid;
            const raw = if (n.linear_source) |id| try tensorAt(values, id) else null;
            const grid = b orelse return error.MissingGrid;
            const central: ?Tensor = if (n.drba_central_pixels) |id| pixels: {
                const raw_rgb = try tensorAt(values, id);
                if (raw != null or n.interleaved_xy or !std.mem.eql(i64, i, &.{ 0, 1, 1, 0 }) or
                    x.dtype() != .f32 or raw_rgb.dtype() != .f16 or grid.dtype() != .f32 or
                    !std.mem.eql(i64, x.dims(), &.{ 1, 3, 768, 1280 }) or !std.mem.eql(i64, x.dims(), raw_rgb.dims()) or
                    !std.mem.eql(i64, grid.dims(), &.{ 1, 768, 1280, 2 })) return error.InvalidCentralPixels;
                break :pixels raw_rgb.withTags(.{ .n, .c, .h, .w });
            } else null;
            break :blk sampling.gridSampleFrom(x, grid, .{
                .mode = if (i[0] != 0) .nearest else .bilinear,
                .padding = if (i[1] != 0) .border else .zeros,
                .align_corners = i[2] != 0,
                .grid_nchw = i[3] != 0,
                .interleaved_xy = n.interleaved_xy,
                .integer_zero_margin = gmflow_pair,
            }, .{ .linear = raw, .central = central });
        },
        .Resize => blk: {
            if (i.len != 3 or n.f.len != 2) return error.InvalidResize;
            if (gmflow_pair and i[0] != 0 and i[1] == 2)
                break :blk try sampling.resizeAligned(x, dst.dim(2), dst.dim(3), arena);
            if (n.compact) |plan| {
                if (plan.factor < 4 or @mod(plan.factor, 2) != 0 or plan.planes.len < 2)
                    return error.InvalidResize;
                const parts = try arena.alloc(Tensor, plan.planes.len);
                for (parts, plan.planes) |*part, id| part.* = try tensorAt(values, id);
                const prepared = try sampling.reducedWarpPlanes(
                    try tensorAt(values, plan.image),
                    try tensorAt(values, plan.grid),
                    plan.grid_options,
                    plan.factor,
                    plan.first_channels,
                    parts,
                    plan.round_warp,
                );
                const rounded = if (plan.round_planes) prepared.convert(.f16) else prepared;
                break :blk sampling.resize(rounded.convert(x.dtype()), dst.dim(2), dst.dim(3), .{});
            }
            break :blk sampling.resize(x, dst.dim(2), dst.dim(3), .{
                .mode = if (i[0] != 0) .linear else .nearest,
                .coordinate = switch (i[1]) {
                    0 => .half_pixel,
                    1 => .pytorch_half_pixel,
                    2 => .align_corners,
                    3 => .asymmetric,
                    else => return error.InvalidResize,
                },
                .nearest = switch (i[2]) {
                    0 => .round_prefer_floor,
                    1 => .round_prefer_ceil,
                    2 => .floor,
                    3 => .ceil,
                    else => return error.InvalidResize,
                },
                .scale_h = n.f[0],
                .scale_w = n.f[1],
                .upsample_phases = n.upsample_phases,
                .drba_direct = n.drba_direct_resize,
            });
        },
        .DepthToSpace => if (i.len == 2) conv.depthToSpace(x, i[0], i[1] != 0) else error.InvalidBlock,
        .KirariEncoder, .KirariIFBlock, .KirariRifeNextEncoder, .Concat => blk: {
            const args = try arena.alloc(Tensor, n.ins.len);
            for (args, n.ins) |*arg, id| arg.* = try tensorAt(values, id);
            if (op == .KirariRifeNextEncoder) break :blk try features.following(args, n.i, n.layers);
            if (op != .Concat) break :blk try blocks.fromGraph(n.op, args, n.i, n.layers);
            if (i.len != 1) return error.InvalidAxis;
            break :blk Tensor.concatenate(args, i[0]);
        },
        .Add, .Sub, .Mul, .Div, .PRelu => blk: {
            const lhs = x.broadcastLeft(dst);
            const rhs = (b orelse return error.MissingOperand).broadcastLeft(dst);
            break :blk switch (op) {
                .Add => if (n.drba_shared_projection) lhs.add(rhs).optimizationBarrier() else lhs.add(rhs),
                .Sub => lhs.sub(rhs),
                .Mul => lhs.mul(rhs),
                .Div => lhs.div(rhs),
                .PRelu => lhs.relu().add(lhs.minimum(.scalar(0, lhs.dtype())).mul(rhs)),
                else => unreachable,
            };
        },
        .Clip => blk: {
            var y = x;
            if (b) |lower| y = y.maximum(lower.broadcastLeft(dst));
            if (c) |upper| y = y.minimum(upper.broadcastLeft(dst));
            break :blk y;
        },
        .LeakyRelu => if (n.f.len == 1) x.relu().add(x.minimum(.scalar(0, x.dtype())).scale(n.f[0])) else error.InvalidSlope,
        .Relu => x.relu(),
        .Sigmoid => x.sigmoid(),
        .Tanh => x.tanh(),
        .Neg => x.negate(),
        .Abs => x.abs(),
        .Cast => blk: {
            const converted = x.convert(dst.dtype());
            // L’ONNX de GMFlow arrondit explicitement avant la reprise en FP32.
            // XLA peut sinon supprimer les deux conversions pour garder la précision excédentaire.
            if (gmflow_pair and x.dtype() == .f32 and dst.dtype() == .f16)
                break :blk converted.optimizationBarrier();
            break :blk converted;
        },
        .Expand => x.broadcastLeft(dst),
        .Identity, .Reshape, .Squeeze, .Unsqueeze => x.reshape(dst),
        .Transpose => x.transpose(i),
        .FakeQuantE4M3 => blk: {
            const scale = (b orelse return error.MissingScale).convert(.f32).reshape(.{});
            const y = x.convert(.f32).div(scale).clamp(.scalar(-448, .f32), .scalar(448, .f32));
            break :blk y.convert(.f8e4m3fn).convert(.f32).mul(scale).convert(x.dtype());
        },
    };
}

// Les petits fragments de l'abaisseur commun (notamment Split et Slice) sont importés dans
// le bloc actif. Le compilateur voit un unique graphe sans frontière d'appel.
fn inlineFragment(compiler: *zml.Compiler, program: []const u8, n: Node, values: []?Tensor) !void {
    const module = try mlir.Module.parse(compiler.mlir_ctx, program);
    defer module.deinit();
    const fn_op = module.body().firstOperation() orelse return error.EmptyFunction;
    if (!std.mem.eql(u8, fn_op.name(), "func.func") or fn_op.numRegions() != 1) return error.InvalidFunction;
    const source = fn_op.region(0).firstBlock() orelse return error.EmptyFunction;
    const ret = source.terminator() orelse return error.InvalidFunction;
    if (!std.mem.eql(u8, ret.name(), "func.return") or ret.numOperands() != n.outs.len) return error.InvalidFunction;
    var index: usize = 0;
    for (n.ins) |id| {
        if (id == missing) continue;
        if (index >= source.numArguments()) return error.InvalidOperand;
        source.argument(index).replaceAllUsesWith((try tensorAt(values, id)).value());
        index += 1;
    }
    if (index != source.numArguments()) return error.InvalidFunction;
    while (source.firstOperation()) |op| {
        if (op == ret) break;
        op.removeFromParent();
        _ = op.appendTo(compiler.currentScope().block);
    }
    for (n.outs, 0..) |id, k| values[id] = Tensor.fromMlirValue(ret.operand(k));
}

fn gmflowPrecision(config: *const mlir.Attribute, block: *mlir.Block) void {
    var op = block.firstOperation();
    while (op) |current| : (op = @import("trace.zig").SourceLocations.nextOperation(current)) {
        const name = current.name();
        if ((std.mem.eql(u8, name, "stablehlo.dot_general") or std.mem.eql(u8, name, "stablehlo.convolution")) and
            Tensor.fromMlirValue(current.operand(0)).dtype() == .f32 and Tensor.fromMlirValue(current.operand(1)).dtype() == .f32)
            current.setAttributeByName("precision_config", config);
        // Les régions StableHLO (while, reduce, etc.) contiennent un seul bloc.
        for (0..current.numRegions()) |i| {
            if (current.region(i).firstBlock()) |nested| gmflowPrecision(config, nested);
        }
    }
}

const Motion = struct {
    program: Program,
    blob: []const u8,
    alloc: std.mem.Allocator,

    pub fn trace(self: Motion, compiler: *zml.Compiler, scope: *zml.Compiler.Scope) !void {
        const p = self.program;
        const blob = self.blob;
        const alloc = self.alloc;
        const block = scope.block;
        const loc = compiler.unknown_location;
        const values = try alloc.alloc(?Tensor, p.tensors.len);
        @memset(values, null);
        for (p.inputs, 0..) |id, k| {
            if (id >= values.len or values[id] != null) return error.InvalidInput;
            const shape = try p.tensors[id].shape();
            const x = Tensor.init(shape.dims(), p.io.zmlType());
            _ = block.addArgument(mlir.Type.rankedTensor(shape.dims(), compiler.mlirType(x.dtype())), loc);
            scope.registerTensorAsBlockArgument(x.id, k);
            values[id] = x.convert(shape.dtype());
        }
        for (p.tensors, values) |t, *v| {
            if (t.offset) |off| {
                const sh = try t.shape();
                if (v.* != null or sh.byteSize() != t.size) return error.InvalidConstant;
                v.* = Tensor.constantTensor(sh, try bytesAt(blob, off, t.size));
            }
        }
        var locations: @import("trace.zig").SourceLocations = .{};
        const packed_d2: ?u32 = if (@import("rife_warp_options").enabled) @import("rife_warp_d2.zig").select(p) else null;
        locations.advance(compiler);
        for (p.nodes) |n| {
            if (n.outs.len == 0) return error.MissingOutput;
            for (n.outs) |id| if (id >= values.len or values[id] != null) return error.InvalidOutput;
            compiler.pushLocation(@src(), if (n.name.len == 0) n.op else n.name);
            defer compiler.popLocation();
            if (n.offset) |off| {
                try inlineFragment(compiler, try bytesAt(blob, off, n.size), n, values);
            } else {
                if (n.outs.len != 1) return error.InvalidOutput;
                const id = n.outs[0];
                values[id] = if (packed_d2 == id)
                    try sampling.gridSampleD2(try tensorAt(values, n.ins[0]), try tensorAt(values, n.ins[1]))
                else
                    direct(n, values, try p.tensors[id].shape(), alloc, p.family == .gmfss and p.outputs.len == 14) catch |err| {
                        std.log.err("{s} ({s}) : {t}", .{ n.name, n.op, err });
                        return err;
                    };
            }
            for (n.outs) |id| {
                const v = values[id] orelse return error.MissingOutput;
                if (v.dtype() != p.tensors[id].dtype.zmlType() or !std.mem.eql(i64, v.dims(), p.tensors[id].dims)) return error.OutputShapeMismatch;
            }
            locations.advance(compiler);
        }
        const results = try alloc.alloc(*const mlir.Value, p.outputs.len);
        for (results, p.outputs) |*r, id| {
            const output = (try tensorAt(values, id)).convert(p.io.zmlType());
            if (p.drba_output_pixel and (p.family != .motion or results.len != 1)) return error.InvalidDrbaOutput;
            r.* = if (p.drba_output_pixel) (try @import("drba_pixel.zig").rgb(output)).value() else output.value();
        }
        _ = func.returns(compiler.mlir_ctx, results, loc).appendTo(block);
        if (p.family == .gmfss and p.outputs.len == 14) {
            // Une accumulation FP32 seule laisse le choix de la précision des
            // multiplicandes au backend. Le flot amplifie cet écart près du softmax.
            const highest = stablehlo.precisionAttr(compiler.mlir_ctx, .HIGHEST);
            const config = mlir.Attribute.array(compiler.mlir_ctx, &.{ highest, highest });
            gmflowPrecision(config, block);
        }
    }
};

pub fn emit(init: std.process.Init, input_path: []const u8, output_path: []const u8, rife_cache: bool) !void {
    const alloc = init.arena.allocator();
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, input_path, alloc, .limited(512 * 1024 * 1024));
    if (bytes.len < 16 or !std.mem.eql(u8, bytes[0..8], "KZMLIR01")) return error.InvalidFormat;
    const json_len = std.math.cast(usize, std.mem.readInt(u64, bytes[8..16], .little)) orelse return error.InvalidFormat;
    const json = try bytesAt(bytes, 16, json_len);
    const blob = bytes[16 + json.len ..];
    const parsed = try std.json.parseFromSlice(Program, alloc, json, .{});
    const p = parsed.value;
    const name: []const u8 = switch (p.family) {
        .motion => "kirari_motion",
        .upscale => "kirari_upscale",
        .temporal => "kirari_temporal",
        .gmfss => "kirari_gmfss",
    };
    const network = Motion{ .program = p, .blob = blob, .alloc = alloc };
    try @import("trace.zig").write(init, output_path, name, network);
    if (!rife_cache or !features.recognized(p)) return;
    for ([_]features.Phase{ .froid, .instants, .suite }) |phase| {
        var derived = network;
        derived.program = try features.specialize(p, phase, alloc);
        const path = try std.fmt.allocPrint(alloc, "{s}.rife-{s}.mlir", .{ output_path, @tagName(phase) });
        try @import("trace.zig").write(init, path, name, derived);
    }
}
