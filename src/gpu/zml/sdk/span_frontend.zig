//! SPAN-F3 en zml.Tensor. Le lecteur valide l'architecture ONNX puis passe les poids FP16 du modèle.
const std = @import("std");
const zml = @import("zml/core");
const mlir = @import("mlir");
const func = @import("mlir/dialects/func");
const Tensor = zml.Tensor;
const convolution = @import("span_convolution.zig");
const span_gate = @import("span_gate.zig");

const Weights = struct {
    body: []const u8,
    blocks: u32,
    channels: i64,
    gate_options: ?span_gate.Options = null,
    attention_dot: bool = false,
    projection_dot: bool = false,
    projection_strips: bool = false,
    projection_pixel_strips: bool = false,
    projection_gate_strips: bool = false,
    locations: @import("trace.zig").SourceLocations = .{},

    fn init(bytes: []const u8) !Weights {
        if (bytes.len < 40 or !std.mem.eql(u8, bytes[0..8], "AJPSPAN8")) return error.InvalidWeights;
        const blocks = std.mem.readInt(u32, bytes[12..16], .little);
        const channels = std.mem.readInt(u32, bytes[20..24], .little);
        if (std.mem.readInt(u32, bytes[8..12], .little) != 2 or
            std.mem.readInt(u32, bytes[16..20], .little) != 64 or
            !((blocks == 8 and channels == 64) or (blocks == 5 and channels == 48)) or
            std.mem.readInt(u32, bytes[36..40], .little) != 2 or
            std.mem.readInt(u64, bytes[24..32], .little) != bytes.len - 40 or
            std.mem.readInt(u32, bytes[32..36], .little) != std.hash.Crc32.hash(bytes[40..])) return error.InvalidWeights;
        var result: Weights = .{ .body = bytes[40..], .channels = channels, .blocks = blocks };
        _ = try result.take((1 + 3 * blocks) * 4);
        return result;
    }

    fn take(self: *Weights, count: usize) ![]const u8 {
        if (count > self.body.len) return error.TruncatedWeights;
        const result = self.body[0..count];
        self.body = self.body[count..];
        return result;
    }

    // Poids du paquet en [sortie, y, x, groupe, canal]. Les canaux de bourrage sont retirés
    // des constantes à la préparation. Les activations restent en NHWC dans le corps du modèle.
    fn readKernel(self: *Weights, inputs: i64, outputs: i64, stored_outputs: i64, side: i64, groups: i64, stored_inputs_per_group: i64) !convolution.Kernel {
        const weight_shape: zml.Shape = .init(.{ stored_outputs, side, side, groups, stored_inputs_per_group }, .f16);
        const constant = Tensor.constantTensor(weight_shape, try self.take(weight_shape.byteSize()));
        const filter = constant.slice(0, .{ .end = outputs })
            .slice(4, .{ .end = @divExact(inputs, groups) })
            .reshape(.{ outputs, side, side, inputs });
        const bias_shape: zml.Shape = .init(.{stored_outputs}, .f32);
        const bias = Tensor.constantTensor(bias_shape, try self.take(bias_shape.byteSize())).slice(0, .{ .end = outputs }).convert(.f16);
        return .{ .filter = filter, .bias = bias };
    }

    fn conv(self: *Weights, name: []const u8, input: Tensor, outputs: i64, stored_outputs: i64, side: i64, groups: i64, stored_inputs_per_group: i64) !Tensor {
        const compiler = zml.Compiler.current();
        self.locations.advance(compiler);
        compiler.pushLocation(@src(), name);
        defer compiler.popLocation();
        defer self.locations.advance(compiler);
        const inputs = input.dim(3);
        const kernel = try self.readKernel(inputs, outputs, stored_outputs, side, groups, stored_inputs_per_group);
        // Garder H et W comme axes libres du produit. Le compilateur peut fusionner
        // le biais, tanh et la multiplication résiduelle dans son épilogue.
        // conv_cat utilise les quatre tranches consécutives du filtre sur l'axe contracté.
        // Un seul produit conserve la réduction sur 4*C ; XLA choisit la fusion du concat.
        if ((self.attention_dot and std.mem.eql(u8, name, "att_conv")) or
            (self.projection_dot and std.mem.eql(u8, name, "conv_cat")))
        {
            std.debug.assert(side == 1);
            const projection = input.dotGeneral(kernel.filter.reshape(.{ outputs, inputs }), &.{.{ 3, 1 }}, &.{});
            return projection.add(kernel.bias.broadcast(projection.shape(), &.{3}));
        }
        return kernel.apply(input);
    }

    fn project(self: *Weights, views: [4]Tensor, gate: ?convolution.Gate) !Tensor {
        const compiler = zml.Compiler.current();
        const ch = self.channels;
        if (self.projection_strips and convolution.supportsProjectionStrips(views[0].dim(1), views[0].dim(2), ch)) {
            self.locations.advance(compiler);
            compiler.pushLocation(@src(), "conv_cat/bandes_30");
            defer compiler.popLocation();
            defer self.locations.advance(compiler);
            const kernel = try self.readKernel(4 * ch, ch, 64, 1, 4, 64);
            return convolution.projectStrips(views, kernel, gate);
        }
        const joined = blk: {
            compiler.pushLocation(@src(), "conv_cat/Concat");
            defer compiler.popLocation();
            defer self.locations.advance(compiler);
            break :blk Tensor.concatenate(&views, 3);
        };
        return self.conv("conv_cat", joined, ch, 64, 1, 4, 64);
    }

    fn forward(self: *Weights, image: Tensor) !Tensor {
        const compiler = zml.Compiler.current();
        const h = @divExact(image.dim(2), 2);
        const w = @divExact(image.dim(3), 2);
        const ch = self.channels;
        const unshuffle = blk: {
            compiler.pushLocation(@src(), "SpaceToDepth");
            defer compiler.popLocation();
            defer self.locations.advance(compiler);
            break :blk image.reshape(.{ 1, 3, h, 2, w, 2 }).transpose(.{ 0, 2, 4, 1, 3, 5 }).reshape(.{ 1, h, w, 12 });
        };
        const stem = try self.conv("conv_1.1.eval_conv", unshuffle, ch, 64, 3, 1, 16);
        const defer_gate = self.projection_gate_strips and ch == 48 and convolution.supportsProjectionStrips(h, w, ch);
        var tail: ?convolution.Gate = null;
        var x = stem;
        var first_block = stem;
        var last_relu = stem;
        for (0..self.blocks) |i| {
            compiler.pushLocationFmt(@src(), "block_{d}", .{i + 1});
            defer compiler.popLocation();
            defer self.locations.advance(compiler);
            last_relu = (try self.conv("c1_r.eval_conv", x, ch, 64, 3, 1, 64)).relu();
            const second = (try self.conv("c2_r.eval_conv", last_relu, ch, 64, 3, 1, 64)).relu();
            const residual = try self.conv("c3_r.eval_conv", second, ch, 64, 3, 1, 64);
            if (self.gate_options) |gate_options| {
                const kernel = try self.readKernel(ch, ch, 64, 1, 1, 64);
                x = try span_gate.apply(gate_options, i + 1, x, residual, kernel.filter, kernel.bias);
                if (i == 0) first_block = x;
                continue;
            }
            const logits = try self.conv("att_conv", residual, ch, 64, 1, 1, 64);
            if (defer_gate and i + 1 == self.blocks) {
                tail = .{ .residual = residual, .logits = logits };
            } else {
                x = x.add(residual).mul(logits.tanh());
            }
            if (i == 0) first_block = x;
        }
        const views = [4]Tensor{ stem, x, first_block, last_relu };
        if (self.projection_pixel_strips and convolution.supportsProjectionStrips(h, w, ch)) {
            compiler.pushLocation(@src(), "conv_cat/to_pixel/bandes_30");
            defer compiler.popLocation();
            const projection = try self.readKernel(4 * ch, ch, 64, 1, 4, 64);
            const pixel = try self.readKernel(ch, 48, 48, 3, 1, 64);
            if (self.body.len != 0) return error.TrailingWeights;
            return convolution.projectPixelStrips(views, projection, pixel);
        }
        const merged = try self.selectProjection(views, tail);
        const pixel = try self.conv("to_pixel.eval_conv", merged, 48, 48, 3, 1, 64);
        if (self.body.len != 0) return error.TrailingWeights;
        compiler.pushLocation(@src(), "DepthToSpace");
        defer compiler.popLocation();
        defer self.locations.advance(compiler);
        return convolution.arrangePixels(pixel);
    }
    fn selectProjection(self: *Weights, views: [4]Tensor, pending_gate: ?convolution.Gate) !Tensor {
        if (self.gate_options) |options| if (options.projection_region) {
            const kernel = try self.readKernel(256, 64, 64, 1, 4, 64);
            return span_gate.project(views, kernel.filter, kernel.bias);
        };
        return self.project(views, pending_gate);
    }
};

const Span = struct {
    weights: *Weights,
    height: i64,
    width: i64,

    pub fn trace(self: Span, compiler: *zml.Compiler, scope: *zml.Compiler.Scope) !void {
        const image: Tensor = .init(.{ 1, 3, self.height, self.width }, .f16);
        const loc = compiler.unknown_location;
        _ = scope.block.addArgument(mlir.Type.rankedTensor(image.dims(), compiler.mlirType(.f16)), loc);
        scope.registerTensorAsBlockArgument(image.id, 0);
        const output = try self.weights.forward(image);
        _ = func.return_(compiler.mlir_ctx, output.value(), loc).appendTo(scope.block);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len == 3) return @import("motion_frontend.zig").emit(init, args[1], args[2], false);
    if (args.len == 4 and std.mem.eql(u8, args[3], "--rife-cache"))
        return @import("motion_frontend.zig").emit(init, args[1], args[2], true);
    if (args.len < 5) return error.ExpectedWeightsHeightWidthOutput;
    const h = try std.fmt.parseInt(i64, args[2], 10);
    const w = try std.fmt.parseInt(i64, args[3], 10);
    if (h <= 0 or w <= 0 or h > 16384 or w > 16384 or @mod(h, 2) != 0 or @mod(w, 2) != 0) return error.InvalidDimensions;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, args[1], allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(bytes);
    var weights = try Weights.init(bytes);
    var projection_region = false;
    for (args[5..]) |option| {
        if (span_gate.Options.fromArgument(option)) |gate_options| {
            if (weights.gate_options != null) return error.ConflictingGateOptions;
            weights.gate_options = gate_options;
        } else if (std.mem.eql(u8, option, "--span-balanced-triton-projection")) {
            projection_region = true;
        } else if (std.mem.eql(u8, option, "--attention-dot")) {
            weights.attention_dot = true;
        } else if (std.mem.eql(u8, option, "--projection-dot")) {
            weights.projection_dot = true;
        } else if (std.mem.eql(u8, option, "--projection-strips")) {
            weights.projection_strips = true;
        } else if (std.mem.eql(u8, option, "--projection-pixel-strips")) {
            weights.projection_pixel_strips = true;
            weights.projection_strips = true;
        } else if (std.mem.eql(u8, option, "--projection-gate-strips")) {
            weights.projection_gate_strips = true;
            weights.projection_strips = true;
        } else return error.UnknownSpanOption;
    }
    if (weights.projection_pixel_strips and weights.projection_gate_strips) return error.ConflictingProjectionOptions;
    if (weights.projection_dot and weights.projection_strips) return error.ConflictingProjectionOptions;

    if (projection_region) {
        if (weights.projection_strips or weights.projection_dot) return error.ConflictingProjectionOptions;
        const options = if (weights.gate_options) |*value| value else return error.IncompatibleSpanGate;
        if (options.family != .Balanced or !options.region) return error.IncompatibleSpanGate;
        options.projection_region = true;
    }

    const name: []const u8 = if (weights.gate_options) |gate_options| gate_name: {
        if (weights.attention_dot or weights.projection_gate_strips or !gate_options.supports(weights.blocks, weights.channels, h, w))
            return error.IncompatibleSpanGate;
        break :gate_name gate_options.moduleName();
    } else "kirari_span";
    try @import("trace.zig").write(init, args[4], name, Span{ .weights = &weights, .height = h, .width = w });
}
