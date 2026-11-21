//! Convolutions ONNX des réseaux de mouvement. Images en NCHW, noyaux Conv en OIHW.
const std = @import("std");
const zml = @import("zml/core");
const Tensor = zml.Tensor;

pub const ConvOptions = struct {
    transposed: bool = false,
    strides: [2]i64 = .{ 1, 1 },
    dilations: [2]i64 = .{ 1, 1 },
    pads: [4]i64 = .{ 0, 0, 0, 0 }, // haut, gauche, bas, droite
    output_padding: [2]i64 = .{ 0, 0 },
    group: i64 = 1,
    polyphase: bool = false,
};

// Une convolution dense 2×2 traite les quatre phases en une fois. Le recadrage
// d'un pixel sur chaque bord correspond au padding ONNX (1,1,1,1).
fn phaseConv(x: Tensor, weights: Tensor) Tensor {
    const outputs = weights.dim(1);
    const inputs = weights.dim(0);
    const kernel = weights.reshape(.{ inputs, outputs, 2, 2, 2, 2 })
        .transpose(.{ 1, 3, 5, 0, 2, 4 }).reshape(.{ outputs * 4, inputs, 2, 2 }).reverse(.{ 2, 3 });
    const low = x.conv2d(kernel, .{ .padding = &.{ 1, 1, 1, 1 } });
    return depthToSpace(low, 2, true).slice(2, .{ .start = 1, .end = 1 + x.dim(2) * 2 })
        .slice(3, .{ .start = 1, .end = 1 + x.dim(3) * 2 });
}

pub fn convolution(x: Tensor, weights: Tensor, bias: ?Tensor, opts: ConvOptions) Tensor {
    const expected_channels = if (opts.transposed) weights.dim(0) else weights.dim(1) * opts.group;
    if (x.dim(1) != expected_channels) std.debug.panic(
        "convolution canaux ONNX incompatibles à {f}, entrée {any}, poids {any}, transposée {}, groupes {d}",
        .{ zml.Compiler.current().location, x.dims(), weights.dims(), opts.transposed, opts.group },
    );
    if (opts.polyphase and opts.transposed and opts.group == 1 and weights.dim(2) == 4 and weights.dim(3) == 4 and
        std.mem.eql(i64, &opts.strides, &.{ 2, 2 }) and std.mem.eql(i64, &opts.dilations, &.{ 1, 1 }) and
        std.mem.eql(i64, &opts.pads, &.{ 1, 1, 1, 1 }) and std.mem.eql(i64, &opts.output_padding, &.{ 0, 0 }))
    {
        const y = phaseConv(x, weights);
        return if (bias) |b| y.add(b.broadcast(y.shape(), &.{1})) else y;
    }
    var pad = [4]i64{ opts.pads[0], opts.pads[2], opts.pads[1], opts.pads[3] };
    var kernel = weights;
    if (opts.transposed) {
        // ONNX regroupe les canaux d'entrée : [g*in, out, ky, kx].
        const h = weights.dim(2);
        const w = weights.dim(3);
        const inputs = @divExact(weights.dim(0), opts.group);
        const outputs = weights.dim(1);
        kernel = weights.reshape(.{ opts.group, inputs, outputs, h, w })
            .transpose(.{ 0, 2, 1, 3, 4 }).reshape(.{ opts.group * outputs, inputs, h, w })
            .reverse(.{ 2, 3 });
        for (0..2) |axis| {
            const extent = (weights.dim(axis + 2) - 1) * opts.dilations[axis];
            pad[2 * axis] = extent - opts.pads[axis];
            pad[2 * axis + 1] = extent - opts.pads[axis + 2] + opts.output_padding[axis];
        }
    }
    const y = x.conv2d(kernel, .{
        .padding = &pad,
        .window_strides = if (opts.transposed) &.{ 1, 1 } else &opts.strides,
        .lhs_dilation = if (opts.transposed) &opts.strides else &.{ 1, 1 },
        .rhs_dilation = &opts.dilations,
        .feature_group_count = opts.group,
    });
    return if (bias) |b| y.add(b.broadcast(y.shape(), &.{1})) else y;
}

pub fn depthToSpace(x: Tensor, block: i64, crd: bool) Tensor {
    const n = x.dim(0);
    const c = @divExact(x.dim(1), block * block);
    const h = x.dim(2);
    const w = x.dim(3);
    const interleaved = if (crd)
        x.reshape(.{ n, c, block, block, h, w }).transpose(.{ 0, 1, 4, 2, 5, 3 })
    else
        x.reshape(.{ n, block, block, c, h, w }).transpose(.{ 0, 3, 4, 1, 5, 2 });
    return interleaved.reshape(.{ n, c, h * block, w * block });
}
