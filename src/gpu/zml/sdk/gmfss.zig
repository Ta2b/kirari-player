//! Produits matriciels ONNX de GMFlow, y compris les projections diffusées sur les fenêtres.
const zml = @import("zml/core");
const Tensor = zml.Tensor;

pub fn matmul(lhs: Tensor, rhs: Tensor, out: zml.Shape) !Tensor {
    const rank = out.rank();
    if (rank < 2 or rank > 8 or lhs.rank() < 2 or rhs.rank() < 2 or lhs.rank() > rank or rhs.rank() > rank or
        lhs.dtype() != rhs.dtype() or lhs.dtype() != out.dtype() or lhs.dim(-1) != rhs.dim(-2) or
        lhs.dim(-2) != out.dim(-2) or rhs.dim(-1) != out.dim(-1)) return error.InvalidMatMul;
    var batch: [6][2]i8 = undefined;
    for (batch[0 .. rank - 2], 0..) |*axis, i| {
        const index: i8 = @intCast(i);
        axis.* = .{ index, index };
        for ([_]Tensor{ lhs, rhs }) |v| {
            const leading = rank - v.rank();
            if (i < leading) continue;
            const size = v.dim(@as(i8, @intCast(i - leading)));
            if (size != 1 and size != out.dim(index)) return error.InvalidMatMul;
        }
    }
    const a = lhs.broadcastLeft(out.setDim(-1, lhs.dim(-1)));
    const b = rhs.broadcastLeft(out.setDim(-2, rhs.dim(-2)));
    return a.dotGeneral(b, &.{.{ -1, -2 }}, batch[0 .. rank - 2]);
}

// Projection de l’upsampler GMFlow telle qu’exprimée dans le graphe de référence.
// Les deux opérandes gardent C en axe contractant. Un changement de plan près du
// softmax de ces poids peut déplacer les coordonnées du flot interpolé.
pub fn reduceChannels(x: Tensor, weights: Tensor, bias: ?Tensor, opts: @import("motion_conv.zig").ConvOptions) ?Tensor {
    if (opts.transposed or opts.group != 1 or weights.dim(2) != 1 or weights.dim(3) != 1 or
        weights.dim(1) <= weights.dim(0) or opts.strides[0] != 1 or opts.strides[1] != 1) return null;
    for (opts.pads) |pad| if (pad != 0) return null;
    const n = x.dim(0);
    const h = x.dim(2);
    const w = x.dim(3);
    const c = weights.dim(1);
    const out = weights.dim(0);
    const pixels = x.transpose(.{ 0, 2, 3, 1 }).reshape(.{ n * h * w, c });
    const dot = pixels.dotGeneral(weights.reshape(.{ out, c }), &.{.{ 1, 1 }}, &.{});
    const result = dot.reshape(.{ n, h, w, out }).transpose(.{ 0, 3, 1, 2 });
    return if (bias) |b| result.add(b.broadcast(result.shape(), &.{1})) else result;
}
