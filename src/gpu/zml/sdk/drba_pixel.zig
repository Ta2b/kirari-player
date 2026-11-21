const zml = @import("zml/core");
const Tensor = zml.Tensor;

// Le voisinage des deux warps dépend du pixel et reste commun aux trois couleurs.
// La barrière matérialise un pixel de quatre demi-flottants. Le quatrième est nul.
pub fn rgb(output: Tensor) !Tensor {
    if (output.dtype() != .f16 or output.rank() != 4 or output.dim(0) != 1 or output.dim(1) != 3 or
        output.dim(2) != 1088 or output.dim(3) != 1920) return error.InvalidDrbaOutput;
    const compiler = zml.Compiler.current();
    compiler.pushLocation(@src(), "DistilDRBA/warp_final_rgb64");
    defer compiler.popLocation();
    const pixels = output.withTags(.{ .n, .c, .h, .w }).transpose(.{ 0, 2, 3, 1 });
    const rgba = pixels.pad(0, .{ .c = Tensor.Pad{ .high = 1 } });
    const word = rgba.bitCast(.u64).optimizationBarrier();
    return word.bitCast(.f16).slice(3, .{ .end = 3 }).transpose(.{ 0, 3, 1, 2 }).withTags(output.shape().tags());
}
