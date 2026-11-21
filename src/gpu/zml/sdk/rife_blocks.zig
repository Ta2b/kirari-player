//! Encodeur partagé et IFBlock de RIFE 4.18, 4.22 lite, 4.25, 4.25 lite et 4.26.
//! Les tenseurs sont les opérandes du graphe ONNX reconnu, dans leur ordre d'origine.
const std = @import("std");
const zml = @import("zml/core");
const conv = @import("motion_conv.zig");
const activation = @import("rife_activation.zig");
const Tensor = zml.Tensor;

pub const Affine = struct {
    weight: Tensor,
    bias: Tensor,
    name: []const u8,

    fn apply(self: Affine, x: Tensor, options: conv.ConvOptions) Tensor {
        const compiler = zml.Compiler.current();
        compiler.pushLocation(@src(), self.name);
        defer compiler.popLocation();
        return conv.convolution(x, self.weight, self.bias, options);
    }
};

pub const Encoder = struct {
    // encode.cnn0, cnn1, cnn2 et cnn3. DistilDRBA les partage entre les trois images.
    stages: [4]Affine,

    pub fn forward(self: Encoder, rgb: Tensor, polyphase: bool, use_select: bool) Tensor {
        var x = rgb;
        for (self.stages[0..3], 0..) |layer, i| {
            const step: i64 = if (i == 0) 2 else 1;
            x = activation.leaky(layer.apply(x, .{
                .strides = .{ step, step },
                .pads = .{ 1, 1, 1, 1 },
            }), use_select);
        }
        const up = self.stages[3];
        return up.apply(x, .{
            .transposed = true,
            .strides = .{ 2, 2 },
            .pads = .{ 1, 1, 1, 1 },
            .polyphase = polyphase,
        });
    }
};

pub const Residual = struct {
    layer: Affine,
    // blockK.convblock.i.beta, absent de 4.25 lite dont le corps est une série de Conv + LeakyRelu.
    beta: ?Tensor,

    pub fn forward(self: Residual, x: Tensor, use_select: bool) Tensor {
        const y = self.layer.apply(x, .{ .pads = .{ 1, 1, 1, 1 } });
        const before_activation = if (self.beta) |beta| y.mul(beta.broad(y.shape())).add(x) else y;
        return activation.leaky(before_activation, use_select);
    }
};

pub const IFBlock = struct {
    // blockK.conv0.0.0 et blockK.conv0.1.0, convblock.0…7, lastconv.0 en IOHW.
    down: [2]Affine,
    body: [8]Residual,
    head: Affine,

    /// Entrée NCHW déjà réduite du facteur du bloc. Sortie à la même résolution après le
    /// DepthToSpace CRD de bloc 2 : 4 plans de flot, un masque, puis les caractéristiques lues
    /// par le prochain bloc. Le reconnaisseur fournit les poids réduits à vingt filtres
    /// uniquement si les sorties au-delà des cinq premiers plans sont mortes.
    pub fn forward(self: IFBlock, input: Tensor, polyphase: bool, use_select: bool, select_body: bool) Tensor {
        var x = input;
        for (self.down) |layer| x = activation.leaky(layer.apply(x, .{
            .strides = .{ 2, 2 },
            .pads = .{ 1, 1, 1, 1 },
        }), use_select);
        for (self.body) |unit| x = unit.forward(x, use_select or select_body);
        const y = self.head.apply(x, .{
            .transposed = true,
            .strides = .{ 2, 2 },
            .pads = .{ 1, 1, 1, 1 },
            .polyphase = polyphase,
        });
        return conv.depthToSpace(y, 2, true);
    }
};

fn affineAt(inputs: []const Tensor, index: usize, layers: []const []const u8, stage: usize) !Affine {
    const pair = inputs[index..][0..2];
    if (pair[0].rank() != 4 or pair[1].rank() != 1) return error.InvalidMotionWeights;
    return .{ .weight = pair[0], .bias = pair[1], .name = if (layers.len > 0) layers[stage] else "Conv" };
}

/// Les membres ont été reconnus par leurs arêtes et leurs attributs dans motion_groups.cpp.
pub fn fromGraph(op: []const u8, inputs: []const Tensor, integers: []const i64, layers: []const []const u8) !Tensor {
    if (std.mem.eql(u8, op, "KirariEncoder")) {
        if (inputs.len != 9 or (integers.len != 1 and integers.len != 2) or (layers.len != 0 and layers.len != 4)) return error.InvalidMotionGroup;
        var encoder: Encoder = undefined;
        for (&encoder.stages, 0..) |*stage, i| stage.* = try affineAt(inputs, 1 + 2 * i, layers, i);
        return encoder.forward(inputs[0], integers[0] != 0, integers.len == 2 and integers[1] == 1);
    }
    if (!std.mem.eql(u8, op, "KirariIFBlock") or (integers.len != 2 and integers.len != 3) or integers[0] < 0 or integers[0] > 255)
        return error.InvalidMotionGroup;
    const mask: u8 = @intCast(integers[0]);
    if (inputs.len != 23 + @as(usize, @popCount(mask)) or (layers.len != 0 and layers.len != 11)) return error.InvalidMotionGroup;
    var block: IFBlock = undefined;
    for (&block.down, 0..) |*stage, i| stage.* = try affineAt(inputs, 1 + 2 * i, layers, i);
    for (&block.body, 0..) |*unit, i| {
        const bit = @as(u8, 1) << @as(u3, @intCast(i));
        const index = 5 + 2 * i + @as(usize, @popCount(mask & (bit - 1)));
        unit.* = .{
            .layer = try affineAt(inputs, index, layers, 2 + i),
            .beta = if (mask & bit != 0) inputs[index + 2] else null,
        };
    }
    // Poids et biais de ConvTranspose terminent toujours le groupe, même si le dernier
    // bloc a un coefficient bêta. Les placer depuis la fin suit exactement le manifeste.
    block.head = try affineAt(inputs, inputs.len - 2, layers, 10);
    // 1 sélectionne toutes les activations, 2 seulement celles du corps dont
    // le manifeste a incorporé l'identité aux filtres après reconnaissance.
    const mode = if (integers.len == 3) integers[2] else 0;
    if (mode < 0 or mode > 2 or (mode == 2 and mask != 0)) return error.InvalidMotionGroup;
    return block.forward(inputs[0], integers[1] != 0, mode == 1, mode == 2);
}
