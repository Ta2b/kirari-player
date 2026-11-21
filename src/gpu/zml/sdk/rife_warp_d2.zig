//! Pied bilinéaire D2 du RIFE 4.25 dense. Les deux images suivent des grilles distinctes.
const std = @import("std");
const Tensor = @import("zml/core").Tensor;

pub const Foot = struct { indices: Tensor, wx: Tensor, wy: Tensor, valid: Tensor };
const h: i64 = 768;
const w: i64 = 1280;

const Axis = struct {
    size: i64,
    index_bit: u6,
    index_mask: i64,
    fraction_bit: u6,

    fn encode(self: Axis, pixel: Tensor) Tensor {
        const last: f32 = @floatFromInt(self.size - 2);
        const origin = pixel.floor().minimum(.scalar(last, .f32));
        const packed_index = origin.convert(.i32).convert(.i64).shiftLeft(.scalar(self.index_bit, .i64));
        const fraction = pixel.sub(origin).scale(1048576).roundNearestEven().convert(.i64);
        return packed_index.logical(.OR, fraction.shiftLeft(.scalar(self.fraction_bit, .i64)));
    }

    fn index(self: Axis, word: Tensor) Tensor {
        return word.shiftRightLogical(.scalar(self.index_bit, .i64)).logical(.AND, .scalar(self.index_mask, .i64)).convert(.i32);
    }

    fn weight(self: Axis, word: Tensor) Tensor {
        const q20 = word.shiftRightLogical(.scalar(self.fraction_bit, .i64)).logical(.AND, .scalar(2097151, .i64));
        return q20.convert(.f32).scale(0x1p-20);
    }
};

pub fn footprint(px: Tensor, py: Tensor, valid: Tensor) Foot {
    // Les fractions 0 et 1 sont exactes. L'erreur maximale de l'arrondi Q20 est de 2⁻²¹.
    const x = Axis{ .size = w, .index_bit = 0, .index_mask = 2047, .fraction_bit = 21 };
    const y = Axis{ .size = h, .index_bit = 11, .index_mask = 1023, .fraction_bit = 42 };
    const bits = x.encode(px).logical(.OR, y.encode(py));
    const flag = valid.convert(.i64).shiftLeft(.scalar(63, .i64));
    const word = valid.select(bits, .scalar(0, .i64)).logical(.OR, flag).optimizationBarrier();
    return .{
        .indices = Tensor.stack(&.{ y.index(word), x.index(word) }, .last, .coord),
        .wx = x.weight(word),
        .wy = y.weight(word),
        .valid = word.shiftRightLogical(.scalar(63, .i64)).convert(.bool),
    };
}

fn tensor(p: anytype, id: u32, dtype: @TypeOf(p.io), dims: []const i64) bool {
    return id < p.tensors.len and p.tensors[id].dtype == dtype and std.mem.eql(i64, p.tensors[id].dims, dims);
}

fn edge(child: anytype, input: usize, parent: anytype, output: usize) bool {
    return input < child.ins.len and output < parent.outs.len and child.ins[input] == parent.outs[output];
}

fn target(p: anytype, nodes: anytype) bool {
    if (nodes.len < 9) return false;
    const ops = [_][]const u8{ "GridSample", "Cast", "Split", "Reshape", "Reshape", "Mul", "Concat", "Resize", "KirariIFBlock" };
    for (ops, nodes[0..9]) |op, n| if (!std.mem.eql(u8, n.op, op)) return false;
    const warp = nodes[0];
    const down = nodes[7];
    if (!std.mem.eql(u8, warp.name, "/GridSample_2") or !std.mem.eql(u8, down.name, "/block3/Resize") or
        !std.mem.eql(i64, warp.i, &.{ 0, 1, 1, 0 }) or !std.mem.eql(i64, down.i, &.{ 1, 0, 2 }) or
        !std.mem.eql(f32, down.f, &.{ 0.5, 0.5 }) or warp.ins.len != 2 or warp.outs.len != 1 or
        !tensor(p, warp.ins[0], .f32, &.{ 2, 7, h, w }) or !tensor(p, warp.ins[1], .f32, &.{ 2, h, w, 2 }) or
        !tensor(p, warp.outs[0], .f32, &.{ 2, 7, h, w }) or nodes[6].ins.len != 6 or down.outs.len != 1 or
        !tensor(p, down.outs[0], .f16, &.{ 1, 28, h / 2, w / 2 })) return false;
    return edge(nodes[1], 0, warp, 0) and edge(nodes[2], 0, nodes[1], 0) and
        edge(nodes[3], 0, nodes[2], 0) and edge(nodes[4], 0, nodes[2], 1) and
        edge(nodes[6], 0, nodes[3], 0) and edge(nodes[6], 1, nodes[4], 0) and
        edge(down, 0, nodes[6], 0) and edge(nodes[8], 0, down, 0);
}

// Les blocs validés par le préparateur Kirari ont déjà leurs poids et leur câblage regroupés.
// Restreindre l'expérience aux cinq groupes livrés et à leur réduction de 28 canaux avant block3.
pub fn select(p: anytype) ?u32 {
    if (p.family != .motion or p.io != .f16 or p.inputs.len != 1 or p.outputs.len != 2 or
        !tensor(p, p.inputs[0], .f16, &.{ 1, 11, h, w }) or !tensor(p, p.outputs[0], .f16, &.{ 1, 4, h, w }) or
        !tensor(p, p.outputs[1], .f16, &.{ 1, 1, h, w })) return null;
    const widths = [_]i64{ 96, 64, 48, 32, 16 };
    var blocks: usize = 0;
    var encoders: usize = 0;
    var result: ?u32 = null;
    for (p.nodes, 0..) |n, i| {
        if (std.mem.eql(u8, n.op, "KirariEncoder")) encoders += 1;
        if (std.mem.eql(u8, n.op, "KirariIFBlock")) {
            if (blocks >= widths.len or n.ins.len < 2 or
                !tensor(p, n.ins[1], .f16, &.{ widths[blocks], if (blocks == 0) 15 else 28, 3, 3 })) return null;
            blocks += 1;
        }
        if (blocks == 3 and std.mem.eql(u8, n.op, "GridSample") and target(p, p.nodes[i..])) {
            if (result != null) return null;
            result = n.outs[0];
        }
    }
    return if (blocks == widths.len and encoders == 1) result else null;
}
