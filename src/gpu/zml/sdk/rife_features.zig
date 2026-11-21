// Les deux premières images du tenseur RIFE sont les seules entrées de l'encodeur.
// Le module froid expose ses traits en troisième sortie, puis ils restent dans PJRT.
const std = @import("std");
const Tensor = @import("zml/core").Tensor;

pub const Phase = enum { froid, instants, suite };

fn fp16(p: anytype, id: u32, shape: []const i64) bool {
    return id < p.tensors.len and p.tensors[id].dtype == .f16 and std.mem.eql(i64, p.tensors[id].dims, shape);
}

pub fn recognized(p: anytype) bool {
    if (p.family != .motion or p.io != .f16 or p.inputs.len != 1 or p.outputs.len != 2 or p.nodes.len < 3 or
        !fp16(p, p.inputs[0], &.{ 1, 11, 768, 1280 }) or !fp16(p, p.outputs[0], &.{ 1, 4, 768, 1280 }) or
        !fp16(p, p.outputs[1], &.{ 1, 1, 768, 1280 })) return false;
    const split = p.nodes[0];
    const rgb = p.nodes[1];
    const encode = p.nodes[2];
    if (!std.mem.eql(u8, split.op, "Split") or split.ins.len < 1 or split.ins[0] != p.inputs[0] or split.outs.len != 4 or
        !fp16(p, split.outs[0], &.{ 1, 6, 768, 1280 }) or !std.mem.eql(u8, rgb.op, "Reshape") or rgb.ins.len == 0 or
        rgb.ins[0] != split.outs[0] or rgb.outs.len != 1 or !fp16(p, rgb.outs[0], &.{ 2, 3, 768, 1280 }) or
        !std.mem.eql(u8, encode.op, "KirariEncoder") or encode.ins.len != 9 or encode.ins[0] != rgb.outs[0] or
        encode.outs.len != 1 or !fp16(p, encode.outs[0], &.{ 2, 4, 768, 1280 })) return false;
    for (encode.ins[1..]) |id| if (id >= p.tensors.len or p.tensors[id].offset == null) return false;
    const widths = [_]i64{ 96, 64, 48, 32, 16 };
    var blocks: usize = 0;
    for (p.nodes[3..]) |n| {
        if (std.mem.eql(u8, n.op, "KirariEncoder")) return false;
        if (!std.mem.eql(u8, n.op, "KirariIFBlock")) continue;
        if (blocks >= widths.len or n.ins.len < 2 or !fp16(p, n.ins[1], &.{ widths[blocks], if (blocks == 0) 15 else 28, 3, 3 })) return false;
        blocks += 1;
    }
    return blocks == widths.len;
}

pub fn specialize(p: anytype, phase: Phase, alloc: std.mem.Allocator) !@TypeOf(p) {
    var variant = p;
    const enc = p.nodes[2];
    const features = enc.outs[0];
    if (phase != .instants) {
        const outputs = try alloc.alloc(u32, 3);
        @memcpy(outputs[0..2], p.outputs);
        outputs[2] = features;
        variant.outputs = outputs;
    }
    if (phase == .froid) return variant;
    const inputs = try alloc.alloc(u32, 2);
    inputs[0] = p.inputs[0];
    variant.inputs = inputs;
    if (phase == .instants) {
        inputs[1] = features;
        const nodes = try alloc.alloc(@TypeOf(enc), p.nodes.len - 1);
        @memcpy(nodes[0..2], p.nodes[0..2]);
        @memcpy(nodes[2..], p.nodes[3..]);
        variant.nodes = nodes;
    } else {
        inputs[1] = @intCast(p.tensors.len);
        const tensors = try alloc.alloc(@TypeOf(p.tensors[0]), p.tensors.len + 1);
        @memcpy(tensors[0..p.tensors.len], p.tensors);
        tensors[p.tensors.len] = p.tensors[features];
        variant.tensors = tensors;
        const nodes = try alloc.dupe(@TypeOf(enc), p.nodes);
        const args = try alloc.alloc(u32, enc.ins.len + 1);
        @memcpy(args[0..enc.ins.len], enc.ins);
        args[enc.ins.len] = inputs[1];
        nodes[2].op = "KirariRifeNextEncoder";
        nodes[2].ins = args;
        variant.nodes = nodes;
    }
    return variant;
}

pub fn following(inputs: []Tensor, integers: []const i64, layers: []const []const u8) !Tensor {
    if (inputs.len != 10) return error.InvalidMotionGroup;
    const old_b = inputs[9].slice(0, .{ .start = 1, .end = 2 });
    inputs[0] = inputs[0].slice(0, .{ .start = 1, .end = 2 });
    const new_b = try @import("rife_blocks.zig").fromGraph("KirariEncoder", inputs[0..9], integers, layers);
    return Tensor.concatenate(&.{ old_b, new_b }, 0);
}
