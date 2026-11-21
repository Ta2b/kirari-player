//! Appariement global de GMFlow s05 en deux sens, sur la grille de 36 × 60.
const std = @import("std");
const zml = @import("zml/core");
const Tensor = zml.Tensor;
const tokens = 36 * 60;

pub fn project(scores: Tensor, coords: Tensor) Tensor {
    // Même softmax stable que normalize() dans src/gpu/zml/attention.cpp.
    const exp = scores.sub(scores.max(2)).exp();
    const prob = exp.div(exp.sum(2));
    return prob.dotGeneral(coords, &.{.{ 2, 1 }}, &.{.{ 0, 0 }});
}

const rows: i64 = 720; // Douze lignes complètes de la grille de 36×60.

const Stripe = struct {
    q: Tensor,
    k: Tensor,
    divisor: Tensor,
    xy_forward: Tensor,
    xy_reverse: Tensor,

    pub const State = struct { first: Tensor, forward: Tensor, peak: Tensor, mass: Tensor, moments: Tensor };

    pub fn cond(_: Stripe, s: State) Tensor {
        return s.first.cmp(.LT, .scalar(tokens, .i32));
    }

    pub fn body(self: Stripe, s: State) State {
        const compiler = zml.Compiler.current();
        compiler.pushLocation(@src(), "/flownet/MatMul/correlation_bande");
        const query = self.q.slice(1, .dyn(s.first, rows));
        const scores = query.dotGeneral(self.k, &.{.{ 2, 1 }}, &.{.{ 0, 0 }}).div(self.divisor);
        compiler.popLocation();
        // Chaque ligne a toutes ses clés ; la projection aller est définitive.
        const forward = project(scores, self.xy_forward);
        // Chaque colonne acquiert 720 clés. Pondérer les sommes déjà calculées
        // par exp(ancien maximum - nouveau maximum) avant de réunir les bandes.
        const peak = s.peak.maximum(scores.max(1));
        const keep = s.peak.sub(peak).exp();
        const weights = scores.sub(peak).exp();
        const xy = self.xy_reverse.slice(1, .dyn(s.first, rows));
        const terms = weights.transpose(.{ 0, 2, 1 }).dotGeneral(xy, &.{.{ 2, 1 }}, &.{.{ 0, 0 }});
        return .{
            .first = s.first.addConstant(rows),
            .forward = s.forward.dynamicUpdateSlice1d(forward, 1, s.first),
            .peak = peak,
            .mass = s.mass.mul(keep).add(weights.sum(1)),
            .moments = s.moments.mul(keep.reshape(.{ 1, tokens, 1 })).add(terms),
        };
    }
};

pub fn streamed(q: Tensor, k: Tensor, scale: Tensor, coords: Tensor) !Tensor {
    const values = [_]Tensor{ q, k, scale, coords };
    const shapes = [_][]const i64{ &.{ 1, tokens, 128 }, &.{ 1, 128, tokens }, &.{}, &.{ 2, tokens, 2 } };
    for (values, shapes) |v, dims| {
        if (v.dtype() != .f32 or !std.mem.eql(i64, v.dims(), dims)) return error.InvalidGlobalMatcher;
    }
    const ctx: Stripe = .{
        .q = q,
        .k = k,
        .divisor = scale,
        .xy_forward = coords.slice(0, .{ .end = 1 }),
        .xy_reverse = coords.slice(0, .{ .start = 1, .end = 2 }),
    };
    const moments = Tensor.zeroes(.init(.{ 1, tokens, 2 }, .f32));
    const mass = Tensor.zeroes(.init(.{ 1, 1, tokens }, .f32));
    const state = zml.ops.@"while"(Stripe, ctx, .{
        .first = .scalar(0, .i32),
        // Le traceur de while associe les valeurs aux arguments par Tensor.id.
        // L'aller, entièrement réécrit, doit avoir sa propre valeur initiale.
        .forward = Tensor.scalar(1, .f32).broad(moments.shape()),
        .moments = moments,
        .mass = mass,
        .peak = Tensor.scalar(-std.math.inf(f32), .f32).broad(mass.shape()),
    });
    const reverse = state.moments.div(state.mass.reshape(.{ 1, tokens, 1 }));
    return Tensor.concatenate(&.{ state.forward, reverse }, 0);
}
