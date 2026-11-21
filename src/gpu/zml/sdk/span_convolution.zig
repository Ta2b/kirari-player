const zml = @import("zml/core");
const Tensor = zml.Tensor;

pub const Kernel = struct {
    filter: Tensor,
    bias: Tensor,

    // Activations NHWC et poids OHWI, y compris les quatre vues de conv_cat
    // qui forment un seul axe contracté de 4*C canaux.
    pub fn apply(self: Kernel, x: Tensor) Tensor {
        const radius = @divFloor(self.filter.dim(1), 2);
        const y = x.conv2d(self.filter, .{
            .padding = &.{ radius, radius, radius, radius },
            .input_feature_dimension = 3,
            .input_spatial_dimensions = &.{ 1, 2 },
            .kernel_input_feature_dimension = 3,
            .kernel_spatial_dimensions = &.{ 1, 2 },
            .output_spatial_dimensions = &.{ 1, 2 },
            .output_feature_dimension = 3,
        });
        return y.add(self.bias.broadcast(y.shape(), &.{3}));
    }
};

const rows = 30;
pub const Gate = struct { residual: Tensor, logits: Tensor };

fn Projection(comptime deferred_gate: bool) type {
    return struct {
        views: [if (deferred_gate) 6 else 4]Tensor,
        kernel: Kernel,
        pub const State = struct { first_row: Tensor, output: Tensor };

        pub fn cond(self: @This(), s: State) Tensor {
            return s.first_row.cmp(.LT, .scalar(self.views[0].dim(1), .i32));
        }

        pub fn body(self: @This(), s: State) State {
            var bands: [4]Tensor = undefined;
            for (&bands, self.views[0..4]) |*band, view| band.* = view.slice(1, .dyn(s.first_row, rows));
            if (deferred_gate) {
                const residual = self.views[4].slice(1, .dyn(s.first_row, rows));
                const logits = self.views[5].slice(1, .dyn(s.first_row, rows));
                bands[1] = bands[1].add(residual).mul(logits.tanh());
            }
            const projected = self.kernel.apply(Tensor.concatenate(&bands, 3));
            return .{
                .first_row = s.first_row.addConstant(rows),
                .output = s.output.dynamicUpdateSlice1d(projected, 1, s.first_row),
            };
        }

        fn run(self: @This()) Tensor {
            // Le 1×1 écrit exactement ses 30 lignes et n'a aucun halo.
            return zml.ops.@"while"(@This(), self, .{
                .first_row = .scalar(0, .i32),
                .output = Tensor.zeroes(self.views[0].shape()),
            }).output;
        }
    };
}

pub fn supportsProjectionStrips(height: i64, width: i64, channels: i64) bool {
    return height == 540 and width == 960 and (channels == 48 or channels == 64);
}

pub fn projectStrips(views: [4]Tensor, kernel: Kernel, gate: ?Gate) Tensor {
    if (gate) |tail| {
        return (Projection(true){ .views = views ++ .{ tail.residual, tail.logits }, .kernel = kernel }).run();
    }
    return (Projection(false){ .views = views, .kernel = kernel }).run();
}

pub fn arrangePixels(pixel: Tensor) Tensor {
    const h = pixel.dim(1);
    const w = pixel.dim(2);
    return pixel.reshape(.{ 1, h, w, 3, 4, 4 }).transpose(.{ 0, 3, 1, 4, 2, 5 }).reshape(.{ 1, 3, h * 4, w * 4 });
}

// Deux lignes de halo suffisent pour composer le 1×1 et le 3×3. Au bord on
// sélectionne les lignes extrêmes du résultat, où le 3×3 applique son propre zéro.
const PixelStrips = struct {
    views: [4]Tensor,
    projection: Kernel,
    pixel: Kernel,
    pub const State = struct { first_row: Tensor, output: Tensor };

    pub fn cond(self: PixelStrips, s: State) Tensor {
        return s.first_row.cmp(.LT, .scalar(self.views[0].dim(1), .i32));
    }
    pub fn body(self: PixelStrips, s: State) State {
        const height = self.views[0].dim(1);
        const first = s.first_row.addConstant(-1).clamp(.scalar(0, .i32), .scalar(height - rows - 2, .i32));
        var bands: [4]Tensor = undefined;
        for (&bands, self.views) |*band, view| band.* = view.slice(1, .dyn(first, rows + 2));
        const merged = self.projection.apply(Tensor.concatenate(&bands, 3));
        const valid = self.pixel.apply(merged).slice(1, .dyn(s.first_row.sub(first), rows));
        return .{
            .first_row = s.first_row.addConstant(rows),
            .output = s.output.dynamicUpdateSlice1d(arrangePixels(valid), 2, s.first_row.scale(4)),
        };
    }
};

pub fn projectPixelStrips(views: [4]Tensor, projection: Kernel, pixel: Kernel) Tensor {
    const ctx: PixelStrips = .{ .views = views, .projection = projection, .pixel = pixel };
    return zml.ops.@"while"(PixelStrips, ctx, .{
        .first_row = .scalar(0, .i32),
        .output = Tensor.zeroes(.init(.{ 1, 3, 4 * views[0].dim(1), 4 * views[0].dim(2) }, .f16)),
    }).output;
}
