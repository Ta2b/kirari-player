const std = @import("std");
const zml = @import("zml/core");
const mlir = @import("mlir");
const func = @import("mlir/dialects/func");

// Le SDK émet certaines opérations élémentaires avec UnknownLoc, même dans un contexte nommé.
// Les liaisons Zig n'exposent pas encore le mutateur de l'API C MLIR ni un next nullable.
const RawOperation = @TypeOf(@as(*mlir.Operation, undefined).ptr());
extern fn mlirOperationGetNextInBlock(op: RawOperation) RawOperation;
extern fn mlirOperationSetLocation(op: RawOperation, loc: @TypeOf(@as(*const mlir.Location, undefined).ptr())) void;

pub const SourceLocations = struct {
    last: ?*mlir.Operation = null,

    pub fn advance(self: *SourceLocations, compiler: *zml.Compiler) void {
        var op = if (self.last) |last| nextOperation(last) else compiler.currentScope().block.firstOperation();
        while (op) |current| : (op = nextOperation(current)) {
            if (compiler.location != compiler.unknown_location) switch (current.location().inspect()) {
                .unknown, .file_line_col => mlirOperationSetLocation(current.ptr(), compiler.location.ptr()),
                else => {},
            };
            self.last = current;
        }
    }

    pub fn nextOperation(op: *mlir.Operation) ?*mlir.Operation {
        return @ptrCast(mlirOperationGetNextInBlock(op.ptr()).ptr);
    }
};

// Le contexte fournit l'IR et les passes de ZML. Le modèle sera compilé dans le client PJRT
// du lecteur, sur l'appareil déjà choisi pour la vidéo.
pub fn write(init: std.process.Init, path: []const u8, name: []const u8, model: anytype) !void {
    const allocator = init.gpa;
    const io = init.io;
    const context: *zml.Platform = try .init(allocator, io, .cpu, .{});
    defer context.deinit(allocator, io);
    var compiler = zml.Compiler.init(allocator, io, context, .{ .program_name = name });
    defer compiler.deinit();
    const block = mlir.Block.init(&.{}, &.{});
    var owned = true;
    defer if (owned) block.deinit();
    {
        const scope = compiler.pushBlock(block);
        defer scope.pop();
        compiler.activate();
        defer compiler.deactivate();
        try model.trace(&compiler, scope);
    }
    _ = func.func(compiler.mlir_ctx, .{ .name = "main", .block = block, .location = compiler.unknown_location }).appendTo(compiler.module.body());
    owned = false;
    try compiler.mlir_pass_manager.runOnOp(compiler.module.operation());
    if (!compiler.module.operation().verify()) return error.InvalidGraph;
    const file = try std.Io.Dir.cwd().createFile(io, path, .{});
    defer file.close(io);
    var buf: [8192]u8 = undefined;
    var writer = file.writer(io, &buf);
    // Le format pretty de MLIR est réservé à la lecture humaine. La forme standard permet
    // au client PJRT de reparcourir les NameLoc et CallSiteLoc lors de la conversion XLA.
    try writer.interface.print("{f}", .{compiler.module.operation().fmt(.{ .debug_info = true, .debug_info_pretty_form = false })});
    try writer.interface.flush();
}
