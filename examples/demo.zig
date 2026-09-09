// SPDX-License-Identifier: BSD-2-Clause

//! A tour of Fluxion Physics. Run it with `zig build example`.
//!
//! No window: a physics step is arithmetic and the picture is somebody
//! else's job. What it shows is a heap of crates and balls poured into a
//! bowl, stepped once with every core and once with none - the second
//! being exactly what a browser does - how long each took, whether the two
//! agree to the bit, and the last frame as characters.

const std = @import("std");
const Io = std.Io;
const physics = @import("fluxion_physics");
const World = physics.World;
const Jobs = physics.Jobs;
const Vec2 = physics.Vec2;

const body_count = 600;
const step_count = 240;
const dt: f32 = 1.0 / 60.0;

pub fn main(init: std.process.Init) !void {
    const gpa = init.arena.allocator();
    const io = init.io;

    var stdout_buffer: [4096]u8 = undefined;
    var stdout: Io.File.Writer = .init(.stdout(), io, &stdout_buffer);
    const out = &stdout.interface;

    try out.print("--- {d} bodies, {d} steps ---\n", .{ body_count, step_count });

    const threaded = try run(gpa, .{ .io = io, .workers = .auto }, io, out);
    defer gpa.free(threaded.positions);
    const alone = try run(gpa, .{ .io = null }, io, out);
    defer gpa.free(alone.positions);

    try out.print("same world: {}\n\n", .{std.mem.eql(f32, threaded.positions, alone.positions)});
    try out.writeAll(alone.picture);
    try out.flush();
}

const Result = struct {
    positions: []f32,
    picture: []const u8,
};

fn run(gpa: std.mem.Allocator, options: Jobs.Options, io: Io, out: *Io.Writer) !Result {
    var jobs: Jobs = try .init(gpa, options);
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();

    const bodies = try build(&world);

    const started = Io.Clock.now(.awake, io);
    for (0..step_count) |_| try world.step(dt, &jobs);
    const finished = Io.Clock.now(.awake, io);
    const ms = @as(f64, @floatFromInt(started.durationTo(finished).toNanoseconds())) / 1e6;

    try out.print("{d: >2} workers: {d:.1} ms, {d} touching, {d} colours\n", .{
        jobs.workerCount(),
        ms,
        world.touchingCount(),
        world.colourCount(),
    });

    const positions = try gpa.alloc(f32, bodies.len * 3);
    for (bodies, 0..) |b, i| {
        const body = world.body(b).?;
        positions[i * 3] = body.position().x;
        positions[i * 3 + 1] = body.position().y;
        positions[i * 3 + 2] = body.angle;
    }
    return .{ .positions = positions, .picture = try draw(gpa, &world) };
}

/// A bowl: a floor and two leaning walls, and a column of bodies above it.
fn build(world: *World) ![]physics.BodyId {
    const gpa = world.gpa;
    const floor = try world.createBody(.{ .type = .static, .position = .init(0, 8.5) });
    _ = try world.addShape(floor, .box(14, 1));
    const left = try world.createBody(.{ .type = .static, .position = .init(-10, 2), .angle = -0.35 });
    _ = try world.addShape(left, .box(1, 8));
    const right = try world.createBody(.{ .type = .static, .position = .init(10, 2), .angle = 0.35 });
    _ = try world.addShape(right, .box(1, 8));

    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();

    // A grid rather than a column, so the heap forms in the first second
    // and the timing below measures a pile of contacts and not a fall.
    const columns = 24;
    const bodies = try gpa.alloc(physics.BodyId, body_count);
    for (bodies, 0..) |*b, i| {
        const column: f32 = @floatFromInt(i % columns);
        const row: f32 = @floatFromInt(i / columns);
        b.* = try world.createBody(.{
            .position = .init((column - columns / 2) * 0.8 + random.float(f32) * 0.2, 6 - row * 0.8),
            .angle = random.float(f32) * 6,
        });
        switch (i % 3) {
            0 => _ = try world.addShape(b.*, .{
                .geometry = .{ .circle = .{ .radius = 0.2 + random.float(f32) * 0.2 } },
                .material = .{ .restitution = 0.3 },
            }),
            1 => _ = try world.addShape(b.*, .box(0.2 + random.float(f32) * 0.25, 0.2 + random.float(f32) * 0.25)),
            else => _ = try world.addShape(b.*, .{
                .geometry = .{ .polygon = try .fromPoints(&.{
                    .init(-0.35, 0.2), .init(0.35, 0.2), .init(0, -0.4),
                }) },
            }),
        }
    }
    return bodies;
}

/// The bowl as text: `o` is a ball, `#` a polygon, `=` the walls.
fn draw(gpa: std.mem.Allocator, world: *World) ![]const u8 {
    const width: usize = 72;
    const height: usize = 26;
    const cols: f32 = @floatFromInt(width);
    const rows: f32 = @floatFromInt(height);
    const cells = try gpa.alloc(u8, (width + 1) * height);
    @memset(cells, ' ');
    for (0..height) |r| cells[r * (width + 1) + width] = '\n';

    var it = world.shapeIterator();
    while (it.next()) |entry| {
        const xf = world.shapeTransform(entry.value);
        const box = entry.value.def.geometry.aabb(xf);
        const body = world.bodyConst(entry.value.body).?;
        const glyph: u8 = if (body.type == .static) '=' else switch (entry.value.def.geometry) {
            .circle => 'o',
            .polygon => '#',
        };
        // Twelve metres across the width, y down, and a step of a third
        // of a metre per cell so nothing but a wall fills more than one.
        var y = box.min.y;
        while (y <= box.max.y) : (y += 0.5) {
            var x = box.min.x;
            while (x <= box.max.x) : (x += 0.35) {
                const c: i32 = @intFromFloat(@round((x + 12.5) * (cols / 25.0)));
                const r: i32 = @intFromFloat(@round((y + 4) * (rows / 13.0)));
                if (c < 0 or c >= width or r < 0 or r >= height) continue;
                cells[@as(usize, @intCast(r)) * (width + 1) + @as(usize, @intCast(c))] = glyph;
            }
        }
    }
    return cells;
}

test "the demo scene settles and both schedulers agree" {
    const gpa = std.testing.allocator;
    var results: [2][]f32 = undefined;
    const modes = [_]Jobs.Options{ .{ .io = std.testing.io }, .{ .io = null } };
    for (modes, 0..) |mode, i| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();
        const bodies = try build(&world);
        defer gpa.free(bodies);
        for (0..60) |_| try world.step(dt, &jobs);
        results[i] = try gpa.alloc(f32, bodies.len);
        for (bodies, results[i]) |b, *y| y.* = world.body(b).?.position().y;
    }
    defer for (results) |r| gpa.free(r);
    try std.testing.expectEqualSlices(f32, results[0], results[1]);
}
