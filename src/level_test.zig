// SPDX-License-Identifier: BSD-2-Clause

//! The level: static shapes by the thousand, kept in a tree.
//!
//! What the tree must not change is any answer. A body lands on a tile map
//! the same way whether the map is a body per tile or one body of many
//! shapes; a ray, a point or a box asked of the level finds exactly what
//! asking every shape one by one finds; a level moved by hand, or a tile
//! added or taken away, is where it now is from the next step on. What it
//! must change is the cost: a step over a level of thousands of tiles looks
//! at the few that are near something moving.

const std = @import("std");
const testing = std.testing;

const physics = @import("root.zig");
const World = physics.World;
const Jobs = physics.Jobs;
const Vec2 = physics.Vec2;

const gpa = testing.allocator;

const dt: f32 = 1.0 / 60.0;

fn steps(world: *World, jobs: *Jobs, n: usize) !void {
    for (0..n) |_| try world.step(dt, jobs);
}

const Build = enum { body_per_tile, one_body };

/// A tile map `columns` wide and `rows` deep of half-metre tiles, its top
/// at y = 0, every tile's shape tagged with its number.
fn tileMap(world: *World, build: Build, columns: usize, rows: usize) !void {
    const level = if (build == .one_body) try world.createBody(.{ .type = .static }) else undefined;
    for (0..columns) |c| {
        for (0..rows) |r| {
            const at: Vec2 = .init(0.25 + @as(f32, @floatFromInt(c)) * 0.5, 0.25 + @as(f32, @floatFromInt(r)) * 0.5);
            const tag = c * rows + r;
            switch (build) {
                .one_body => _ = try world.addShape(level, .{ .geometry = .{ .polygon = .offsetBox(0.25, 0.25, at, 0) }, .user_data = tag }),
                .body_per_tile => {
                    const tile = try world.createBody(.{ .type = .static, .position = at });
                    _ = try world.addShape(tile, .{ .geometry = .{ .polygon = .box(0.25, 0.25) }, .user_data = tag });
                },
            }
        }
    }
}

test "crates land on a tile map the same way whether it is a body per tile or one body" {
    var landed: [2][20]Vec2 = undefined;
    for ([_]Build{ .body_per_tile, .one_body }, 0..) |build, bi| {
        var jobs: Jobs = try .init(gpa, .{});
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();
        // Sixty columns, twenty rows: twelve hundred tiles.
        try tileMap(&world, build, 60, 20);

        // Crates and balls, each on a column of its own, from a metre up.
        var dropped: [20]physics.BodyId = undefined;
        for (&dropped, 0..) |*d, i| {
            d.* = try world.createBody(.{ .position = .init(1.25 + @as(f32, @floatFromInt(i)) * 1.5, -1) });
            _ = try world.addShape(d.*, if (i % 2 == 0) .box(0.2, 0.2) else .circle(0.2));
        }
        try steps(&world, &jobs, 120);

        // A level of twelve hundred tiles, and a step looks at a handful of
        // pairs for each thing on it - never at two tiles.
        try testing.expect(world.pairs.items.len <= 3 * dropped.len);
        for (dropped, &landed[bi]) |d, *p| {
            p.* = world.body(d).?.position();
            // Resting on the top, a slop or so into it.
            try testing.expectApproxEqAbs(@as(f32, -0.2), p.y, 0.01);
        }
    }
    for (landed[0], landed[1]) |a, b| try testing.expect(a.distSq(b) < 1e-6);
}

test "a level in the tree answers every query as asking each shape in turn does" {
    // The same tiles twice: as static bodies, which the tree holds, and as
    // kinematic bodies standing still, which the queries walk one by one.
    // With holes, so rays have somewhere to go.
    var tree: World = .init(gpa, .{});
    defer tree.deinit();
    var walked: World = .init(gpa, .{});
    defer walked.deinit();
    var prng: std.Random.DefaultPrng = .init(5);
    const random = prng.random();
    var tag: u64 = 0;
    for (0..40) |c| {
        for (0..10) |r| {
            if (random.float(f32) < 0.3) continue;
            const at: Vec2 = .init(0.25 + @as(f32, @floatFromInt(c)) * 0.5, 0.25 + @as(f32, @floatFromInt(r)) * 0.5);
            const shape: physics.Shape = .{ .geometry = .{ .polygon = .box(0.25, 0.25) }, .user_data = tag };
            _ = try tree.addShape(try tree.createBody(.{ .type = .static, .position = at }), shape);
            _ = try walked.addShape(try walked.createBody(.{ .type = .kinematic, .position = at }), shape);
            tag += 1;
        }
    }

    for (0..300) |_| {
        const from: Vec2 = .init(random.float(f32) * 24 - 2, random.float(f32) * 8 - 3);
        const along: Vec2 = .init(random.float(f32) * 16 - 8, random.float(f32) * 16 - 8);
        const a = tree.castRay(from, along, .{});
        const b = walked.castRay(from, along, .{});
        try testing.expectEqual(a == null, b == null);
        if (a) |hit| {
            // The same fraction to the bit, at the same place and facing
            // the same way. (Two tiles meeting at exactly that point are a
            // tie either may win, so the tiles themselves are not compared.)
            try testing.expectEqual(hit.fraction, b.?.fraction);
            try testing.expect(hit.normal.eql(b.?.normal));
        }

        const point: Vec2 = .init(random.float(f32) * 20, random.float(f32) * 5);
        const in_tree = tree.overlapPoint(point);
        const in_walk = walked.overlapPoint(point);
        try testing.expectEqual(in_tree == null, in_walk == null);
        if (in_tree) |s| try testing.expectEqual(tree.shape(s).?.def.user_data, walked.shape(in_walk.?).?.def.user_data);
    }

    // A box finds the same tiles, however many.
    const Tags = struct {
        world: *World,
        seen: std.bit_set.IntegerBitSet(400) = .initEmpty(),
        fn visit(self: *@This(), s: physics.ShapeId) bool {
            self.seen.set(@intCast(self.world.shape(s).?.def.user_data));
            return true;
        }
    };
    for (0..50) |_| {
        const low: Vec2 = .init(random.float(f32) * 20, random.float(f32) * 5);
        const box: physics.Aabb = .{ .min = low, .max = low.add(.init(random.float(f32) * 4, random.float(f32) * 2)) };
        var from_tree: Tags = .{ .world = &tree };
        var from_walk: Tags = .{ .world = &walked };
        tree.overlapAabb(box, &from_tree, Tags.visit);
        walked.overlapAabb(box, &from_walk, Tags.visit);
        try testing.expect(from_tree.seen.eql(from_walk.seen));
    }
}

test "a level moved by hand is where it now is, and what slept on it wakes" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();
    const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
    _ = try world.addShape(floor, .box(5, 0.5));
    const crate = try world.createBody(.{ .position = .init(0, -0.5) });
    _ = try world.addShape(crate, .box(0.5, 0.5));
    for (0..240) |_| {
        try world.step(dt, &jobs);
        if (world.awakeCount() == 0) break;
    }
    try testing.expectEqual(@as(usize, 0), world.awakeCount());

    // Two metres down, by hand. The crate's sleeping contact with it is not
    // trusted to be true any more; it wakes, falls, and lands on it again.
    world.body(floor).?.setTransform(.init(0, 2.5), 0);
    try world.step(dt, &jobs);
    try testing.expect(world.body(crate).?.isAwake());
    try steps(&world, &jobs, 120);
    try testing.expectApproxEqAbs(@as(f32, 1.5), world.body(crate).?.position().y, 0.01);

    // Its box in the tree moved with it: a ray from above past the crate
    // finds its top at y = 2, and nothing is left where it was.
    const hit = world.castRay(.init(3, -5), .init(0, 10), .{}).?;
    try testing.expectApproxEqAbs(@as(f32, 2), hit.point.y, 1e-4);
    try testing.expect(world.overlapPoint(.init(3, 0.5)) == null);
}

test "a tile taken away or added between steps is gone or there from the next step" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();

    // A row of tiles with a ball asleep on the middle one, and a second row
    // two metres below.
    var row: [5]physics.BodyId = undefined;
    for (&row, 0..) |*t, i| {
        t.* = try world.createBody(.{ .type = .static, .position = .init(@as(f32, @floatFromInt(i)) - 2, 0.25) });
        _ = try world.addShape(t.*, .box(0.25, 0.25));
    }
    const below = try world.createBody(.{ .type = .static, .position = .init(0, 2.75) });
    _ = try world.addShape(below, .box(3, 0.25));
    const ball = try world.createBody(.{ .position = .init(0, -0.2) });
    _ = try world.addShape(ball, .circle(0.2));
    for (0..240) |_| {
        try world.step(dt, &jobs);
        if (world.awakeCount() == 0) break;
    }
    try testing.expectEqual(@as(usize, 0), world.awakeCount());

    // Out from under it: the contact ends, the ball wakes and drops to the
    // row below.
    world.destroyBody(row[2]);
    try steps(&world, &jobs, 90);
    try testing.expectApproxEqAbs(@as(f32, 2.3), world.body(ball).?.position().y, 0.01);

    // And a tile put in a falling crate's way stops it.
    const crate = try world.createBody(.{ .position = .init(1, -3) });
    _ = try world.addShape(crate, .box(0.2, 0.2));
    try steps(&world, &jobs, 10);
    const ledge = try world.createBody(.{ .type = .static, .position = .init(1, -1) });
    _ = try world.addShape(ledge, .box(0.25, 0.25));
    try steps(&world, &jobs, 90);
    try testing.expectApproxEqAbs(@as(f32, -1.45), world.body(crate).?.position().y, 0.01);
}

test "a step over seven thousand tiles pairs a crate with the few it is near" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();
    // A hundred and twenty columns sixty deep: every column's tiles overlap
    // each other along x, which is what a sweep alone was slowest at.
    try tileMap(&world, .body_per_tile, 120, 60);
    const crate = try world.createBody(.{ .position = .init(30, -0.2) });
    _ = try world.addShape(crate, .box(0.2, 0.2));
    try steps(&world, &jobs, 30);

    try testing.expect(world.pairs.items.len <= 3);
    try testing.expectApproxEqAbs(@as(f32, -0.2), world.body(crate).?.position().y, 0.01);
    // And it rests, so it sleeps, and a step then pairs nothing at all.
    try steps(&world, &jobs, 60);
    try testing.expectEqual(@as(usize, 0), world.awakeCount());
    try testing.expectEqual(@as(usize, 0), world.pairs.items.len);
}
