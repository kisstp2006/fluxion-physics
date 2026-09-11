// SPDX-License-Identifier: BSD-2-Clause

//! Fast bodies: nothing passes through the level between one step and the
//! next - and the sweep that sees to it is invisible where it is not needed.
//!
//! Each scene is one a step without continuous collision gets wrong: a ball
//! through a thin wall, a spinning rod through it, a ball dropped from a
//! height through a thin floor, a box through a kerb, a shot through a
//! plank. Each runs with the sweep off too, and has to fail then, so a scene
//! that passed only because it was too easy would show it. The other half is
//! what the sweep must leave alone: a box sliding fast over the seams of a
//! tiled floor keeps its speed, a bounce off a thin wall is a whole bounce,
//! and six hundred shots come out the same with workers and without.

const std = @import("std");
const testing = std.testing;

const physics = @import("root.zig");
const World = physics.World;
const Jobs = physics.Jobs;
const Vec2 = physics.Vec2;

const gpa = testing.allocator;

const modes = [_]Jobs.Options{
    .{ .io = testing.io, .workers = .auto },
    .{ .io = null },
};

const dt: f32 = 1.0 / 60.0;

fn steps(world: *World, jobs: *Jobs, n: usize) !void {
    for (0..n) |_| try world.step(dt, jobs);
}

/// A wall ten centimetres thick and ten metres tall, standing on y = 0 if
/// there is a floor, its near face at `x - 0.05`.
fn wall(world: *World, x: f32) !physics.BodyId {
    const w = try world.createBody(.{ .type = .static, .position = .init(x, -5) });
    _ = try world.addShape(w, .box(0.05, 5));
    return w;
}

/// A floor of half-metre tiles, a body each, its top at y = 0, from x = -2
/// for `width` metres: the way a tile map is built.
fn tiledFloor(world: *World, width: f32) !void {
    const n: usize = @intFromFloat(width / 0.5);
    for (0..n) |i| {
        const tile = try world.createBody(.{ .type = .static, .position = .init(-1.75 + @as(f32, @floatFromInt(i)) * 0.5, 0.25) });
        _ = try world.addShape(tile, .box(0.25, 0.25));
    }
}

test "a ball fired at a thin wall stops at it however fast, and goes through with the sweep off" {
    for (modes) |mode| {
        for ([_]f32{ 10, 30, 100, 300 }) |speed| {
            for ([_]bool{ true, false }) |sweep| {
                var jobs: Jobs = try .init(gpa, mode);
                defer jobs.deinit();
                var world: World = .init(gpa, .{ .gravity = .zero, .enable_continuous = sweep });
                defer world.deinit();
                _ = try wall(&world, 10);
                // Ten centimetres across: at ten metres a second it covers
                // more than the wall's thickness in one step.
                const ball = try world.createBody(.{ .position = .init(0, -2), .linear_velocity = .init(speed, 0) });
                _ = try world.addShape(ball, .circle(0.05));

                try steps(&world, &jobs, 120);
                const x = world.body(ball).?.position().x;
                if (sweep) {
                    // Up against the near face, a slop or so off it.
                    try testing.expect(x < 9.95 - 0.05 + 0.01);
                    try testing.expect(x > 9.95 - 0.05 - 0.02);
                } else {
                    try testing.expect(x > 10.05);
                }
            }
        }
    }
}

test "a rod thrown spinning at a thin wall stops at it" {
    for ([_]f32{ 10, 30, 100 }) |speed| {
        for ([_]bool{ true, false }) |sweep| {
            var jobs: Jobs = try .init(gpa, .{});
            defer jobs.deinit();
            var world: World = .init(gpa, .{ .gravity = .zero, .enable_continuous = sweep });
            defer world.deinit();
            _ = try wall(&world, 10);
            // Half a metre long, six centimetres thick, turning twenty
            // radians a second as it flies.
            const rod = try world.createBody(.{ .position = .init(0, -2), .linear_velocity = .init(speed, 0), .angular_velocity = 20 });
            _ = try world.addShape(rod, .box(0.25, 0.03));

            try steps(&world, &jobs, 120);
            const x = world.body(rod).?.position().x;
            if (sweep) try testing.expect(x < 9.95) else try testing.expect(x > 10.05);
        }
    }
}

test "a ball dropped from a hundred metres lands on a floor ten centimetres thick" {
    for ([_]bool{ true, false }) |sweep| {
        var jobs: Jobs = try .init(gpa, .{});
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .enable_continuous = sweep });
        defer world.deinit();
        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.05) });
        _ = try world.addShape(floor, .box(5, 0.05));
        const ball = try world.createBody(.{ .position = .init(0, -100) });
        _ = try world.addShape(ball, .circle(0.1));

        // Four and a half seconds down, at forty-four metres a second - three
        // quarters of a metre a step - and then a while to settle.
        try steps(&world, &jobs, 480);
        const y = world.body(ball).?.position().y;
        if (sweep) {
            try testing.expectApproxEqAbs(@as(f32, -0.1), y, 0.01);
        } else {
            try testing.expect(y > 0.1);
        }
    }
}

test "a box sliding fast over a tiled floor keeps its speed at every seam, and stops at a wall" {
    for ([_]bool{ true, false }) |sweep| {
        var jobs: Jobs = try .init(gpa, .{});
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .enable_continuous = sweep });
        defer world.deinit();
        try tiledFloor(&world, 40);
        _ = try wall(&world, 30);
        const box = try world.createBody(.{ .position = .init(-1, -0.2) });
        _ = try world.addShape(box, .{ .geometry = .{ .polygon = .box(0.2, 0.2) }, .material = .{ .friction = 0 } });
        try steps(&world, &jobs, 60);
        world.body(box).?.linear_velocity = .init(60, 0);

        // A metre a step, two seams a step: the corner of every tile ahead
        // is touched on the way, and not one of them may stop it.
        try steps(&world, &jobs, 18);
        const b = world.body(box).?;
        try testing.expectApproxEqAbs(@as(f32, 60), b.linear_velocity.x, 0.1);
        try testing.expectApproxEqAbs(@as(f32, -1 + 60 * 18 * dt), b.position().x, 0.05);

        try steps(&world, &jobs, 30);
        const x = world.body(box).?.position().x;
        if (sweep) try testing.expect(x < 29.95 - 0.2 + 0.01) else try testing.expect(x > 30.05);
    }
}

test "a box sliding into a kerb stops at it or trips over it, and is never found inside it" {
    const kerb_shape: physics.Shape = .box(0.25, 0.05);
    const box_shape: physics.Shape = .{ .geometry = .{ .polygon = .box(0.2, 0.2) }, .material = .{ .friction = 0 } };
    for ([_]f32{ 10, 30 }) |speed| {
        for ([_]bool{ true, false }) |sweep| {
            var jobs: Jobs = try .init(gpa, .{});
            defer jobs.deinit();
            var world: World = .init(gpa, .{ .enable_continuous = sweep });
            defer world.deinit();
            try tiledFloor(&world, 40);
            // Ten centimetres high, standing on the tiles at x = 20.
            const kerb = try world.createBody(.{ .type = .static, .position = .init(20, -0.05) });
            _ = try world.addShape(kerb, kerb_shape);
            const box = try world.createBody(.{ .position = .init(-1, -0.2) });
            _ = try world.addShape(box, box_shape);
            try steps(&world, &jobs, 60);
            world.body(box).?.linear_velocity = .init(speed, 0);

            // How deep the box is in the kerb at the end of any step. A
            // box forty centimetres tall is too tall to hop a kerb; hitting
            // it low, it may stop or tumble over it, as a real one would.
            var deepest: f32 = 0;
            for (0..240) |_| {
                try world.step(dt, &jobs);
                const s = physics.continuous.separation(&box_shape.geometry, world.body(box).?.transform, &kerb_shape.geometry, world.body(kerb).?.transform);
                deepest = @min(deepest, s.distance);
            }
            if (sweep) try testing.expect(deepest > -0.03) else try testing.expect(deepest < -0.05);
        }
    }
}

test "a bounce off a thin wall is a whole bounce" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .gravity = .zero });
        defer world.deinit();
        const w = try wall(&world, 10);
        const ball = try world.createBody(.{ .position = .init(0, -2), .linear_velocity = .init(60, 0) });
        _ = try world.addShape(ball, .{ .geometry = .{ .circle = .{ .radius = 0.1 } }, .material = .{ .restitution = 1 } });

        // Stopped short of the wall at the end of one step, it is bounced
        // off it the next, from the speed it came in at.
        var hit = false;
        for (0..60) |_| {
            try world.step(dt, &jobs);
            for (world.beginEvents()) |e| {
                if (e.body_a.eql(w) or e.body_b.eql(w)) hit = true;
            }
        }
        try testing.expect(hit);
        const b = world.body(ball).?;
        try testing.expectApproxEqAbs(@as(f32, -60), b.linear_velocity.x, 0.5);
        try testing.expect(b.position().x < 9.95);
    }
}

test "a bullet hits a plank that an ordinary fast body goes straight through" {
    for ([_]bool{ true, false }) |bullet| {
        var jobs: Jobs = try .init(gpa, .{});
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();
        const floor = try world.createBody(.{ .type = .static, .position = .init(5, 0.5) });
        _ = try world.addShape(floor, .box(10, 0.5));
        // A heavy plank, ten centimetres thick and two metres tall, left to
        // settle - and to fall asleep, which a bullet has to wake.
        const plank = try world.createBody(.{ .position = .init(5, -1) });
        _ = try world.addShape(plank, .{ .geometry = .{ .polygon = .box(0.05, 1) }, .material = .{ .density = 50 } });
        try steps(&world, &jobs, 120);

        // Three and a third metres a step, at a plank ten centimetres thick.
        // The level is swept for every fast body; other moving bodies only
        // for a bullet.
        const shot = try world.createBody(.{ .position = .init(0, -1.5), .linear_velocity = .init(200, 0), .gravity_scale = 0, .bullet = bullet });
        _ = try world.addShape(shot, .{ .geometry = .{ .circle = .{ .radius = 0.03 } }, .material = .{ .density = 20 } });
        var passed = false;
        for (0..60) |_| {
            try world.step(dt, &jobs);
            // Through is on the far side of the plank, seen from the plank:
            // hit high, it tips over, and the shot riding on its face can be
            // ahead of its middle without being through it.
            const p = world.body(plank).?;
            const seen = p.transform.unapply(world.body(shot).?.position());
            if (seen.x > 0.05 and @abs(seen.y) < 1) passed = true;
        }
        const p = world.body(plank).?;
        if (bullet) {
            try testing.expect(!passed);
            // And what it carried went into the plank, which was pushed.
            try testing.expect(p.position().x > 5.1);
        } else {
            try testing.expect(passed);
            try testing.expectApproxEqAbs(@as(f32, 5), p.position().x, 1e-3);
        }
    }
}

test "six hundred shots are swept the same to the bit with workers and without" {
    // Filled in mode by mode; freed whichever mode a failure stops at.
    var results: [modes.len]?[]f32 = @splat(null);
    defer for (results) |r| if (r) |slice| gpa.free(slice);
    for (modes, 0..) |mode, mi| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .gravity = .zero });
        defer world.deinit();
        const target = try world.createBody(.{ .type = .static, .position = .init(20, 0) });
        _ = try world.addShape(target, .box(0.05, 130));

        // More bodies than one job takes, so the sweep is cut across the
        // workers when there are any: rods and balls, each in a lane of its
        // own, spinning, at speeds from thirty to a hundred and fifty metres
        // a second - a tenth of them bullets - and free planks across some
        // of the lanes for the bullets to meet.
        var prng: std.Random.DefaultPrng = .init(3);
        const random = prng.random();
        var shots: [600]physics.BodyId = undefined;
        for (&shots, 0..) |*s, i| {
            const lane: f32 = @floatFromInt(i);
            s.* = try world.createBody(.{
                .position = .init(@as(f32, @floatFromInt(i % 10)) * 0.6, lane * 0.4 - 120),
                .linear_velocity = .init(30 + random.float(f32) * 120, 0),
                .angular_velocity = random.float(f32) * 30 - 15,
                .bullet = i % 10 == 0,
            });
            if (i % 2 == 0) {
                _ = try world.addShape(s.*, .circle(0.04 + random.float(f32) * 0.04));
            } else {
                _ = try world.addShape(s.*, .box(0.15, 0.02 + random.float(f32) * 0.02));
            }
        }
        for (0..12) |k| {
            const plank = try world.createBody(.{ .position = .init(15, @as(f32, @floatFromInt(k)) * 20 - 110) });
            _ = try world.addShape(plank, .box(0.05, 1.5));
        }

        try steps(&world, &jobs, 90);

        const out = try gpa.alloc(f32, shots.len * 3);
        results[mi] = out;
        for (shots, 0..) |s, i| {
            const b = world.body(s).?;
            out[i * 3] = b.position().x;
            out[i * 3 + 1] = b.position().y;
            out[i * 3 + 2] = b.angle;
            // Not one of them is past the wall.
            try testing.expect(b.position().x < 19.95);
        }
    }
    try testing.expectEqualSlices(f32, results[0].?, results[1].?);
}
