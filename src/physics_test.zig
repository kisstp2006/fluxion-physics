// SPDX-License-Identifier: BSD-2-Clause

//! Whole scenes, stepped, and what should be true of them afterwards.
//!
//! The unit tests beside each module check the arithmetic; these check
//! the physics. A ball dropped on the floor should end up on the floor and
//! stay there. A stack should stand. A ball with restitution should come
//! back up. The same scene on every core and on none should be the same
//! scene, to the bit.
//!
//! Every scene here runs twice, with workers and without - the browser's
//! way - because the parallel solver's one promise is that the answer does
//! not depend on which it is.

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

/// A floor at y = 0, ten metres wide, and the world it is in.
fn floored() !struct { world: World, floor: physics.BodyId } {
    var world: World = .init(gpa, .{});
    errdefer world.deinit();
    const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
    _ = try world.addShape(floor, .box(5, 0.5));
    return .{ .world = world, .floor = floor };
}

test "a ball dropped on the floor lands on it and stays" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        // Two metres up. `+y` is down, so up is negative.
        const ball = try world.createBody(.{ .position = .init(0, -2) });
        _ = try world.addShape(ball, .circle(0.25));

        try steps(world, &jobs, 180);
        const b = world.body(ball).?;
        // Resting on y = 0 with radius 0.25, a slop's worth inside.
        try testing.expectApproxEqAbs(@as(f32, -0.25), b.position().y, 0.01);
        try testing.expect(@abs(b.linear_velocity.y) < 0.02);
        try testing.expect(@abs(b.position().x) < 1e-3);
        try testing.expectEqual(@as(usize, 1), world.touchingCount());

        // Another three seconds and it has not moved.
        const before = b.position();
        try steps(world, &jobs, 180);
        try testing.expect(world.body(ball).?.position().approxEql(before) or
            world.body(ball).?.position().distSq(before) < 1e-6);
    }
}

test "a stack of crates stands" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        const count = 6;
        var crates: [count]physics.BodyId = undefined;
        for (&crates, 0..) |*c, i| {
            const y = -0.5 - @as(f32, @floatFromInt(i)) * 1.0;
            c.* = try world.createBody(.{ .position = .init(0, y) });
            _ = try world.addShape(c.*, .box(0.5, 0.5));
        }

        try steps(world, &jobs, 300);
        for (crates, 0..) |c, i| {
            const b = world.body(c).?;
            const expected_y = -0.5 - @as(f32, @floatFromInt(i)) * 1.0;
            // Each crate sinks a little into the one below; a few
            // centimetres over six is the Baumgarte price.
            try testing.expectApproxEqAbs(expected_y, b.position().y, 0.05);
            try testing.expect(@abs(b.position().x) < 0.02);
            try testing.expect(@abs(b.angle) < 0.02);
        }
        // The bottom crate touches the floor and the one above; every
        // other touches two crates; the top touches one.
        try testing.expectEqual(@as(usize, count), world.touchingCount());
    }
}

test "a tower of twenty-five stands straight, and comes to rest" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        var crates: [25]physics.BodyId = undefined;
        for (&crates, 0..) |*c, i| {
            c.* = try world.createBody(.{ .position = .init(0, -0.5 - @as(f32, @floatFromInt(i))) });
            _ = try world.addShape(c.*, .box(0.5, 0.5));
        }
        // Ten crates high was already too many before the corners of a
        // box were solved together: it leaned, and fell. Now twenty-five
        // stand, settle a few millimetres into each other, and sleep.
        var worst_lean: f32 = 0;
        var asleep_at: ?usize = null;
        for (0..300) |s| {
            try world.step(dt, &jobs);
            worst_lean = @max(worst_lean, @abs(world.body(crates[24]).?.position().x));
            if (asleep_at == null and world.awakeCount() == 0) asleep_at = s;
        }
        try testing.expect(worst_lean < 0.05);
        try testing.expect(asleep_at != null and asleep_at.? < 180);
        try testing.expectApproxEqAbs(@as(f32, -24.5), world.body(crates[24]).?.position().y, 0.5);
    }
}

test "two crates made inside each other slide apart, and stop" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{ .gravity = .zero });
    defer world.deinit();

    // Half of one inside the other. Pushed apart, they must not keep the
    // push: a crate a level designer placed a little inside a wall should
    // slide out of it, not be fired across the room.
    const a = try world.createBody(.{});
    _ = try world.addShape(a, .box(0.5, 0.5));
    const b = try world.createBody(.{ .position = .init(0.5, 0) });
    _ = try world.addShape(b, .box(0.5, 0.5));
    try steps(&world, &jobs, 120);
    const apart = world.body(b).?.position().x - world.body(a).?.position().x;
    try testing.expectApproxEqAbs(@as(f32, 1), apart, 0.03);
    try testing.expect(world.body(b).?.linear_velocity.sub(world.body(a).?.linear_velocity).len() < 0.01);
}

test "restitution brings a ball back up, and none leaves it down" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        const bouncy = try world.createBody(.{ .position = .init(-2, -3) });
        _ = try world.addShape(bouncy, .{ .geometry = .{ .circle = .{ .radius = 0.25 } }, .material = .{ .restitution = 0.9 } });
        const dead = try world.createBody(.{ .position = .init(2, -3) });
        _ = try world.addShape(dead, .circle(0.25));

        // Step until the bouncy one has hit and risen again, tracking its
        // highest point after the first contact.
        var hit = false;
        var highest: f32 = 0;
        for (0..240) |_| {
            try world.step(dt, &jobs);
            const b = world.body(bouncy).?;
            if (!hit and world.beginEvents().len != 0) hit = true;
            if (hit and b.linear_velocity.y < 0) highest = @min(highest, b.position().y);
        }
        try testing.expect(hit);
        // Dropped from 2.75 above the surface; came back more than half way.
        try testing.expect(highest < -1.5);
        // The dead one is on the floor.
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(dead).?.position().y, 0.02);
    }
}

test "the same scene is the same to the bit with workers and without" {
    var results: [modes.len][]f32 = undefined;
    for (modes, 0..) |mode, mi| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        // A bowl of static boxes and a heap of mixed bodies in it. The
        // walls lean outwards, so the bowl opens upwards; and they are
        // thick, because a thin wall is what a body squeezed by a pile
        // goes through - there is no continuous collision here, and the
        // README says so.
        const left = try world.createBody(.{ .type = .static, .position = .init(-4, 0), .angle = -0.5 });
        _ = try world.addShape(left, .box(0.5, 4));
        const right = try world.createBody(.{ .type = .static, .position = .init(4, 0), .angle = 0.5 });
        _ = try world.addShape(right, .box(0.5, 4));
        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 3.25) });
        _ = try world.addShape(floor, .box(7, 0.5));

        var prng: std.Random.DefaultPrng = .init(7);
        const random = prng.random();
        var bodies: [80]physics.BodyId = undefined;
        for (&bodies, 0..) |*b, i| {
            b.* = try world.createBody(.{
                .position = .init(random.float(f32) * 4 - 2, -@as(f32, @floatFromInt(i)) * 0.4 - 1),
                .angle = random.float(f32) * 6,
            });
            if (i % 3 == 0) {
                _ = try world.addShape(b.*, .circle(0.15 + random.float(f32) * 0.2));
            } else {
                _ = try world.addShape(b.*, .box(0.15 + random.float(f32) * 0.2, 0.15 + random.float(f32) * 0.2));
            }
        }

        try steps(&world, &jobs, 240);

        const out = try gpa.alloc(f32, bodies.len * 3);
        for (bodies, 0..) |b, i| {
            const body = world.body(b).?;
            out[i * 3] = body.position().x;
            out[i * 3 + 1] = body.position().y;
            out[i * 3 + 2] = body.angle;
        }
        results[mi] = out;

        // Everything ended up in the bowl.
        for (bodies) |b| {
            const p = world.body(b).?.position();
            try testing.expect(p.y < 3 and p.y > -8);
            try testing.expect(@abs(p.x) < 6);
        }
        // With eighty bodies in a heap the colouring is doing real work.
        try testing.expect(world.colourCount() >= 2);
    }
    defer for (results) |r| gpa.free(r);
    try testing.expectEqualSlices(f32, results[0], results[1]);
}

test "filters keep shapes apart and sensors report without pushing" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        // A ghost that falls through the floor because its mask excludes it.
        const ghost = try world.createBody(.{ .position = .init(-2, -1) });
        _ = try world.addShape(ghost, .{ .geometry = .{ .circle = .{ .radius = 0.2 } }, .filter = .{ .category = 2, .mask = 2 } });

        // A sensor plate just above the floor, and a ball dropped on it.
        const plate = try world.createBody(.{ .type = .static, .position = .init(2, -0.5) });
        const plate_shape = try world.addShape(plate, .{ .geometry = .{ .polygon = .box(1, 0.5) }, .sensor = true });
        const ball = try world.createBody(.{ .position = .init(2, -3) });
        _ = try world.addShape(ball, .circle(0.2));

        var entered = false;
        var left = false;
        var entered_at: u64 = 0;
        for (0..240) |_| {
            try world.step(dt, &jobs);
            for (world.beginEvents()) |e| {
                if (e.sensor and (e.shape_a.eql(plate_shape) or e.shape_b.eql(plate_shape))) {
                    entered = true;
                    entered_at = world.step_count;
                }
            }
            for (world.endEvents()) |e| {
                if (e.sensor) left = true;
            }
        }
        try testing.expect(entered);
        try testing.expect(!left);
        try testing.expect(entered_at > 1);
        // The ball is on the floor, not on the plate.
        try testing.expectApproxEqAbs(@as(f32, -0.2), world.body(ball).?.position().y, 0.02);
        // The ghost went through.
        try testing.expect(world.body(ghost).?.position().y > 5);

        // Destroying the ball ends its contacts.
        world.destroyBody(ball);
        try world.step(dt, &jobs);
        var ended: usize = 0;
        for (world.endEvents()) |e| {
            if (e.body_a.eql(ball) or e.body_b.eql(ball)) ended += 1;
        }
        try testing.expectEqual(@as(usize, 2), ended);
    }
}

test "a kinematic platform carries what stands on it" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        const platform = try world.createBody(.{ .type = .kinematic, .position = .init(0, 0), .linear_velocity = .init(1, 0) });
        _ = try world.addShape(platform, .box(2, 0.25));
        const crate = try world.createBody(.{ .position = .init(0, -0.75) });
        _ = try world.addShape(crate, .box(0.25, 0.5));

        try steps(&world, &jobs, 120);
        // Two seconds at one metre a second. The crate came along, minus
        // a little slip while friction caught up.
        try testing.expectApproxEqAbs(@as(f32, 2), world.body(platform).?.position().x, 1e-3);
        try testing.expectApproxEqAbs(@as(f32, 2), world.body(crate).?.position().x, 0.15);
        try testing.expect(@abs(world.body(crate).?.angle) < 0.05);
        // The platform was never pushed down by the crate.
        try testing.expectApproxEqAbs(@as(f32, 0), world.body(platform).?.position().y, 1e-6);
    }
}

test "gravity scale, damping, and a body that will not turn" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();

    // Each on its own row, so none of them meets another.
    const floating = try world.createBody(.{ .gravity_scale = 0, .linear_velocity = .init(1, 0) });
    _ = try world.addShape(floating, .circle(0.1));
    const damped = try world.createBody(.{ .position = .init(0, 5), .gravity_scale = 0, .linear_velocity = .init(1, 0), .linear_damping = 2 });
    _ = try world.addShape(damped, .circle(0.1));
    const upright = try world.createBody(.{ .position = .init(0, 10), .fixed_rotation = true, .gravity_scale = 0, .angular_velocity = 3 });
    _ = try world.addShape(upright, .box(0.1, 0.4));

    try steps(&world, &jobs, 60);
    try testing.expectApproxEqAbs(@as(f32, 1), world.body(floating).?.position().x, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0), world.body(floating).?.position().y, 1e-5);
    try testing.expect(world.body(damped).?.position().x < 0.6);
    try testing.expect(world.body(damped).?.linear_velocity.x < 0.2);
    try testing.expectEqual(@as(f32, 0), world.body(upright).?.inv_inertia);
}

test "rays and points find what is there" {
    var world: World = .init(gpa, .{});
    defer world.deinit();

    const near = try world.createBody(.{ .type = .static, .position = .init(3, 0) });
    const near_shape = try world.addShape(near, .box(0.5, 0.5));
    const far = try world.createBody(.{ .type = .static, .position = .init(6, 0) });
    const far_shape = try world.addShape(far, .{ .geometry = .{ .circle = .{ .radius = 1 } }, .filter = .{ .category = 2 } });

    // Along +x from the origin: the box first, at its near face.
    const hit = world.castRay(.zero, .init(10, 0), .{}).?;
    try testing.expect(hit.shape.eql(near_shape));
    try testing.expectApproxEqAbs(@as(f32, 0.25), hit.fraction, 1e-5);
    try testing.expect(hit.point.approxEql(.init(2.5, 0)));
    try testing.expect(hit.normal.approxEql(.init(-1, 0)));

    // A ray that only sees category 2 skips the box and finds the circle.
    const through = world.castRay(.zero, .init(10, 0), .{ .mask = 2 }).?;
    try testing.expect(through.shape.eql(far_shape));
    try testing.expectApproxEqAbs(@as(f32, 0.5), through.fraction, 1e-5);
    try testing.expect(through.normal.approxEql(.init(-1, 0)));

    // Too short to reach anything, and pointing away.
    try testing.expect(world.castRay(.zero, .init(2, 0), .{}) == null);
    try testing.expect(world.castRay(.zero, .init(-10, 0), .{}) == null);

    // A ray starting inside the box has no entry face and misses it.
    const from_inside = world.castRay(.init(3, 0), .init(10, 0), .{}).?;
    try testing.expect(from_inside.shape.eql(far_shape));

    try testing.expect(world.overlapPoint(.init(3.2, 0.2)).?.eql(near_shape));
    try testing.expect(world.overlapPoint(.init(6.5, 0.5)).?.eql(far_shape));
    try testing.expect(world.overlapPoint(.init(4.5, 0)) == null);

    const Count = struct {
        fn visit(n: *usize, _: physics.ShapeId) bool {
            n.* += 1;
            return true;
        }
    };
    var n: usize = 0;
    world.overlapAabb(.{ .min = .init(2, -1), .max = .init(8, 1) }, &n, Count.visit);
    try testing.expectEqual(@as(usize, 2), n);
    n = 0;
    world.overlapAabb(.{ .min = .init(4, -1), .max = .init(4.5, 1) }, &n, Count.visit);
    try testing.expectEqual(@as(usize, 0), n);
}

test "a step with nothing in it, and a step of no time, do nothing" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();
    try world.step(dt, &jobs);
    try testing.expectEqual(@as(u64, 1), world.step_count);
    const b = try world.createBody(.{});
    try world.step(0, &jobs);
    try testing.expect(world.body(b).?.position().eql(.zero));
    try testing.expectEqual(@as(u64, 1), world.step_count);
}
