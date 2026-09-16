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
        // thick, because a thin wall is what a body squeezed by a pile goes
        // through - slowly, a little each step, which is not what the
        // continuous sweep is for: it catches what is fast.
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

/// The sensor events for `shape` this step: begun and ended.
fn sensorEvents(world: *const World, shape: physics.ShapeId) [2]usize {
    var got: [2]usize = .{ 0, 0 };
    for (world.beginEvents()) |e| {
        if (e.sensor and (e.shape_a.eql(shape) or e.shape_b.eql(shape))) got[0] += 1;
    }
    for (world.endEvents()) |e| {
        if (e.sensor and (e.shape_a.eql(shape) or e.shape_b.eql(shape))) got[1] += 1;
    }
    return got;
}

test "a trigger sees a kinematic body walk in, stand in it, and walk out" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        const trigger = try world.createBody(.{ .type = .static, .position = .init(0, 0) });
        const zone = try world.addShape(trigger, .{ .geometry = .{ .polygon = .box(1, 1) }, .sensor = true });
        const walker = try world.createBody(.{ .type = .kinematic, .position = .init(-3, 0), .linear_velocity = .init(3, 0) });
        _ = try world.addShape(walker, .box(0.25, 0.25));

        var begun: usize = 0;
        var ended: usize = 0;
        for (0..40) |_| {
            try world.step(dt, &jobs);
            const got = sensorEvents(&world, zone);
            begun += got[0];
            ended += got[1];
        }
        try testing.expectEqual(@as(usize, 1), begun);
        try testing.expectEqual(@as(usize, 0), ended);

        // Standing still in it: still in it, every step.
        world.body(walker).?.linear_velocity = .zero;
        for (0..60) |_| {
            try world.step(dt, &jobs);
            const got = sensorEvents(&world, zone);
            begun += got[0];
            ended += got[1];
        }
        try testing.expectEqual(@as(usize, 1), begun);
        try testing.expectEqual(@as(usize, 0), ended);
        try testing.expectEqual(@as(usize, 1), world.touchingCount());

        // And out the other side.
        world.body(walker).?.linear_velocity = .init(3, 0);
        for (0..60) |_| {
            try world.step(dt, &jobs);
            ended += sensorEvents(&world, zone)[1];
        }
        try testing.expectEqual(@as(usize, 1), ended);
    }
}

test "a sensor on a body that does not move sees the level, and an area another area" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        // An area set down on the floor, going nowhere.
        const area = try world.createBody(.{ .type = .kinematic, .position = .init(0, 0) });
        const watcher = try world.addShape(area, .{ .geometry = .{ .polygon = .box(1, 0.5) }, .sensor = true });
        // And another beside it, overlapping it.
        const other = try world.createBody(.{ .type = .kinematic, .position = .init(1.5, 0) });
        _ = try world.addShape(other, .{ .geometry = .{ .polygon = .box(1, 0.5) }, .sensor = true });

        try world.step(dt, &jobs);
        try testing.expectEqual(@as(usize, 2), sensorEvents(world, watcher)[0]);
        var ended: usize = 0;
        for (0..60) |_| {
            try world.step(dt, &jobs);
            ended += sensorEvents(world, watcher)[1];
        }
        try testing.expectEqual(@as(usize, 0), ended);
        // The floor with each area, and the areas with each other.
        try testing.expectEqual(@as(usize, 3), world.touchingCount());
    }
}

test "an area set down on a sleeping body sees it, and leaves it asleep" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        const crate = try world.createBody(.{ .position = .init(0, -0.5) });
        _ = try world.addShape(crate, .box(0.5, 0.5));
        try steps(world, &jobs, 120);
        try testing.expect(!world.body(crate).?.isAwake());

        // Far away first, then put down on the crate by hand.
        const area = try world.createBody(.{ .type = .kinematic, .position = .init(-20, -10) });
        const watcher = try world.addShape(area, .{ .geometry = .{ .polygon = .box(1, 1) }, .sensor = true });
        try world.step(dt, &jobs);
        world.body(area).?.setTransform(.init(0, -0.5), 0);
        try world.step(dt, &jobs);

        try testing.expect(sensorEvents(world, watcher)[0] >= 1);
        try testing.expect(!world.body(crate).?.isAwake());
    }
}

test "a hitbox that asks for nothing is seen by the hurtbox that asks for it" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .gravity = .zero });
        defer world.deinit();

        const hitboxes: u16 = 2;
        const sword = try world.createBody(.{ .type = .kinematic, .position = .init(0, 0) });
        const hitbox = try world.addShape(sword, .{ .geometry = .{ .polygon = .box(0.5, 0.5) }, .sensor = true, .filter = .{ .category = hitboxes, .mask = 0 } });
        const enemy = try world.createBody(.{ .type = .kinematic, .position = .init(0.5, 0) });
        _ = try world.addShape(enemy, .{ .geometry = .{ .polygon = .box(0.5, 0.5) }, .sensor = true, .filter = .{ .category = 0, .mask = hitboxes } });

        try world.step(dt, &jobs);
        try testing.expectEqual(@as(usize, 1), sensorEvents(&world, hitbox)[0]);

        // Two crates with the same bits and no sensor pass through each
        // other: a push takes both sides asking.
        const a = try world.createBody(.{ .position = .init(10, 0), .linear_velocity = .init(1, 0) });
        _ = try world.addShape(a, .{ .geometry = .{ .polygon = .box(0.5, 0.5) }, .filter = .{ .category = hitboxes, .mask = 0 } });
        const b = try world.createBody(.{ .position = .init(11.5, 0) });
        _ = try world.addShape(b, .{ .geometry = .{ .polygon = .box(0.5, 0.5) }, .filter = .{ .category = 0, .mask = hitboxes } });
        try steps(&world, &jobs, 60);
        try testing.expect(world.body(a).?.position().x > 10.9);
        try testing.expectApproxEqAbs(@as(f32, 11.5), world.body(b).?.position().x, 1e-4);
    }
}

test "two things that cannot move still never touch without a sensor" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        // A kinematic block through the floor, and another through it.
        const block = try world.createBody(.{ .type = .kinematic, .position = .init(0, 0.25) });
        _ = try world.addShape(block, .box(1, 1));
        const beside = try world.createBody(.{ .type = .kinematic, .position = .init(0.5, 0.25), .linear_velocity = .init(0.1, 0) });
        _ = try world.addShape(beside, .box(1, 1));

        try steps(world, &jobs, 10);
        try testing.expectEqual(@as(usize, 0), world.touchingCount());
        try testing.expectApproxEqAbs(@as(f32, 0.25), world.body(block).?.position().y, 1e-6);
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

// -------------------------------------------------------------------------
// Godot's rules: which bits touch, exceptions, one-way floors, mixing
// -------------------------------------------------------------------------

test "with Godot's filter rule one mask asking is enough to touch, with Box2D's it takes both" {
    for ([_]physics.FilterRule{ .both, .either }) |rule| {
        var jobs: Jobs = try .init(gpa, .{ .io = null });
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .filter_rule = rule });
        defer world.deinit();
        // A floor that asks for nothing, and a ball in the top category that
        // asks for the floor.
        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
        _ = try world.addShape(floor, .{ .geometry = .{ .polygon = .box(5, 0.5) }, .filter = .{ .category = 1, .mask = 0 } });
        const ball = try world.createBody(.{ .position = .init(0, -1) });
        _ = try world.addShape(ball, .{ .geometry = .{ .circle = .{ .radius = 0.25 } }, .filter = .{ .category = 1 << 31, .mask = 1 } });
        try steps(&world, &jobs, 120);
        const y = world.body(ball).?.position().y;
        switch (rule) {
            .both => try testing.expect(y > 5),
            .either => try testing.expectApproxEqAbs(@as(f32, -0.25), y, 0.02),
        }
    }
}

test "a collision exception lets a body through the floor until the last one is taken back" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try floored();
        var world = &scene.world;
        defer world.deinit();

        // Told twice, taken back once: still kept apart.
        const ghost = try world.createBody(.{ .position = .init(-2, -1) });
        _ = try world.addShape(ghost, .circle(0.25));
        try world.addCollisionException(ghost, scene.floor);
        try world.addCollisionException(scene.floor, ghost);
        world.removeCollisionException(ghost, scene.floor);
        try testing.expect(world.hasCollisionException(scene.floor, ghost));
        try testing.expectError(error.SameBody, world.addCollisionException(ghost, ghost));

        // Told once and taken back: lands.
        const lands = try world.createBody(.{ .position = .init(2, -1) });
        _ = try world.addShape(lands, .circle(0.25));
        try world.addCollisionException(lands, scene.floor);
        world.removeCollisionException(lands, scene.floor);
        try testing.expect(!world.hasCollisionException(lands, scene.floor));

        try steps(world, &jobs, 120);
        try testing.expect(world.body(ghost).?.position().y > 5);
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(lands).?.position().y, 0.02);

        // At rest on the floor, asleep by now, and then told: it falls.
        try world.addCollisionException(lands, scene.floor);
        try steps(world, &jobs, 60);
        try testing.expect(world.body(lands).?.position().y > 1);

        // A body made in the slot of a destroyed one has none of its
        // exceptions.
        world.destroyBody(ghost);
        const heir = try world.createBody(.{ .position = .init(-2, -1) });
        try testing.expectEqual(ghost.index, heir.index);
        _ = try world.addShape(heir, .circle(0.25));
        try testing.expect(!world.hasCollisionException(heir, scene.floor));
        try steps(world, &jobs, 120);
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(heir).?.position().y, 0.02);
    }
}

/// A floor at y = 0, ten metres wide and a metre thick, that holds only what
/// comes down onto it.
fn oneWayFloored(half_thickness: f32) !struct { world: World, floor: physics.BodyId } {
    var world: World = .init(gpa, .{});
    errdefer world.deinit();
    const floor = try world.createBody(.{ .type = .static, .position = .init(0, half_thickness) });
    _ = try world.addShape(floor, .{ .geometry = .{ .polygon = .box(5, half_thickness) }, .one_way = .{} });
    return .{ .world = world, .floor = floor };
}

test "a one-way floor holds what lands on it and lets through what comes up from below" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try oneWayFloored(0.5);
        var world = &scene.world;
        defer world.deinit();

        const lands = try world.createBody(.{ .position = .init(-2, -2) });
        _ = try world.addShape(lands, .circle(0.25));
        // Under the floor, thrown up hard enough to clear it by five metres.
        const jumps = try world.createBody(.{ .position = .init(2, 2), .linear_velocity = .init(0, -12) });
        _ = try world.addShape(jumps, .circle(0.25));

        var cleared = false;
        for (0..300) |_| {
            try world.step(dt, &jobs);
            if (world.body(jumps).?.position().y < -1) cleared = true;
        }
        try testing.expect(cleared);
        // Both on top: the one that came up through landed coming down.
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(lands).?.position().y, 0.02);
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(jumps).?.position().y, 0.02);

        // And it stays: ten seconds of the solver's nudges do not let it
        // through.
        try steps(world, &jobs, 600);
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(lands).?.position().y, 0.02);
    }
}

test "a one-way floor lets through what comes in from its side" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .gravity = .zero });
        defer world.deinit();
        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0) });
        _ = try world.addShape(floor, .{ .geometry = .{ .polygon = .box(1, 0.5) }, .one_way = .{} });
        const ball = try world.createBody(.{ .position = .init(-3, 0), .linear_velocity = .init(4, 0) });
        _ = try world.addShape(ball, .circle(0.25));
        try steps(&world, &jobs, 120);
        try testing.expect(world.body(ball).?.position().x > 3);
    }
}

test "a one-way wall turned on its side holds from one side only" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .gravity = .zero });
        defer world.deinit();
        // Turned a quarter: the arrow, +y in its own frame, points to -x in
        // the world, so it holds what comes from +x going -x.
        const wall = try world.createBody(.{ .type = .static, .position = .zero, .angle = std.math.pi / 2.0 });
        _ = try world.addShape(wall, .{ .geometry = .{ .polygon = .box(2, 0.25) }, .one_way = .{} });
        const from_right = try world.createBody(.{ .position = .init(3, 1), .linear_velocity = .init(-4, 0) });
        _ = try world.addShape(from_right, .circle(0.25));
        const from_left = try world.createBody(.{ .position = .init(-3, -1), .linear_velocity = .init(4, 0) });
        _ = try world.addShape(from_left, .circle(0.25));
        try steps(&world, &jobs, 120);
        // Stopped at the wall's right face, at x = 0.25, less its radius.
        try testing.expect(world.body(from_right).?.position().x > 0.4);
        try testing.expect(world.body(from_left).?.position().x > 3);
    }
}

test "the sweep stops a fast body coming down onto a thin one-way floor, not one going up through" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try oneWayFloored(0.02);
        var world = &scene.world;
        defer world.deinit();
        // Sixty metres a second is a metre a step: through a floor four
        // centimetres thick in one, without the sweep.
        const down = try world.createBody(.{ .position = .init(-2, -3), .linear_velocity = .init(0, 60) });
        _ = try world.addShape(down, .circle(0.1));
        const up = try world.createBody(.{ .position = .init(2, 3), .linear_velocity = .init(0, -60) });
        _ = try world.addShape(up, .circle(0.1));

        var up_cleared = false;
        for (0..12) |_| {
            try world.step(dt, &jobs);
            if (world.body(up).?.position().y < -1) up_cleared = true;
        }
        try testing.expect(up_cleared);
        try steps(world, &jobs, 180);
        try testing.expectApproxEqAbs(@as(f32, -0.1), world.body(down).?.position().y, 0.02);
    }
}

test "friction and restitution mix as the world says: Box2D's, or Godot's" {
    const Mixes = struct { friction: physics.Mix, restitution: physics.Mix };
    var slid: [2]f32 = undefined;
    var rose: [2]f32 = undefined;
    for ([_]Mixes{ .{ .friction = .geometric_mean, .restitution = .maximum }, .{ .friction = .minimum, .restitution = .sum_clamped } }, 0..) |mix, i| {
        var jobs: Jobs = try .init(gpa, .{ .io = null });
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .friction_mix = mix.friction, .restitution_mix = mix.restitution });
        defer world.deinit();
        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
        _ = try world.addShape(floor, .{ .geometry = .{ .polygon = .box(40, 0.5) }, .material = .{ .friction = 1, .restitution = 0.4 } });

        // A box sliding along the floor: sqrt(0.2) of friction stops it
        // sooner than 0.2.
        const slider = try world.createBody(.{ .position = .init(-30, -0.5), .linear_velocity = .init(5, 0) });
        _ = try world.addShape(slider, .{ .geometry = .{ .polygon = .box(0.5, 0.5) }, .material = .{ .friction = 0.2 } });
        // A ball dropped from three metres: 0.4 of the speed comes back, or
        // 0.3 + 0.4.
        const ball = try world.createBody(.{ .position = .init(30, -3), .linear_velocity = .zero });
        _ = try world.addShape(ball, .{ .geometry = .{ .circle = .{ .radius = 0.25 } }, .material = .{ .restitution = 0.3, .friction = 0 } });

        var hit = false;
        var highest: f32 = 0;
        for (0..240) |_| {
            try world.step(dt, &jobs);
            const b = world.body(ball).?;
            if (!hit and b.linear_velocity.y < 0 and b.position().y > -1) hit = true;
            if (hit and b.linear_velocity.y < 0) highest = @min(highest, b.position().y);
        }
        slid[i] = world.body(slider).?.position().x + 30;
        // Above where it rests, a radius up.
        rose[i] = -highest - 0.25;
        try testing.expect(hit);
    }
    // 0.2 against sqrt(0.2) of the grip: over twice as far, less what
    // starting to slide takes.
    try testing.expect(slid[1] > 1.8 * slid[0]);
    // 0.7 against 0.4 of the speed back: about three times as high.
    try testing.expect(rose[1] > 2 * rose[0]);
}

test "a one-way floor made after what lands on it holds it the same" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();
        // Made first, so the floor is the second of each pair.
        const lands = try world.createBody(.{ .position = .init(-2, -2) });
        _ = try world.addShape(lands, .circle(0.25));
        const jumps = try world.createBody(.{ .position = .init(2, 2), .linear_velocity = .init(0, -12) });
        _ = try world.addShape(jumps, .circle(0.25));
        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
        _ = try world.addShape(floor, .{ .geometry = .{ .polygon = .box(5, 0.5) }, .one_way = .{} });
        try steps(&world, &jobs, 300);
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(lands).?.position().y, 0.02);
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(jumps).?.position().y, 0.02);
    }
}

test "what a one-way floor decides at the first touch stands until the two part" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();
        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
        const floor_shape = try world.addShape(floor, .{ .geometry = .{ .polygon = .box(5, 0.5) }, .one_way = .{} });
        // Kept awake, so the pair is looked at every step.
        const held = try world.createBody(.{ .position = .init(-2, -1), .allow_sleep = false });
        _ = try world.addShape(held, .circle(0.25));
        try steps(&world, &jobs, 120);
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(held).?.position().y, 0.02);

        // Turned round, the floor holds only what comes up from below. What
        // stands on it was decided, and stays; what comes down onto it now
        // goes through.
        world.shape(floor_shape).?.def.one_way = .{ .direction = .init(0, -1) };
        const late = try world.createBody(.{ .position = .init(2, -1) });
        _ = try world.addShape(late, .circle(0.25));
        try steps(&world, &jobs, 120);
        try testing.expectApproxEqAbs(@as(f32, -0.25), world.body(held).?.position().y, 0.02);
        try testing.expect(world.body(late).?.position().y > 3);
    }
}

test "a body that rises into a one-way floor past its middle and falls back goes back down through" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var scene = try oneWayFloored(0.5);
        var world = &scene.world;
        defer world.deinit();
        // From under the floor, fast enough to rise past its middle, at
        // y = 0.5, and not out of its top.
        const ball = try world.createBody(.{ .position = .init(0, 2), .linear_velocity = .init(0, -5.8) });
        _ = try world.addShape(ball, .circle(0.25));
        var highest: f32 = 2;
        for (0..180) |_| {
            try world.step(dt, &jobs);
            highest = @min(highest, world.body(ball).?.position().y);
        }
        // Past the middle, where the floor's top is nearer and a new touch
        // would be held;
        try testing.expect(highest < 0.45);
        // but it was let in, so it is let out again below.
        try testing.expect(world.body(ball).?.position().y > 3);
    }
}

test "a body let into a one-way block from its side is still let through when it wakes there" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();
        // A one-way block a metre thick, and a plain ledge along its middle.
        const block = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
        _ = try world.addShape(block, .{ .geometry = .{ .polygon = .box(1, 0.5) }, .one_way = .{} });
        const ledge = try world.createBody(.{ .type = .static, .position = .init(0, 0.55) });
        _ = try world.addShape(ledge, .box(5, 0.05));
        // A crate on the ledge slides in at the block's side, stops in its top
        // half, and falls asleep there.
        const crate = try world.createBody(.{ .position = .init(-1.3, 0.25), .linear_velocity = .init(3, 0) });
        _ = try world.addShape(crate, .box(0.25, 0.25));
        try steps(&world, &jobs, 120);
        try testing.expect(!world.body(crate).?.awake);
        try testing.expect(world.body(crate).?.position().x > -0.75);
        try testing.expectApproxEqAbs(@as(f32, 0.25), world.body(crate).?.position().y, 0.02);

        // Woken, it is let through still: the block's top, a new touch would
        // hold it to, does not lift it out.
        world.body(crate).?.wake();
        try steps(&world, &jobs, 60);
        try testing.expectApproxEqAbs(@as(f32, 0.25), world.body(crate).?.position().y, 0.02);
    }
}

test "the sweep lets a fast body through a one-way floor's side, and past one it was let into, without a hitch" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{ .gravity = .zero });
        defer world.deinit();
        const thin = try world.createBody(.{ .type = .static, .position = .init(0, 0.1) });
        _ = try world.addShape(thin, .{ .geometry = .{ .polygon = .box(1, 0.1) }, .one_way = .{} });
        // Level with the thin floor, going right and a little down: through
        // its side in one step, and swept, since it moves half a metre a step.
        const across = try world.createBody(.{ .position = .init(-2, 0.1), .linear_velocity = .init(30, 3) });
        _ = try world.addShape(across, .circle(0.1));

        // A tall block, and a ball already in at its side, going on down and
        // across: its centre goes into the block during the step.
        const tall = try world.createBody(.{ .type = .static, .position = .init(10, 2) });
        _ = try world.addShape(tall, .{ .geometry = .{ .polygon = .box(1, 2) }, .one_way = .{} });
        const inside = try world.createBody(.{ .position = .init(8.8, 1), .linear_velocity = .init(12, 24) });
        _ = try world.addShape(inside, .circle(0.25));

        // A fast one-way plank of its own, level with a plain block and
        // going right into its side: the block is not on the side the
        // plank holds from.
        const block = try world.createBody(.{ .type = .static, .position = .init(0, -5) });
        _ = try world.addShape(block, .box(1, 0.1));
        const plank = try world.createBody(.{ .position = .init(-2, -5), .linear_velocity = .init(30, 0) });
        _ = try world.addShape(plank, .{ .geometry = .{ .polygon = .box(0.1, 0.1) }, .one_way = .{} });

        // Nothing slowed any of them for a moment on the way: a second of
        // each one's speed, with nothing pulling, to the thousandth.
        try steps(&world, &jobs, 60);
        try testing.expectApproxEqAbs(@as(f32, 28), world.body(across).?.position().x, 1e-3);
        try testing.expectApproxEqAbs(@as(f32, 20.8), world.body(inside).?.position().x, 1e-3);
        try testing.expectApproxEqAbs(@as(f32, 28), world.body(plank).?.position().x, 1e-3);
    }
}
