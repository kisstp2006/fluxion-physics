// SPDX-License-Identifier: BSD-2-Clause

//! Whole scenes held together by joints, stepped, and what should be true
//! of them afterwards.
//!
//! `joint.zig` checks one pass of each joint against numbers worked out by
//! hand; these check what a game sees. A pendulum keeps its length and
//! comes back up. A chain hangs together. A rope lets go when slack, a
//! spring bounces at the frequency it was given, a car drives. And the
//! same scene with a thousand joints is the same to the bit on every core
//! and on none.

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

test "a pendulum swings about its pin, keeps its length, and comes back up" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        const pivot = try world.createBody(.{ .type = .static });
        const bob = try world.createBody(.{ .position = .init(2, 0) });
        _ = try world.addShape(bob, .circle(0.1));
        _ = try world.createJoint(.{ .revolute = .{ .body_a = pivot, .body_b = bob, .anchor = .zero } });

        // Let go level with the pin: down through the bottom, up the far
        // side. Half a swing of a two metre pendulum from level is about
        // 1.7 seconds.
        var lowest: f32 = 0;
        var highest_far: f32 = 2;
        var stretch: f32 = 0;
        for (0..130) |_| {
            try world.step(dt, &jobs);
            const p = world.body(bob).?.center;
            stretch = @max(stretch, @abs(p.len() - 2));
            lowest = @max(lowest, p.y);
            if (p.x < 0) highest_far = @min(highest_far, p.y);
        }
        // A centimetre or so at the bottom of the swing, and not less: a
        // step moves the bob along the tangent, which on a circle is
        // outwards by `v^2 dt^2 / 2L`, and the solver takes a fifth of the
        // error back per step. At 6 m/s that settles near 1.3 cm - the
        // price of having no position pass, and invisible at 2 m.
        try testing.expect(stretch < 0.02);
        try testing.expectApproxEqAbs(@as(f32, 2), lowest, 0.02);
        // Back up to within a few centimetres of where it was let go: the
        // solver loses very little of the swing.
        try testing.expect(highest_far < 0.05);
    }
}

test "a thousand-link chain hangs together, and is the same on every core and on none" {
    const chains = 40;
    const links = 25;
    var results: [modes.len][]f32 = undefined;
    for (modes, 0..) |mode, mi| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        // Chains start level and fall. They never touch one another - one
        // negative group - so this is the joints alone, and with a thousand
        // of them each colour has hundreds and is solved on every core.
        const ceiling = try world.createBody(.{ .type = .static });
        var ends: [chains]physics.BodyId = undefined;
        var starts: [chains]Vec2 = undefined;
        for (&ends, &starts, 0..) |*end, *start, c| {
            const x0 = @as(f32, @floatFromInt(c)) * 0.5;
            const y0 = @as(f32, @floatFromInt(c)) * 0.25;
            var previous = ceiling;
            for (0..links) |l| {
                const left = x0 + @as(f32, @floatFromInt(l)) * 0.5;
                const link = try world.createBody(.{ .position = .init(left + 0.25, y0) });
                _ = try world.addShape(link, .{ .geometry = .{ .polygon = .box(0.25, 0.05) }, .filter = .{ .group = -1 } });
                _ = try world.createJoint(.{ .revolute = .{ .body_a = previous, .body_b = link, .anchor = .init(left, y0) } });
                previous = link;
            }
            end.* = previous;
            start.* = .init(x0, y0);
        }
        try testing.expectEqual(@as(usize, chains * links), world.jointCount());

        try steps(&world, &jobs, 180);

        // Every pin still holds its two ends together.
        var worst: f32 = 0;
        var it = world.jointIterator();
        while (it.next()) |entry| {
            const anchors = world.jointAnchors(entry.value);
            worst = @max(worst, anchors[0].dist(anchors[1]));
        }
        try testing.expect(worst < 0.02);
        // Three seconds is about half a swing of a chain this long: every
        // free end has been down through the bottom and is on its way up
        // the far side, still well below where it started.
        for (ends, starts) |end, pivot| {
            const p = world.body(end).?.position().sub(pivot);
            try testing.expect(p.x < -5);
            try testing.expect(p.y > 1);
        }
        try testing.expect(world.joint_colouring.colourCount() >= 2);

        const out = try gpa.alloc(f32, ends.len * 3);
        for (ends, 0..) |end, i| {
            const b = world.body(end).?;
            out[i * 3] = b.position().x;
            out[i * 3 + 1] = b.position().y;
            out[i * 3 + 2] = b.angle;
        }
        results[mi] = out;
    }
    defer for (results) |r| gpa.free(r);
    try testing.expectEqualSlices(f32, results[0], results[1]);
}

test "a rod holds its length, a rope only its longest, and a spring bounces at its frequency" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();

    const hook = try world.createBody(.{ .type = .static });

    // A rod, swinging from level like the pendulum: its length never moves.
    const on_rod = try world.createBody(.{ .position = .init(-3, 0) });
    _ = try world.addShape(on_rod, .{ .geometry = .{ .circle = .{ .radius = 0.1 } }, .filter = .{ .group = -1 } });
    _ = try world.createJoint(.{ .distance = .{ .body_a = hook, .body_b = on_rod, .anchor_a = .zero, .anchor_b = .init(-3, 0) } });

    // A rope two metres long, with its crate hanging a metre down: slack,
    // so the crate falls a metre and then hangs at two.
    const on_rope = try world.createBody(.{ .position = .init(0, 1) });
    _ = try world.addShape(on_rope, .{ .geometry = .{ .polygon = .box(0.1, 0.1) }, .filter = .{ .group = -1 } });
    _ = try world.createJoint(.{ .distance = .{
        .body_a = hook,
        .body_b = on_rope,
        .anchor_a = .zero,
        .anchor_b = .init(0, 1),
        .spring = .slack,
        .max_length = 2,
    } });

    var rod_stretch: f32 = 0;
    var rope_longest: f32 = 0;
    var rope_went_taut_at: usize = 0;
    for (0..180) |i| {
        try world.step(dt, &jobs);
        rod_stretch = @max(rod_stretch, @abs(world.body(on_rod).?.center.len() - 3));
        const length = world.body(on_rope).?.center.len();
        rope_longest = @max(rope_longest, length);
        if (rope_went_taut_at == 0 and length > 1.99) rope_went_taut_at = i;
    }
    // The pendulum's centimetre, for the same reason: see above.
    try testing.expect(rod_stretch < 0.02);
    try testing.expect(rope_longest < 2.02);
    // A free fall of one metre takes 0.45 seconds: the rope did nothing
    // until then.
    try testing.expect(rope_went_taut_at >= 25 and rope_went_taut_at <= 29);
    try testing.expectApproxEqAbs(@as(f32, 2), world.body(on_rope).?.center.len(), 0.01);

    // A spring of one hertz with nothing to damp it, in space: pulled half
    // a metre past its rest length and let go, it should cross the rest
    // length twice a second.
    var space: World = .init(gpa, .{ .gravity = .zero });
    defer space.deinit();
    const wall = try space.createBody(.{ .type = .static });
    const weight = try space.createBody(.{ .position = .init(1.5, 0) });
    _ = try space.addShape(weight, .circle(0.1));
    _ = try space.createJoint(.{ .distance = .{
        .body_a = wall,
        .body_b = weight,
        .anchor_a = .zero,
        .anchor_b = .init(1.5, 0),
        .length = 1,
        .spring = .{ .hertz = 1 },
    } });
    var crossings: [8]usize = undefined;
    var crossed: usize = 0;
    var was_long = true;
    for (0..300) |i| {
        try space.step(dt, &jobs);
        const long = space.body(weight).?.center.x > 1;
        if (long != was_long and crossed < crossings.len) {
            crossings[crossed] = i;
            crossed += 1;
        }
        was_long = long;
    }
    try testing.expect(crossed >= 6);
    // Half a period between crossings: thirty steps, give or take the
    // step's own rounding and the soft solver's slight slowing.
    for (1..crossed) |k| {
        const half = crossings[k] - crossings[k - 1];
        try testing.expect(half >= 28 and half <= 33);
    }
}

test "a hinge stops at its limits, a motor turns it, and a spring brings it back" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        const ground = try world.createBody(.{ .type = .static });

        // A bar hinged at its left end, level, allowed a quarter of a
        // radian each way: gravity pulls it down onto the lower... the
        // clockwise stop, which on this screen is down.
        const bar = try world.createBody(.{ .position = .init(1, 0) });
        _ = try world.addShape(bar, .box(1, 0.05));
        _ = try world.createJoint(.{ .revolute = .{
            .body_a = ground,
            .body_b = bar,
            .anchor = .zero,
            .limit = .{ .lower = -0.25, .upper = 0.25 },
        } });

        // A wheel on a motor: two radians a second, as much torque as it
        // likes.
        const wheel = try world.createBody(.{ .position = .init(10, 0), .gravity_scale = 0 });
        _ = try world.addShape(wheel, .circle(0.5));
        const motor = try world.createJoint(.{ .revolute = .{
            .body_a = ground,
            .body_b = wheel,
            .anchor = .init(10, 0),
            .motor = .{ .speed = 2, .max_torque = 1000 },
        } });

        // A flap on a spring, knocked sideways: it swings back towards
        // where it was made, and settles there.
        const flap = try world.createBody(.{ .position = .init(20, 0), .gravity_scale = 0, .angular_velocity = 5 });
        _ = try world.addShape(flap, .box(0.5, 0.05));
        _ = try world.createJoint(.{ .revolute = .{
            .body_a = ground,
            .body_b = flap,
            .anchor = .init(20, 0),
            .spring = .{ .hertz = 2, .damping_ratio = 0.5 },
        } });

        var widest: f32 = 0;
        for (0..120) |_| {
            try world.step(dt, &jobs);
            widest = @max(widest, @abs(world.body(bar).?.angle));
        }
        try testing.expect(widest < 0.27);
        try testing.expectApproxEqAbs(@as(f32, 0.25), world.body(bar).?.angle, 0.01);
        try testing.expectApproxEqAbs(@as(f32, 2), world.body(wheel).?.angular_velocity, 1e-3);
        try testing.expect(@abs(world.body(flap).?.angle) < 0.05);

        // The motor is a field: turned round, the wheel follows.
        world.joint(motor).?.kind.revolute.motor.?.speed = -1;
        try steps(&world, &jobs, 10);
        try testing.expectApproxEqAbs(@as(f32, -1), world.body(wheel).?.angular_velocity, 1e-3);
        // What it took is a torque, and none of it is needed to hold the
        // speed once it is there.
        try testing.expect(@abs(world.jointReaction(motor).?.torque) < 1e-2);
    }
}

test "a slider keeps its body on the line and square, and its limit and motor work" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        const ground = try world.createBody(.{ .type = .static });

        // A lift shaft: straight down from the origin, at most three metres.
        // The box is thrown sideways and set spinning, and neither sticks.
        const car = try world.createBody(.{ .linear_velocity = .init(4, 0), .angular_velocity = 3 });
        _ = try world.addShape(car, .box(0.5, 0.5));
        _ = try world.createJoint(.{ .prismatic = .{
            .body_a = ground,
            .body_b = car,
            .anchor = .zero,
            .axis = .init(0, 1),
            .limit = .{ .lower = 0, .upper = 3 },
        } });

        // A piston along +x, driven by a motor at one metre a second, in a
        // world where nothing else would move it.
        const piston = try world.createBody(.{ .position = .init(10, 0), .gravity_scale = 0 });
        _ = try world.addShape(piston, .box(0.2, 0.2));
        _ = try world.createJoint(.{ .prismatic = .{
            .body_a = ground,
            .body_b = piston,
            .anchor = .init(10, 0),
            .axis = .init(1, 0),
            .motor = .{ .speed = 1, .max_force = 500 },
        } });

        try steps(&world, &jobs, 120);
        const c = world.body(car).?;
        try testing.expect(@abs(c.position().x) < 0.01);
        try testing.expect(@abs(c.angle) < 0.01);
        try testing.expectApproxEqAbs(@as(f32, 3), c.position().y, 0.02);
        try testing.expectApproxEqAbs(@as(f32, 12), world.body(piston).?.position().x, 0.02);
        try testing.expectApproxEqAbs(@as(f32, 0), world.body(piston).?.position().y, 1e-3);
    }
}

test "a weld holds two bodies as one through a fall onto the floor" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
        _ = try world.addShape(floor, .box(10, 0.5));

        // An L: a post and a beam welded at its foot, dropped tilted.
        const post = try world.createBody(.{ .position = .init(0, -3), .angle = 0.3 });
        _ = try world.addShape(post, .box(0.1, 0.6));
        const beam = try world.createBody(.{ .position = .init(0, -3), .angle = 0.3 });
        _ = try world.addShape(beam, .{ .geometry = .{ .polygon = .offsetBox(0.6, 0.1, .init(0.5, 0.5), 0) } });
        _ = try world.createJoint(.{ .weld = .{ .body_a = post, .body_b = beam, .anchor = world.body(post).?.transform.apply(.init(0, 0.5)) } });

        const before = world.body(beam).?.transform.p.sub(world.body(post).?.transform.p);
        try steps(&world, &jobs, 180);
        const p = world.body(post).?;
        const b = world.body(beam).?;
        try testing.expectApproxEqAbs(p.angle, b.angle, 0.01);
        try testing.expect(b.transform.p.sub(p.transform.p).dist(before) < 0.02);
        // And it came to rest on the floor, not through it.
        try testing.expect(p.position().y < 0 and p.position().y > -1.5);
        try testing.expect(p.linear_velocity.len() < 0.05);
    }
}

test "a mouse joint drags a body to the pointer, no harder than it is allowed" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{ .units_per_metre = 100 });
    defer world.deinit();

    const floor = try world.createBody(.{ .type = .static, .position = .init(0, 50) });
    _ = try world.addShape(floor, .box(1000, 50));
    const crate = try world.createBody(.{ .position = .init(0, -25) });
    _ = try world.addShape(crate, .box(25, 25));
    try steps(&world, &jobs, 30);

    // Grab it by a corner and pull it up and to the right.
    const grabbed_at = world.body(crate).?.transform.apply(.init(20, -20));
    const drag = try world.createJoint(.{ .mouse = .{ .body = crate, .target = grabbed_at } });
    world.joint(drag).?.kind.mouse.target = .init(300, -200);
    try steps(&world, &jobs, 120);
    const anchors = world.jointAnchors(world.joint(drag).?);
    try testing.expect(anchors[0].dist(anchors[1]) < 5);

    // Let go and it falls back to the floor.
    world.destroyJoint(drag);
    try testing.expect(world.joint(drag) == null);
    try steps(&world, &jobs, 120);
    try testing.expect(world.body(crate).?.position().y > -40);

    // A weak pull cannot lift it at all: less than its weight.
    const weight = world.body(crate).?.mass * 9.81 * 100;
    const weak = try world.createJoint(.{ .mouse = .{
        .body = crate,
        .target = world.body(crate).?.center,
        .max_force = 0.5 * weight,
    } });
    world.joint(weak).?.kind.mouse.target = world.body(crate).?.center.add(.init(0, -300));
    const resting = world.body(crate).?.center.y;
    try steps(&world, &jobs, 60);
    try testing.expectApproxEqAbs(resting, world.body(crate).?.center.y, 1);
}

test "a car on sprung wheels drives along the floor" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();

        const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
        _ = try world.addShape(floor, .{ .geometry = .{ .polygon = .box(100, 0.5) }, .material = .{ .friction = 0.9 } });

        const chassis = try world.createBody(.{ .position = .init(0, -1) });
        _ = try world.addShape(chassis, .box(1.2, 0.25));
        var wheels: [2]physics.BodyId = undefined;
        var axles: [2]physics.JointId = undefined;
        for (&wheels, &axles, [_]f32{ -0.8, 0.8 }) |*wheel, *axle, x| {
            wheel.* = try world.createBody(.{ .position = .init(x, -0.6) });
            _ = try world.addShape(wheel.*, .{ .geometry = .{ .circle = .{ .radius = 0.35 } }, .material = .{ .friction = 0.9 } });
            axle.* = try world.createJoint(.{ .wheel = .{
                .body_a = chassis,
                .body_b = wheel.*,
                .anchor = .init(x, -0.6),
                .axis = .init(0, -1),
                .motor = .{ .speed = 0, .max_torque = 40 },
            } });
        }

        // Settle on the suspension, then drive: positive is towards +x.
        try steps(&world, &jobs, 60);
        const parked = world.body(chassis).?.position();
        try testing.expect(@abs(parked.x) < 0.01);
        for (axles) |axle| world.joint(axle).?.kind.wheel.motor.?.speed = 10;
        try steps(&world, &jobs, 180);

        const c = world.body(chassis).?;
        // Ten radians a second on a 0.35 wheel is 3.5 m/s, less slip and
        // the time to get going: well over five metres in three seconds.
        try testing.expect(c.position().x > 5);
        try testing.expect(@abs(c.angle) < 0.2);
        // Riding on its wheels, not on its belly.
        try testing.expect(c.position().y < -0.5);
        for (wheels) |wheel| try testing.expect(world.body(wheel).?.center.y > -0.5);
    }
}

test "bodies held by a joint do not collide unless it says they should" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{ .gravity = .zero });
    defer world.deinit();

    // Two pairs of overlapping boxes, each pair pinned where they overlap.
    // One joint lets its pair collide, the other does not.
    var pairs: [2][2]physics.BodyId = undefined;
    for (&pairs, [_]bool{ false, true }, 0..) |*pair, collide, i| {
        const x = @as(f32, @floatFromInt(i)) * 10;
        pair[0] = try world.createBody(.{ .position = .init(x, 0) });
        _ = try world.addShape(pair[0], .box(1, 0.2));
        pair[1] = try world.createBody(.{ .position = .init(x + 1, 0) });
        _ = try world.addShape(pair[1], .box(1, 0.2));
        _ = try world.createJoint(.{ .revolute = .{
            .body_a = pair[0],
            .body_b = pair[1],
            .anchor = .init(x + 0.5, 0),
            .collide_connected = collide,
        } });
    }
    try world.step(dt, &jobs);
    try testing.expectEqual(@as(usize, 1), world.touchingCount());
    try testing.expectEqual(@as(usize, 1), world.no_collide.count());

    // The joint that forbade it goes, and the pair it held now collide.
    var it = world.jointIterator();
    const first = it.next().?.handle;
    world.destroyJoint(first);
    try testing.expectEqual(@as(usize, 0), world.no_collide.count());
    try world.step(dt, &jobs);
    try testing.expectEqual(@as(usize, 2), world.touchingCount());
}

test "destroying a body takes its joints with it, and bad definitions are refused" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();

    const a = try world.createBody(.{ .type = .static });
    const b = try world.createBody(.{ .position = .init(1, 0) });
    _ = try world.addShape(b, .circle(0.2));
    const c = try world.createBody(.{ .position = .init(2, 0) });
    _ = try world.addShape(c, .circle(0.2));
    const ab = try world.createJoint(.{ .revolute = .{ .body_a = a, .body_b = b, .anchor = .init(0.5, 0) } });
    const bc = try world.createJoint(.{ .distance = .{ .body_a = b, .body_b = c, .anchor_a = .init(1, 0), .anchor_b = .init(2, 0) } });
    try testing.expectEqual(@as(usize, 2), world.jointCount());

    world.destroyBody(b);
    try testing.expect(world.joint(ab) == null);
    try testing.expect(world.joint(bc) == null);
    try testing.expectEqual(@as(usize, 0), world.jointCount());
    try testing.expectEqual(@as(usize, 0), world.no_collide.count());
    try steps(&world, &jobs, 10);

    try testing.expectError(error.NoSuchBody, world.createJoint(.{ .revolute = .{ .body_a = a, .body_b = b, .anchor = .zero } }));
    try testing.expectError(error.SameBody, world.createJoint(.{ .weld = .{ .body_a = c, .body_b = c, .anchor = .zero } }));
    try testing.expectError(error.NoSuchBody, world.addShape(b, .circle(1)));
}
