// SPDX-License-Identifier: BSD-2-Clause

//! Sleeping: what rests stops costing anything, and everything that should
//! wake it does.
//!
//! The rule sleeping has to keep is that a game cannot tell it is there.
//! A sleeping stack must look exactly like a resting one - no end events,
//! nothing moving - and must come back to life the moment anything could
//! have changed its mind: a push, a hit, a moved platform, its support
//! destroyed, a motor switched on. Each of those is a scene here.

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

/// Step until nothing is awake, or give up after `limit` steps.
fn untilAsleep(world: *World, jobs: *Jobs, limit: usize) !bool {
    for (0..limit) |_| {
        try world.step(dt, jobs);
        if (world.awakeCount() == 0) return true;
    }
    return false;
}

/// A floor with its top at y = 0, and a stack of `n` unit crates on it.
fn stack(world: *World, crates: []physics.BodyId) !physics.BodyId {
    const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
    _ = try world.addShape(floor, .box(5, 0.5));
    for (crates, 0..) |*c, i| {
        c.* = try world.createBody(.{ .position = .init(0, -0.5 - @as(f32, @floatFromInt(i))) });
        _ = try world.addShape(c.*, .box(0.5, 0.5));
    }
    return floor;
}

test "a stack at rest falls asleep, stays put, and costs nothing" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();
        var crates: [4]physics.BodyId = undefined;
        _ = try stack(&world, &crates);

        try testing.expect(try untilAsleep(&world, &jobs, 240));
        var before: [4]Vec2 = undefined;
        for (crates, &before) |c, *p| p.* = world.body(c).?.position();

        // Asleep, a stack is still touching - four contacts, none of them
        // ending - but nothing is tested and nothing is solved.
        try steps(&world, &jobs, 60);
        try testing.expectEqual(@as(usize, 4), world.touchingCount());
        try testing.expectEqual(@as(usize, 0), world.endEvents().len);
        try testing.expectEqual(@as(usize, 0), world.pairs.items.len);
        try testing.expectEqual(@as(usize, 0), world.constraints.items.len);
        for (crates, before) |c, p| try testing.expect(world.body(c).?.position().eql(p));
    }
}

test "a ball dropped on a sleeping stack wakes all of it" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();
    var crates: [4]physics.BodyId = undefined;
    _ = try stack(&world, &crates);
    try testing.expect(try untilAsleep(&world, &jobs, 240));

    const ball = try world.createBody(.{ .position = .init(0.2, -6) });
    _ = try world.addShape(ball, .circle(0.3));
    var woke = false;
    for (0..120) |_| {
        try world.step(dt, &jobs);
        if (world.awakeCount() == crates.len + 1) {
            woke = true;
            break;
        }
    }
    try testing.expect(woke);
    // And it all goes back to sleep once it has settled, ball included.
    try testing.expect(try untilAsleep(&world, &jobs, 600));
}

test "destroying the bottom of a sleeping stack brings the rest down" {
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();
        var world: World = .init(gpa, .{});
        defer world.deinit();
        var crates: [4]physics.BodyId = undefined;
        _ = try stack(&world, &crates);
        try testing.expect(try untilAsleep(&world, &jobs, 240));
        const top_before = world.body(crates[3]).?.position().y;

        world.destroyBody(crates[0]);
        try world.step(dt, &jobs);
        // The contacts with what is gone ended, and waking spread up the
        // stack in the same step.
        try testing.expectEqual(@as(usize, 3), world.awakeCount());
        try steps(&world, &jobs, 120);
        try testing.expectApproxEqAbs(top_before + 1, world.body(crates[3]).?.position().y, 0.05);
    }
}

test "a push of any kind wakes a sleeping body and whatever it rests on" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();
    var crates: [2]physics.BodyId = undefined;
    _ = try stack(&world, &crates);

    const Push = enum { velocity, impulse, force, torque, transform, shape };
    for (std.enums.values(Push)) |push| {
        try testing.expect(try untilAsleep(&world, &jobs, 600));
        const top = world.body(crates[1]).?;
        try testing.expect(!top.isAwake());
        switch (push) {
            .velocity => top.linear_velocity = .init(0.5, 0),
            .impulse => top.applyImpulse(.init(0.5, 0), top.center),
            .force => top.applyForce(.init(30, 0)),
            .torque => top.applyTorque(5),
            .transform => top.setTransform(top.position().add(.init(0.1, 0)), 0),
            .shape => _ = try world.addShape(crates[1], .{ .geometry = .{ .circle = .{ .radius = 0.2 } } }),
        }
        // The body it rests on is woken with it, by the end of the step.
        try world.step(dt, &jobs);
        try testing.expect(world.body(crates[1]).?.isAwake());
        try testing.expect(world.body(crates[0]).?.isAwake());
    }
}

test "a platform that starts to move wakes what sleeps on it, and carries it" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();

    const platform = try world.createBody(.{ .type = .kinematic });
    _ = try world.addShape(platform, .box(2, 0.25));
    const crate = try world.createBody(.{ .position = .init(0, -0.75) });
    _ = try world.addShape(crate, .box(0.5, 0.5));
    try testing.expect(try untilAsleep(&world, &jobs, 240));

    world.body(platform).?.linear_velocity = .init(1, 0);
    try steps(&world, &jobs, 60);
    try testing.expect(world.body(crate).?.isAwake());
    try testing.expect(world.body(crate).?.position().x > 0.8);

    // And a platform moved by hand, rather than driven, wakes it too:
    // the sleeping contact against it is not trusted to still be true.
    world.body(platform).?.linear_velocity = .zero;
    try testing.expect(try untilAsleep(&world, &jobs, 240));
    world.body(platform).?.setTransform(.init(10, 1), 0);
    try world.step(dt, &jobs);
    try testing.expect(world.body(crate).?.isAwake());
    try steps(&world, &jobs, 60);
    try testing.expect(world.body(crate).?.position().y > 1);
}

test "a motor given a speed, or a pointer moved, wakes what its joint holds" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();

    const floor = try world.createBody(.{ .type = .static, .position = .init(0, 0.5) });
    _ = try world.addShape(floor, .box(50, 0.5));

    // A wheel on a pin, its motor holding it still: asleep, like anything
    // at rest. Given a speed, it wakes and turns.
    const wheel = try world.createBody(.{ .position = .init(0, -3) });
    _ = try world.addShape(wheel, .circle(0.3));
    const axle = try world.createJoint(.{ .revolute = .{
        .body_a = floor,
        .body_b = wheel,
        .anchor = .init(0, -3),
        .motor = .{ .speed = 0, .max_torque = 50 },
    } });
    try testing.expect(try untilAsleep(&world, &jobs, 600));

    world.joint(axle).?.kind.revolute.motor.?.speed = 8;
    try steps(&world, &jobs, 30);
    try testing.expect(world.body(wheel).?.isAwake());
    try testing.expectApproxEqAbs(@as(f32, 8), world.body(wheel).?.angular_velocity, 1e-3);

    // A mouse joint, with the pointer left where it was, lets its body
    // sleep; moving the pointer wakes it.
    world.joint(axle).?.kind.revolute.motor.?.speed = 0;
    const crate = try world.createBody(.{ .position = .init(-5, -0.5) });
    _ = try world.addShape(crate, .box(0.5, 0.5));
    try testing.expect(try untilAsleep(&world, &jobs, 600));
    const drag = try world.createJoint(.{ .mouse = .{ .body = crate, .target = world.body(crate).?.center } });
    try testing.expect(try untilAsleep(&world, &jobs, 600));
    world.joint(drag).?.kind.mouse.target = world.body(crate).?.center.add(.init(0, -2));
    try world.step(dt, &jobs);
    try testing.expect(world.body(crate).?.isAwake());
}

test "a body that may not sleep keeps its island awake, and sleep can be turned off" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();
    var crates: [2]physics.BodyId = undefined;
    _ = try stack(&world, &crates);
    const insomniac = try world.createBody(.{ .position = .init(0, -2.5), .allow_sleep = false });
    _ = try world.addShape(insomniac, .box(0.5, 0.5));

    try steps(&world, &jobs, 240);
    try testing.expectEqual(@as(usize, 3), world.awakeCount());

    // Without it the stack sleeps; with sleeping switched off, it wakes.
    world.destroyBody(insomniac);
    try testing.expect(try untilAsleep(&world, &jobs, 240));
    world.settings.enable_sleep = false;
    try world.step(dt, &jobs);
    try testing.expectEqual(@as(usize, 2), world.awakeCount());
    try steps(&world, &jobs, 120);
    try testing.expectEqual(@as(usize, 2), world.awakeCount());
}

test "a body put to sleep by hand stays asleep until something wakes it" {
    var jobs: Jobs = try .init(gpa, .{});
    defer jobs.deinit();
    var world: World = .init(gpa, .{});
    defer world.deinit();

    // A crate in mid-air, put to sleep: it hangs there, as a level loaded
    // at rest would, until pushed.
    const crate = try world.createBody(.{ .position = .init(0, -5) });
    _ = try world.addShape(crate, .box(0.5, 0.5));
    world.body(crate).?.sleep();
    try steps(&world, &jobs, 60);
    try testing.expect(!world.body(crate).?.isAwake());
    try testing.expectApproxEqAbs(@as(f32, -5), world.body(crate).?.position().y, 1e-6);

    world.body(crate).?.wake();
    try steps(&world, &jobs, 30);
    try testing.expect(world.body(crate).?.position().y > -4.5);
}
