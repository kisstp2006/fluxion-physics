// SPDX-License-Identifier: BSD-2-Clause

//! Fluxion Physics in a browser. Built by `zig build web` into
//! `zig-out/web/`, and driven by `index.html` next to it.
//!
//! The same world and the same step as the native demo. What is different
//! is that `wasm32-freestanding` has no threads, so the scheduler has no
//! workers and every phase runs on the thread that asked - the page's
//! animation frame. Nothing is compiled out and nothing is stubbed.
//!
//! Two scenes: a heap poured into a bowl, and a yard of joints - a rope
//! bridge, a chain, a spring, and a car the arrow keys drive. In both, the
//! pointer drags whatever it is pressed on through a mouse joint, and a
//! body that has fallen asleep is drawn grey.
//!
//! The page reads the world straight out of wasm memory: `shapes()` fills a
//! flat float buffer with one record per shape and `joints()` another with
//! one per joint, and the page draws each.

const std = @import("std");
const physics = @import("fluxion_physics");
const World = physics.World;
const Jobs = physics.Jobs;
const Vec2 = physics.Vec2;

const gpa = std.heap.wasm_allocator;

var world: ?World = null;
var jobs: ?Jobs = null;

/// One record per shape: kind (0 circle, 1 polygon), x, y, angle, radius
/// or vertex count, up to eight local vertices, and whether it is awake.
const record_len = 22;
var records: []f32 = &.{};
/// One record per joint: kind, then its two anchors in the world.
const joint_record_len = 5;
var joint_records: []f32 = &.{};

/// What the pointer is holding, if anything.
var grabbed: physics.JointId = .none;
/// The car's two axles, in the yard of joints.
var axles: [2]physics.JointId = .{ .none, .none };

const dt: f32 = 1.0 / 60.0;

/// A bowl `w` by `h` pixels with `n` bodies dropped into it. A hundred
/// pixels is a metre, so gravity and the slop are sensible for the size
/// of thing on screen.
export fn start(w: f32, h: f32, n: u32) u32 {
    var made = begin() orelse return 0;
    bowl(&made, w, h) catch return 0;

    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();
    for (0..n) |i| {
        const b = made.createBody(.{
            .position = .init(w * 0.2 + random.float(f32) * w * 0.6, -@as(f32, @floatFromInt(i)) * 30),
            .angle = random.float(f32) * 6,
        }) catch break;
        if (i % 2 == 0) {
            _ = made.addShape(b, .{
                .geometry = .{ .circle = .{ .radius = 8 + random.float(f32) * 12 } },
                .material = .{ .restitution = 0.4 },
            }) catch break;
        } else {
            _ = made.addShape(b, .box(8 + random.float(f32) * 14, 8 + random.float(f32) * 14)) catch break;
        }
    }
    return finish(made);
}

/// A yard of joints: a rope bridge between two posts with crates to drop
/// on it, a chain, a weight on a spring, and a car.
export fn startJoints(w: f32, h: f32) u32 {
    var made = begin() orelse return 0;
    yard(&made, w, h) catch return 0;
    return finish(made);
}

fn begin() ?World {
    stop();
    jobs = Jobs.init(gpa, .{}) catch return null;
    return World.init(gpa, .{ .units_per_metre = 100 });
}

fn finish(made: World) u32 {
    records = gpa.alloc(f32, made.shapes.slotCount() * record_len) catch return 0;
    // One more joint than there are, for the pointer's.
    joint_records = gpa.alloc(f32, (made.joints.slotCount() + 1) * joint_record_len) catch return 0;
    world = made;
    return @intCast(made.bodyCount());
}

/// Thick walls, leaning outwards. Thin ones are what a squeezed body
/// tunnels through, and the static geometry is the cheap place to be
/// generous.
fn bowl(made: *World, w: f32, h: f32) !void {
    const floor = try made.createBody(.{ .type = .static, .position = .init(w / 2, h + 40) });
    _ = try made.addShape(floor, .box(w, 50));
    const left = try made.createBody(.{ .type = .static, .position = .init(-30, h / 2), .angle = -0.12 });
    _ = try made.addShape(left, .box(50, h));
    const right = try made.createBody(.{ .type = .static, .position = .init(w + 30, h / 2), .angle = 0.12 });
    _ = try made.addShape(right, .box(50, h));
}

fn yard(made: *World, w: f32, h: f32) !void {
    const ground_y = h - 10;
    const floor = try made.createBody(.{ .type = .static, .position = .init(w / 2, ground_y + 50) });
    _ = try made.addShape(floor, .{ .geometry = .{ .polygon = .box(w, 50) }, .material = .{ .friction = 0.9 } });
    const left = try made.createBody(.{ .type = .static, .position = .init(-40, h / 2) });
    _ = try made.addShape(left, .box(50, h));
    const right = try made.createBody(.{ .type = .static, .position = .init(w + 40, h / 2) });
    _ = try made.addShape(right, .box(50, h));

    // The bridge: two posts, and twelve planks pinned end to end between
    // them. The planks' pins are what hold it; nothing else does.
    const deck_y: f32 = 230;
    const post_left: f32 = 120;
    const planks = 12;
    const plank_w: f32 = 25;
    for ([_]f32{ post_left - 10, post_left + planks * plank_w + 10 }) |x| {
        const post = try made.createBody(.{ .type = .static, .position = .init(x, deck_y + (ground_y - deck_y) / 2) });
        _ = try made.addShape(post, .box(10, (ground_y - deck_y) / 2));
    }
    const anchor_left = try made.createBody(.{ .type = .static, .position = .init(post_left, deck_y) });
    var previous = anchor_left;
    for (0..planks) |i| {
        const x = post_left + (@as(f32, @floatFromInt(i)) + 0.5) * plank_w;
        const plank = try made.createBody(.{ .position = .init(x, deck_y) });
        _ = try made.addShape(plank, .{ .geometry = .{ .polygon = .box(plank_w / 2, 4) }, .material = .{ .friction = 0.8 } });
        _ = try made.createJoint(.{ .revolute = .{ .body_a = previous, .body_b = plank, .anchor = .init(x - plank_w / 2, deck_y) } });
        previous = plank;
    }
    const anchor_right = try made.createBody(.{ .type = .static, .position = .init(post_left + planks * plank_w, deck_y) });
    _ = try made.createJoint(.{ .revolute = .{ .body_a = previous, .body_b = anchor_right, .anchor = .init(post_left + planks * plank_w, deck_y) } });

    // Crates to drop on it.
    for (0..4) |i| {
        const crate = try made.createBody(.{ .position = .init(post_left + 60 + @as(f32, @floatFromInt(i)) * 60, 120 - @as(f32, @floatFromInt(i)) * 30), .angle = 0.3 });
        _ = try made.addShape(crate, .box(12, 12));
    }

    // A chain hanging from a hook, its links pinned end to end, and a ball
    // on the end of it.
    const hook_x = w - 90;
    const hook = try made.createBody(.{ .type = .static, .position = .init(hook_x, 40) });
    previous = hook;
    const link_len: f32 = 18;
    for (0..10) |i| {
        const y = 40 + (@as(f32, @floatFromInt(i)) + 0.5) * link_len;
        const link = try made.createBody(.{ .position = .init(hook_x, y) });
        _ = try made.addShape(link, .{ .geometry = .{ .polygon = .box(3, link_len / 2) }, .filter = .{ .group = -1 } });
        _ = try made.createJoint(.{ .revolute = .{ .body_a = previous, .body_b = link, .anchor = .init(hook_x, y - link_len / 2) } });
        previous = link;
    }
    const bob_y = 40 + 10 * link_len + 14;
    const bob = try made.createBody(.{ .position = .init(hook_x, bob_y) });
    _ = try made.addShape(bob, .{ .geometry = .{ .circle = .{ .radius = 14 } }, .filter = .{ .group = -1 } });
    _ = try made.createJoint(.{ .revolute = .{ .body_a = previous, .body_b = bob, .anchor = .init(hook_x, bob_y - 14) } });

    // A weight on a spring, bouncing at one and a half times a second.
    const spring_top = try made.createBody(.{ .type = .static, .position = .init(60, 30) });
    const weight = try made.createBody(.{ .position = .init(60, 130) });
    _ = try made.addShape(weight, .box(16, 16));
    _ = try made.createJoint(.{ .distance = .{
        .body_a = spring_top,
        .body_b = weight,
        .anchor_a = .init(60, 30),
        .anchor_b = .init(60, 114),
        .length = 60,
        .spring = .{ .hertz = 1.5, .damping_ratio = 0.1 },
    } });

    // A car on sprung wheels, waiting for the arrow keys.
    const car_x = w * 0.6;
    const car_y = ground_y - 45;
    const chassis = try made.createBody(.{ .position = .init(car_x, car_y) });
    _ = try made.addShape(chassis, .box(55, 10));
    _ = try made.addShape(chassis, .{ .geometry = .{ .polygon = .offsetBox(22, 8, .init(-8, -16), 0) } });
    for ([_]f32{ -38, 38 }, &axles) |dx, *axle| {
        const wheel = try made.createBody(.{ .position = .init(car_x + dx, car_y + 22) });
        _ = try made.addShape(wheel, .{ .geometry = .{ .circle = .{ .radius = 16 } }, .material = .{ .friction = 1.2 } });
        axle.* = try made.createJoint(.{ .wheel = .{
            .body_a = chassis,
            .body_b = wheel,
            .anchor = .init(car_x + dx, car_y + 22),
            .axis = .init(0, -1),
            .spring = .{ .hertz = 4, .damping_ratio = 0.7 },
            .motor = .{ .speed = 0, .max_torque = 0 },
        } });
    }
    // Enough torque to climb out of trouble: the whole car's weight, at
    // the rim of a wheel, twice over.
    var mass: f32 = made.body(chassis).?.mass;
    for (axles) |axle| mass += made.body(made.joint(axle).?.body_b).?.mass;
    const torque = 2 * mass * 981 * 16;
    for (axles) |axle| made.joint(axle).?.kind.wheel.motor.?.max_torque = torque;
}

/// One fixed step. Returns how many pairs are touching.
export fn tick() u32 {
    const w = &(world orelse return 0);
    const js = &(jobs orelse return 0);
    w.step(dt, js) catch return 0;
    return @intCast(w.touchingCount());
}

/// Write every shape into the record buffer and say how many there are.
export fn shapes() u32 {
    const w = &(world orelse return 0);
    var count: u32 = 0;
    var it = w.shapeIterator();
    while (it.next()) |entry| {
        const xf = w.shapeTransform(entry.value);
        const b = w.bodyConst(entry.value.body).?;
        const r = records[count * record_len ..][0..record_len];
        r[1] = xf.p.x;
        r[2] = xf.p.y;
        r[3] = xf.q.angle();
        switch (entry.value.def.geometry) {
            .circle => |c| {
                r[0] = 0;
                const centre = xf.apply(c.center);
                r[1] = centre.x;
                r[2] = centre.y;
                r[4] = c.radius;
            },
            .polygon => |*p| {
                r[0] = 1;
                r[4] = @floatFromInt(p.count);
                for (p.vertexSlice(), 0..) |v, i| {
                    r[5 + i * 2] = v.x;
                    r[6 + i * 2] = v.y;
                }
            },
            // Drawn as the outline a polygon of eight corners makes of it:
            // four round each end.
            .capsule => |c| {
                r[0] = 1;
                r[4] = 8;
                const along = c.center2.sub(c.center1);
                const turn = std.math.atan2(along.y, along.x);
                for (0..8) |i| {
                    const end = if (i < 4) c.center2 else c.center1;
                    const angle = turn - std.math.pi / 2.0 + @as(f32, @floatFromInt(i)) * std.math.pi / 3.0 - (if (i < 4) @as(f32, 0) else std.math.pi / 3.0);
                    r[5 + i * 2] = end.x + c.radius * @cos(angle);
                    r[6 + i * 2] = end.y + c.radius * @sin(angle);
                }
            },
        }
        // Static bodies are neither: draw them as the scenery they are.
        r[21] = switch (b.type) {
            .static => 2,
            else => if (b.isAwake()) 1 else 0,
        };
        count += 1;
    }
    return count;
}

/// Write every joint into the joint buffer and say how many there are.
export fn joints() u32 {
    const w = &(world orelse return 0);
    var count: u32 = 0;
    var it = w.jointIterator();
    while (it.next()) |entry| {
        if (count * joint_record_len >= joint_records.len) break;
        const anchors = w.jointAnchors(entry.value);
        const r = joint_records[count * joint_record_len ..][0..joint_record_len];
        r[0] = @floatFromInt(@intFromEnum(entry.value.kind));
        r[1] = anchors[0].x;
        r[2] = anchors[0].y;
        r[3] = anchors[1].x;
        r[4] = anchors[1].y;
        count += 1;
    }
    return count;
}

export fn recordsPtr() [*]f32 {
    return records.ptr;
}

export fn recordLen() u32 {
    return record_len;
}

export fn jointRecordsPtr() [*]f32 {
    return joint_records.ptr;
}

export fn jointRecordLen() u32 {
    return joint_record_len;
}

/// Press: take hold of the dynamic body under the pointer, if there is one.
export fn grab(x: f32, y: f32) u32 {
    const w = &(world orelse return 0);
    release();
    const shape = w.overlapPoint(.init(x, y)) orelse return 0;
    const body = w.shape(shape).?.body;
    if (w.bodyConst(body).?.type != .dynamic) return 0;
    grabbed = w.createJoint(.{ .mouse = .{ .body = body, .target = .init(x, y) } }) catch return 0;
    return 1;
}

/// Move: pull what is held towards the pointer. Moving the target is all
/// it takes - and it is also what wakes a body that had fallen asleep.
export fn drag(x: f32, y: f32) void {
    const w = &(world orelse return);
    const j = w.joint(grabbed) orelse return;
    j.kind.mouse.target = .init(x, y);
}

/// Let go.
export fn release() void {
    const w = &(world orelse return);
    w.destroyJoint(grabbed);
    grabbed = .none;
}

/// The car's throttle, in radians per second of the wheels. Writing a
/// motor's speed is all it takes, asleep or not.
export fn drive(speed: f32) void {
    const w = &(world orelse return);
    for (axles) |axle| {
        const j = w.joint(axle) orelse continue;
        j.kind.wheel.motor.?.speed = speed;
    }
}

/// Kick everything upwards. Proof the world is live, not a recording.
export fn shake() void {
    const w = &(world orelse return);
    var it = w.bodyIterator();
    while (it.next()) |entry| {
        entry.value.applyImpulse(.init(0, -entry.value.mass * 600), entry.value.center);
    }
}

export fn awake() u32 {
    const w = &(world orelse return 0);
    return @intCast(w.awakeCount());
}

export fn workers() u32 {
    const js = &(jobs orelse return 0);
    return @intCast(js.workerCount());
}

export fn stop() void {
    if (world) |*w| w.deinit();
    if (jobs) |*js| js.deinit();
    world = null;
    jobs = null;
    grabbed = .none;
    axles = .{ .none, .none };
    if (records.len != 0) gpa.free(records);
    if (joint_records.len != 0) gpa.free(joint_records);
    records = &.{};
    joint_records = &.{};
}
