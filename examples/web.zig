// SPDX-License-Identifier: BSD-2-Clause

//! Fluxion Physics in a browser. Built by `zig build web` into
//! `zig-out/web/`, and driven by `index.html` next to it.
//!
//! The same world and the same step as the native demo. What is different
//! is that `wasm32-freestanding` has no threads, so the scheduler has no
//! workers and every phase runs on the thread that asked - the page's
//! animation frame. Nothing is compiled out and nothing is stubbed.
//!
//! The page reads the shapes straight out of wasm memory: `shapes()` fills
//! a flat float buffer with one record per shape, and the page draws each.

const std = @import("std");
const physics = @import("fluxion_physics");
const World = physics.World;
const Jobs = physics.Jobs;

const gpa = std.heap.wasm_allocator;

var world: ?World = null;
var jobs: ?Jobs = null;
/// One record per shape: kind (0 circle, 1 polygon), x, y, angle, radius
/// or vertex count, then up to eight local vertices. Twenty-two floats.
const record_len = 22;
var records: []f32 = &.{};

const dt: f32 = 1.0 / 60.0;

/// A bowl `w` by `h` pixels with `n` bodies dropped into it. A hundred
/// pixels is a metre, so gravity and the slop are sensible for the size
/// of thing on screen.
export fn start(w: f32, h: f32, n: u32) u32 {
    stop();
    var made: World = .init(gpa, .{ .units_per_metre = 100 });
    const js: Jobs = Jobs.init(gpa, .{}) catch return 0;

    // Thick walls, leaning outwards. Thin ones are what a squeezed body
    // tunnels through, and the static geometry is the cheap place to be
    // generous.
    const floor = made.createBody(.{ .type = .static, .position = .init(w / 2, h + 40) }) catch return 0;
    _ = made.addShape(floor, .box(w, 50)) catch return 0;
    const left = made.createBody(.{ .type = .static, .position = .init(-30, h / 2), .angle = -0.12 }) catch return 0;
    _ = made.addShape(left, .box(50, h)) catch return 0;
    const right = made.createBody(.{ .type = .static, .position = .init(w + 30, h / 2), .angle = 0.12 }) catch return 0;
    _ = made.addShape(right, .box(50, h)) catch return 0;

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

    records = gpa.alloc(f32, made.shapes.slotCount() * record_len) catch return 0;
    world = made;
    jobs = js;
    return @intCast(made.bodyCount());
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
        }
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

/// Kick everything upwards. Proof the world is live, not a recording.
export fn shake() void {
    const w = &(world orelse return);
    var it = w.bodyIterator();
    while (it.next()) |entry| {
        entry.value.applyImpulse(.init(0, -entry.value.mass * 600), entry.value.center);
    }
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
    if (records.len != 0) gpa.free(records);
    records = &.{};
}
