// SPDX-License-Identifier: BSD-2-Clause

//! A contact as the solver sees it: two bodies, a normal, and for each
//! point the numbers that turn a relative velocity into an impulse.
//!
//! **Sequential impulses.** Every contact is solved on its own, several
//! times over, each pass correcting what the last pass of the others undid.
//! A pass through one contact is a handful of multiplies, so eight passes
//! through a thousand contacts is nothing, and the answer converges to the
//! one a global solve would give - slowly for a tall stack, which is the
//! known weakness and the reason a stack has a limit. Box2D, Bullet and
//! every game engine of the last twenty years do it this way.
//!
//! **Warm starting is where the stability is.** The impulse a point ended
//! the last step with is applied before this step's first pass, so the
//! solver starts from an answer that was right a sixtieth of a second ago
//! and has only the change to find. Without it a resting stack is solved
//! from zero every step, never quite gets there, and shivers.
//!
//! **Two points are solved as one.** A box on a box touches at two corners,
//! and solving them one after the other gives the second the last word in
//! every pass: the stack leans, and ten crates high it falls. `solvePair`
//! solves both corners together, which is what lets a stack of twenty-five
//! stand at eight passes.
//!
//! **Sinking in is pushed out on a velocity of its own.** A point sunk in
//! by more than the slop has to be pushed out, and the obvious way -
//! Baumgarte's, asking the solver for a separating speed - leaves that
//! speed in the bodies afterwards: every correction adds energy, two
//! crates made inside each other leave at three metres a second for ever,
//! and a pile pushed apart a little everywhere every step never rests. So
//! the pushing is done on each body's *push velocity* instead, in passes
//! of its own after the solve: the push moves the body when positions are
//! integrated and is then forgotten, and the velocity a game sees never
//! had it in. These are *split impulses*, Bullet's name for them; Box2D
//! used them for a while too, before it grew a position pass.
//!
//! Nothing here touches a body that cannot move, so two contacts against
//! one wall may be solved at once. See `solver`.

const std = @import("std");
const testing = std.testing;

const geometry = @import("geometry.zig");
const Vec2 = geometry.Vec2;
const cross = geometry.cross;
const crossSV = geometry.crossSV;
const Body = @import("body.zig");
const shape = @import("shape.zig");
const collide = @import("collide.zig");
const Manifold = collide.Manifold;

/// The two shapes a contact is between, as their handles' bits. What the
/// impulses are remembered by from one step to the next.
pub const PairKey = struct {
    a: u64,
    b: u64,
};

/// What a point ended a step with, kept for the next one.
pub const Impulse = struct {
    id: u16 = 0,
    normal: f32 = 0,
    tangent: f32 = 0,
};

/// The knobs a step turns bias into, worked out once per step.
pub const Bias = struct {
    /// How much of the penetration to push out per step, as a fraction.
    baumgarte: f32,
    /// One over the step, to turn a distance into a velocity.
    inv_dt: f32,
    /// How deep a point may sit before it is pushed. Some is necessary:
    /// pushing to exactly zero makes a contact flicker on and off.
    slop: f32,
    /// Slower than this, nothing bounces. Stops a resting ball from
    /// twitching on the tiny approach speed gravity gives it each step.
    restitution_threshold: f32,
    /// The fastest a push may be.
    max_push_speed: f32,
};

/// One point of a contact, prepared.
pub const PointConstraint = struct {
    /// From each body's centre of mass to the point.
    ra: Vec2,
    rb: Vec2,
    /// The effective mass along the normal and along the tangent: what one
    /// unit of impulse there changes the relative velocity by, inverted.
    normal_mass: f32,
    tangent_mass: f32,
    /// The separating speed the solve aims for: a bounce, when the two came
    /// together fast enough to be a hit, and nothing for a rest.
    bounce: f32,
    /// The separating speed the push aims for: a fifth of the sinking-in
    /// per step, no faster than `Bias.max_push_speed`.
    push: f32,
    normal_impulse: f32,
    tangent_impulse: f32,
    /// What the push passes have pushed this step. Never warm-started:
    /// last step's sinking-in was last step's.
    push_impulse: f32,
    id: u16,
};

pub const Constraint = struct {
    key: PairKey,
    /// Slot indices into the world's body table.
    body_a: u32,
    body_b: u32,
    normal: Vec2,
    tangent: Vec2,
    friction: f32,
    restitution: f32,
    inv_mass_a: f32,
    inv_mass_b: f32,
    inv_inertia_a: f32,
    inv_inertia_b: f32,
    points: [2]PointConstraint,
    count: u32,
    /// For two points: how each one's normal impulse moves both points,
    /// and the inverse, so the two can be solved at once. See `solvePair`.
    block: Block = .{},

    pub fn pointSlice(self: *const Constraint) []const PointConstraint {
        return self.points[0..self.count];
    }
};

/// A symmetric two-by-two matrix `k` and its inverse `m`.
pub const Block = struct {
    k11: f32 = 0,
    k12: f32 = 0,
    k22: f32 = 0,
    m11: f32 = 0,
    m12: f32 = 0,
    m22: f32 = 0,
};

/// Past this, two points are so nearly one that the matrix built from them
/// cannot be trusted to invert, and only one is kept. Box2D's number.
const max_condition = 1000;

/// Build the constraint for one manifold. `warm` is what the same pair
/// ended the last step with, or null if it was not touching then.
pub fn prepare(
    key: PairKey,
    manifold: *const Manifold,
    a: *const Body,
    b: *const Body,
    index_a: u32,
    index_b: u32,
    material_a: shape.Material,
    material_b: shape.Material,
    warm: ?[2]Impulse,
    bias: Bias,
) Constraint {
    const normal = manifold.normal;
    const tangent = geometry.crossVS(normal, 1);
    var c: Constraint = .{
        .key = key,
        .body_a = index_a,
        .body_b = index_b,
        .normal = normal,
        .tangent = tangent,
        // Ice on anything is slippery; a superball on anything bounces.
        .friction = @sqrt(material_a.friction * material_b.friction),
        .restitution = @max(material_a.restitution, material_b.restitution),
        .inv_mass_a = a.inv_mass,
        .inv_mass_b = b.inv_mass,
        .inv_inertia_a = a.inv_inertia,
        .inv_inertia_b = b.inv_inertia,
        .points = undefined,
        .count = manifold.count,
    };

    for (manifold.pointSlice(), 0..) |mp, i| {
        const ra = mp.point.sub(a.center);
        const rb = mp.point.sub(b.center);

        const rna = cross(ra, normal);
        const rnb = cross(rb, normal);
        const k_normal = c.inv_mass_a + c.inv_mass_b + c.inv_inertia_a * rna * rna + c.inv_inertia_b * rnb * rnb;

        const rta = cross(ra, tangent);
        const rtb = cross(rb, tangent);
        const k_tangent = c.inv_mass_a + c.inv_mass_b + c.inv_inertia_a * rta * rta + c.inv_inertia_b * rtb * rtb;

        // The approach speed before anything is solved. Restitution is a
        // fraction of it, and only when it is fast enough to be a hit
        // rather than a rest.
        const dv = b.linear_velocity.add(crossSV(b.angular_velocity, rb))
            .sub(a.linear_velocity).sub(crossSV(a.angular_velocity, ra));
        const vn = dv.dot(normal);

        var point: PointConstraint = .{
            .ra = ra,
            .rb = rb,
            .normal_mass = if (k_normal > 0) 1 / k_normal else 0,
            .tangent_mass = if (k_tangent > 0) 1 / k_tangent else 0,
            .bounce = if (vn < -bias.restitution_threshold) -c.restitution * vn else 0,
            .push = @min(
                -bias.baumgarte * bias.inv_dt * @min(0, mp.separation + bias.slop),
                bias.max_push_speed,
            ),
            .normal_impulse = 0,
            .tangent_impulse = 0,
            .push_impulse = 0,
            .id = mp.id,
        };
        if (warm) |last| {
            for (last) |w| {
                if (w.id == mp.id) {
                    point.normal_impulse = w.normal;
                    point.tangent_impulse = w.tangent;
                    break;
                }
            }
        }
        c.points[i] = point;
    }

    if (c.count == 2) {
        const ma = c.inv_mass_a;
        const mb = c.inv_mass_b;
        const ia = c.inv_inertia_a;
        const ib = c.inv_inertia_b;
        const rn1a = cross(c.points[0].ra, normal);
        const rn1b = cross(c.points[0].rb, normal);
        const rn2a = cross(c.points[1].ra, normal);
        const rn2b = cross(c.points[1].rb, normal);
        const k11 = ma + mb + ia * rn1a * rn1a + ib * rn1b * rn1b;
        const k22 = ma + mb + ia * rn2a * rn2a + ib * rn2b * rn2b;
        const k12 = ma + mb + ia * rn1a * rn2a + ib * rn1b * rn2b;
        const det = k11 * k22 - k12 * k12;
        if (k11 * k11 < max_condition * det) {
            const inv = 1 / det;
            c.block = .{ .k11 = k11, .k12 = k12, .k22 = k22, .m11 = inv * k22, .m12 = -inv * k12, .m22 = inv * k11 };
        } else {
            // Two points saying the same thing twice - a body that cannot
            // turn, or two corners on top of each other: keep one.
            c.count = 1;
        }
    }
    return c;
}

/// Which velocities a pass works on. See the module comment.
pub const Pass = enum {
    /// The bodies' velocities: friction, what stops them going into each
    /// other, and bounces.
    solve,
    /// Their push velocities: sunk-in points pushed apart.
    push,
};

/// The four numbers a pass works on, copied out of the two bodies so the
/// pass works on locals, and written back after.
const Velocities = struct {
    va: Vec2,
    wa: f32,
    vb: Vec2,
    wb: f32,

    fn of(a: *const Body, b: *const Body, comptime pass: Pass) Velocities {
        return switch (pass) {
            .solve => .{ .va = a.linear_velocity, .wa = a.angular_velocity, .vb = b.linear_velocity, .wb = b.angular_velocity },
            .push => .{ .va = a.push_velocity, .wa = a.push_angular, .vb = b.push_velocity, .wb = b.push_angular },
        };
    }

    /// How fast the point at `ra` and `rb` is separating along `n`.
    inline fn along(v: *const Velocities, n: Vec2, ra: Vec2, rb: Vec2) f32 {
        return v.vb.add(crossSV(v.wb, rb)).sub(v.va).sub(crossSV(v.wa, ra)).dot(n);
    }

    inline fn push(v: *Velocities, c: *const Constraint, impulse: Vec2, ra: Vec2, rb: Vec2) void {
        v.va = v.va.mulAdd(impulse, -c.inv_mass_a);
        v.wa -= c.inv_inertia_a * cross(ra, impulse);
        v.vb = v.vb.mulAdd(impulse, c.inv_mass_b);
        v.wb += c.inv_inertia_b * cross(rb, impulse);
    }

    /// Write back - only to bodies that can move. A body with no inverse
    /// mass had nothing added to its copy, and writing the same number back
    /// would still be a write from two jobs at once.
    fn store(v: Velocities, c: *const Constraint, a: *Body, b: *Body, comptime pass: Pass) void {
        if (c.inv_mass_a != 0) switch (pass) {
            .solve => {
                a.linear_velocity = v.va;
                a.angular_velocity = v.wa;
            },
            .push => {
                a.push_velocity = v.va;
                a.push_angular = v.wa;
            },
        };
        if (c.inv_mass_b != 0) switch (pass) {
            .solve => {
                b.linear_velocity = v.vb;
                b.angular_velocity = v.wb;
            },
            .push => {
                b.push_velocity = v.vb;
                b.push_angular = v.wb;
            },
        };
    }
};

/// Apply last step's impulses before the first pass.
pub fn warmStart(c: *const Constraint, a: *Body, b: *Body) void {
    var v: Velocities = .of(a, b, .solve);
    for (c.pointSlice()) |p| {
        v.push(c, c.normal.scale(p.normal_impulse).add(c.tangent.scale(p.tangent_impulse)), p.ra, p.rb);
    }
    v.store(c, a, b, .solve);
}

/// One pass. For `.solve`, friction first and then the normal, Box2D's
/// order: the normal impulse is the more important and going last lets it
/// have the final say. For `.push`, only the normal: a push has no
/// friction, because it is not a motion, only a correction.
///
/// `pass` is `comptime`, so each kind is compiled as its own function, and
/// the loop that calls it a dozen times a step never asks which it is.
pub fn solve(c: *Constraint, a: *Body, b: *Body, comptime pass: Pass) void {
    var v: Velocities = .of(a, b, pass);

    if (pass == .solve) {
        for (c.points[0..c.count]) |*p| {
            const vt = v.along(c.tangent, p.ra, p.rb);
            var lambda = p.tangent_mass * (-vt);

            // Coulomb: friction can be no more than a fraction of how hard
            // the surfaces are pressed together, which is the normal impulse.
            const max_friction = c.friction * p.normal_impulse;
            const new_impulse = std.math.clamp(p.tangent_impulse + lambda, -max_friction, max_friction);
            lambda = new_impulse - p.tangent_impulse;
            p.tangent_impulse = new_impulse;
            v.push(c, c.tangent.scale(lambda), p.ra, p.rb);
        }
    }

    if (c.count == 2) {
        solvePair(c, &v, pass);
    } else for (c.points[0..c.count]) |*p| {
        const vn = v.along(c.normal, p.ra, p.rb);
        const total = impulseOf(p, pass);
        var lambda = -p.normal_mass * (vn - targetOf(p, pass));

        // A contact can push and never pull: the accumulated impulse is
        // clamped, not this pass's, which is what lets a pass take back
        // what an earlier one over-applied.
        const new_impulse = @max(total.* + lambda, 0);
        lambda = new_impulse - total.*;
        total.* = new_impulse;
        v.push(c, c.normal.scale(lambda), p.ra, p.rb);
    }

    v.store(c, a, b, pass);
}

/// The accumulated impulse a pass adds to.
inline fn impulseOf(p: *PointConstraint, comptime pass: Pass) *f32 {
    return switch (pass) {
        .solve => &p.normal_impulse,
        .push => &p.push_impulse,
    };
}

/// The separating speed a pass aims for.
inline fn targetOf(p: *const PointConstraint, comptime pass: Pass) f32 {
    return switch (pass) {
        .solve => p.bounce,
        .push => p.push,
    };
}

/// The normal impulses of a two-point contact, both at once.
///
/// **Why not one after the other**: a box resting on another touches it at
/// two corners, and solving the left corner and then the right leaves the
/// right with the last word in every pass. The left one is always a little
/// short, the box always leans a little that way, and in a stack the lean
/// adds up floor by floor until the stack falls - ten crates high, at
/// eight passes, before this was here. Solved together, the two corners
/// share the load the way the real ones do.
///
/// Together means a tiny linear complementarity problem: find impulses
/// `x >= 0` such that the speeds they leave, `vn = K x + b`, are at least
/// the targets, and each corner either pushes or is separating, never both.
/// With two unknowns there are four cases - both push, only the first,
/// only the second, neither - and the first that is consistent is the
/// answer. Box2D's block solver, due to Erin Catto and Dirk Gregorius.
fn solvePair(c: *Constraint, v: *Velocities, comptime pass: Pass) void {
    const p1 = &c.points[0];
    const p2 = &c.points[1];
    const total1 = impulseOf(p1, pass);
    const total2 = impulseOf(p2, pass);
    const k = c.block;
    const a1 = total1.*;
    const a2 = total2.*;

    // The unknowns are the new totals, not this pass's change, so that
    // clamping them at zero clamps the totals: `b` moves to account for
    // what has already been applied.
    const b1 = v.along(c.normal, p1.ra, p1.rb) - targetOf(p1, pass) - (k.k11 * a1 + k.k12 * a2);
    const b2 = v.along(c.normal, p2.ra, p2.rb) - targetOf(p2, pass) - (k.k12 * a1 + k.k22 * a2);

    var x1: f32 = undefined;
    var x2: f32 = undefined;
    cases: {
        // Both push, and both end up at their target along the normal.
        x1 = -(k.m11 * b1 + k.m12 * b2);
        x2 = -(k.m12 * b1 + k.m22 * b2);
        if (x1 >= 0 and x2 >= 0) break :cases;
        // Only the first pushes; the second must be separating.
        x1 = -p1.normal_mass * b1;
        x2 = 0;
        if (x1 >= 0 and k.k12 * x1 + b2 >= 0) break :cases;
        // Only the second.
        x1 = 0;
        x2 = -p2.normal_mass * b2;
        if (x2 >= 0 and k.k12 * x2 + b1 >= 0) break :cases;
        // Neither: both separating already.
        x1 = 0;
        x2 = 0;
        if (b1 >= 0 and b2 >= 0) break :cases;
        // No case fits, which rounding can cause. Leave this pass out, as
        // Box2D does; the next one starts from somewhere a little different.
        return;
    }

    total1.* = x1;
    total2.* = x2;
    // Not `i1` and `i2`: those are type names in Zig - integers one and two
    // bits wide - and a variable may not shadow a type.
    const push1 = c.normal.scale(x1 - a1);
    const push2 = c.normal.scale(x2 - a2);
    v.push(c, push1, p1.ra, p1.rb);
    v.push(c, push2, p2.ra, p2.rb);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const no_bias: Bias = .{ .baumgarte = 0, .inv_dt = 60, .slop = 0, .restitution_threshold = 1, .max_push_speed = 3 };

fn ball(inv_mass: f32, velocity: Vec2) Body {
    var b: Body = .fromDef(.{ .linear_velocity = velocity });
    b.mass = if (inv_mass == 0) 0 else 1 / inv_mass;
    b.inv_mass = inv_mass;
    if (inv_mass == 0) b.type = .static;
    return b;
}

test "a head-on hit with no restitution stops, and with full restitution bounces" {
    var wall = ball(0, .zero);
    var m: Manifold = .{ .normal = .init(-1, 0), .count = 1 };
    m.points[0] = .{ .point = .init(-0.5, 0), .separation = 0, .id = 0 };

    for ([_]f32{ 0, 1 }) |e| {
        var mover = ball(1, .init(3, 0));
        const soft: shape.Material = .{ .restitution = e, .friction = 0 };
        var c = prepare(.{ .a = 1, .b = 2 }, &m, &wall, &mover, 0, 1, soft, soft, null, no_bias);
        for (0..4) |_| solve(&c, &wall, &mover, .solve);
        try testing.expectApproxEqAbs(-3 * e, mover.linear_velocity.x, 1e-5);
        // The wall was not written to.
        try testing.expect(wall.linear_velocity.eql(.zero));
    }
}

test "sinking in is pushed out on the push velocity, and the velocity never sees it" {
    var floor = ball(0, .zero);
    // Sunk a tenth into the floor and at rest.
    var m: Manifold = .{ .normal = .init(0, -1), .count = 1 };
    m.points[0] = .{ .point = .init(0, 0.5), .separation = -0.1, .id = 0 };
    var box = ball(1, .zero);
    const bias: Bias = .{ .baumgarte = 0.2, .inv_dt = 60, .slop = 0, .restitution_threshold = 1, .max_push_speed = 3 };
    var c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, .{}, .{}, null, bias);
    for (0..4) |_| solve(&c, &floor, &box, .solve);
    for (0..4) |_| solve(&c, &floor, &box, .push);

    // Nothing moving, so nothing for the solve to do...
    try testing.expect(box.linear_velocity.eql(.zero));
    // ...and the push takes a fifth of the depth this step: 0.02 over a
    // sixtieth is 1.2 m/s, upwards.
    try testing.expectApproxEqAbs(@as(f32, -1.2), box.push_velocity.y, 1e-4);
    try testing.expect(floor.push_velocity.eql(.zero));

    // A box coming down onto the floor is stopped by the solve, whatever
    // the push is doing, and keeps the impulse that stopped it.
    var falling = ball(1, .init(0, 2));
    var held = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &falling, 0, 1, .{}, .{}, null, bias);
    for (0..4) |_| solve(&held, &floor, &falling, .solve);
    try testing.expectApproxEqAbs(@as(f32, 0), falling.linear_velocity.y, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 2), held.points[0].normal_impulse, 1e-5);
}

test "two corners are solved together, and share a load evenly" {
    var floor = ball(0, .zero);
    // A unit box, turned so it can: inertia of a unit square of mass one.
    var box = ball(1, .init(0, 1));
    box.inertia = 1.0 / 6.0;
    box.inv_inertia = 6;
    var m: Manifold = .{ .normal = .init(0, -1), .count = 2 };
    m.points[0] = .{ .point = .init(-0.5, 0.5), .separation = 0, .id = 1 };
    m.points[1] = .{ .point = .init(0.5, 0.5), .separation = 0, .id = 2 };
    var c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, .{}, .{}, null, no_bias);
    try testing.expectEqual(@as(u32, 2), c.count);

    // One pass is enough: the block gets both corners exactly, where one
    // after the other would leave the second with more.
    solve(&c, &floor, &box, .solve);
    try testing.expectApproxEqAbs(@as(f32, 0), box.linear_velocity.y, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), box.angular_velocity, 1e-5);
    try testing.expectApproxEqAbs(c.points[0].normal_impulse, c.points[1].normal_impulse, 1e-5);
}

test "friction slows a sliding box and cannot exceed its share of the normal impulse" {
    var floor = ball(0, .zero);
    var box = ball(1, .init(4, 1));
    // The floor is below (+y); the normal from floor to box points up (-y).
    var m: Manifold = .{ .normal = .init(0, -1), .count = 1 };
    m.points[0] = .{ .point = .init(0, 0.5), .separation = 0, .id = 0 };

    const rough: shape.Material = .{ .friction = 0.5 };
    var c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, rough, rough, null, no_bias);
    for (0..8) |_| solve(&c, &floor, &box, .solve);

    // The downward speed of 1 is stopped, which took a normal impulse of 1;
    // friction may then take at most 0.5 off the sideways speed.
    try testing.expectApproxEqAbs(@as(f32, 0), box.linear_velocity.y, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.5), box.linear_velocity.x, 1e-4);
}

test "warm starting picks up the impulse by id, not by position" {
    var floor = ball(0, .zero);
    // A box that can turn. One that cannot has two corners that say the
    // same thing, and the block solver keeps only one of them.
    var box = ball(1, .zero);
    box.inertia = 1.0 / 6.0;
    box.inv_inertia = 6;
    var m: Manifold = .{ .normal = .init(0, -1), .count = 2 };
    m.points[0] = .{ .point = .init(-0.5, 0.5), .separation = 0, .id = 7 };
    m.points[1] = .{ .point = .init(0.5, 0.5), .separation = 0, .id = 9 };
    const last = [2]Impulse{ .{ .id = 9, .normal = 2 }, .{ .id = 7, .normal = 1 } };
    const c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, .{}, .{}, last, no_bias);
    try testing.expectEqual(@as(f32, 1), c.points[0].normal_impulse);
    try testing.expectEqual(@as(f32, 2), c.points[1].normal_impulse);

    warmStart(&c, &floor, &box);
    // Three units of impulse along -y on a unit mass.
    try testing.expectApproxEqAbs(@as(f32, -3), box.linear_velocity.y, 1e-6);
}
