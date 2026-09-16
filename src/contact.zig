// SPDX-License-Identifier: BSD-2-Clause

//! A contact as the solver sees it: two bodies, a normal, and for each
//! point the numbers that turn a relative velocity into an impulse.
//!
//! **Sequential impulses.** Every contact is solved on its own, each pass
//! correcting what the last pass of the others undid, and the answer
//! converges to the one a global solve would give. Box2D, Bullet and every
//! game engine of the last twenty years do it this way.
//!
//! **Solved in substeps.** A step is cut into a few substeps, and each one
//! integrates the velocities, solves once, moves the bodies, and relaxes
//! once - see `World.step`. The manifold is found once per step, but how
//! deep each point is sunk is worked out again every substep from where
//! the bodies are now: each point is kept in both bodies' own frames, and
//! the gap between the two is the depth. That is Box2D v3's "soft step",
//! and it is what makes a heavy crate on a light bridge, or a tower of
//! twenty-five, hold: four small solves each starting from the truth beat
//! eight passes over a picture of the world that went stale on the first.
//!
//! **Sinking in is taken back as a stiff spring would take it back**, then
//! taken back out of the velocity. While solving, a sunk-in point is pushed
//! out by a damped spring of `Settings.contact_hertz` - see `Softness` for
//! why a spring and not a fixed fraction - never faster than
//! `Settings.max_push_speed`. Once the bodies have moved, a relaxing pass
//! solves again with no push at all, which leaves the velocities what they
//! would have been had nothing needed pushing apart. So a crate made half
//! inside another slides out of it and stops, rather than leaving at speed.
//!
//! **Speculative when apart.** A point whose bodies have separated during
//! the step may close the gap by the next substep and no more, rather than
//! being ignored until they collide - which is what keeps a box on a slope
//! from bouncing on its corners.
//!
//! **Restitution comes last**, in a pass of its own after the substeps,
//! because the relaxing would otherwise take a bounce back along with the
//! push. It bounces from the speed the point came in at, before anything
//! was solved, and only where the point actually pushed.
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
const collide = @import("collide.zig");
const Manifold = collide.Manifold;
const Softness = @import("Softness.zig");

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

/// What every pass is told, worked out once per step.
pub const Step = struct {
    /// One over the substep.
    inv_h: f32,
    /// How deep a point may sit before it is pushed. Some is necessary:
    /// pushing to exactly zero makes a contact flicker on and off.
    slop: f32,
    /// The fastest a push may be.
    max_push: f32,
};

/// One point of a contact, prepared.
pub const PointConstraint = struct {
    /// From each body's centre of mass to the point, at the start of the
    /// step: the lever arms, kept through the substeps as Box2D v3 keeps
    /// them. Turning them with the bodies would be more exact and is not
    /// needed at a substep's worth of turn.
    ra: Vec2,
    rb: Vec2,
    /// The point in each body's own frame. Where the two copies are now,
    /// along the normal, is how far the point has closed or opened.
    local_a: Vec2,
    local_b: Vec2,
    /// How deep it was when the manifold was made.
    separation: f32,
    /// The effective mass along the normal and along the tangent: what one
    /// unit of impulse there changes the relative velocity by, inverted.
    normal_mass: f32,
    tangent_mass: f32,
    /// How fast the two were coming together here before anything was
    /// solved: what restitution bounces from. Negative when approaching.
    approach: f32,
    normal_impulse: f32,
    tangent_impulse: f32,
    /// The largest push any one pass gave it. Zero means it never pushed -
    /// it was near, not hit - and does not bounce.
    max_normal_impulse: f32,
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
    /// The spring sinking-in is taken back with: stiffer against something
    /// that cannot move, which has no give of its own to share.
    softness: Softness,
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

/// Build the constraint for one manifold. `friction` and `restitution` are
/// the pair's, already made from both surfaces' - see `Settings.friction_mix`.
/// `warm` is what the same pair ended the last step with, or null if it was
/// not touching then.
pub fn prepare(
    key: PairKey,
    manifold: *const Manifold,
    a: *const Body,
    b: *const Body,
    index_a: u32,
    index_b: u32,
    friction: f32,
    restitution: f32,
    warm: ?[2]Impulse,
    softness: Softness,
    static_softness: Softness,
) Constraint {
    const normal = manifold.normal;
    const tangent = geometry.crossVS(normal, 1);
    var c: Constraint = .{
        .key = key,
        .body_a = index_a,
        .body_b = index_b,
        .normal = normal,
        .tangent = tangent,
        .friction = friction,
        .restitution = restitution,
        .inv_mass_a = a.inv_mass,
        .inv_mass_b = b.inv_mass,
        .inv_inertia_a = a.inv_inertia,
        .inv_inertia_b = b.inv_inertia,
        .softness = if (a.inv_mass == 0 or b.inv_mass == 0) static_softness else softness,
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

        const dv = b.linear_velocity.add(crossSV(b.angular_velocity, rb))
            .sub(a.linear_velocity).sub(crossSV(a.angular_velocity, ra));

        var point: PointConstraint = .{
            .ra = ra,
            .rb = rb,
            .local_a = a.transform.unapply(mp.point),
            .local_b = b.transform.unapply(mp.point),
            .separation = mp.separation,
            .normal_mass = if (k_normal > 0) 1 / k_normal else 0,
            .tangent_mass = if (k_tangent > 0) 1 / k_tangent else 0,
            .approach = dv.dot(normal),
            .normal_impulse = 0,
            .tangent_impulse = 0,
            .max_normal_impulse = 0,
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

/// Which kind of pass `solve` is making. See the module comment.
pub const Pass = enum {
    /// Push sunk-in points apart, on the contact's spring.
    solve,
    /// Take back what the pushing left in the velocities.
    relax,
};

/// The four numbers a pass works on, copied out of the two bodies so the
/// pass works on locals, and written back after.
const Velocities = struct {
    va: Vec2,
    wa: f32,
    vb: Vec2,
    wb: f32,

    fn of(a: *const Body, b: *const Body) Velocities {
        return .{ .va = a.linear_velocity, .wa = a.angular_velocity, .vb = b.linear_velocity, .wb = b.angular_velocity };
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
    fn store(v: Velocities, c: *const Constraint, a: *Body, b: *Body) void {
        if (c.inv_mass_a != 0) {
            a.linear_velocity = v.va;
            a.angular_velocity = v.wa;
        }
        if (c.inv_mass_b != 0) {
            b.linear_velocity = v.vb;
            b.angular_velocity = v.wb;
        }
    }
};

/// Apply what the contact pushed last substep before this one's pass.
pub fn warmStart(c: *const Constraint, a: *Body, b: *Body) void {
    var v: Velocities = .of(a, b);
    for (c.pointSlice()) |p| {
        v.push(c, c.normal.scale(p.normal_impulse).add(c.tangent.scale(p.tangent_impulse)), p.ra, p.rb);
    }
    v.store(c, a, b);
}

/// How deep a point is now: its depth when the manifold was made, and how
/// far its two copies - one on each body - have moved apart along the
/// normal since. Negative is sunk in.
inline fn separationNow(c: *const Constraint, p: *const PointConstraint, a: *const Body, b: *const Body) f32 {
    const pa = a.transform.apply(p.local_a);
    const pb = b.transform.apply(p.local_b);
    return p.separation + pb.sub(pa).dot(c.normal);
}

/// What a point asks of its normal impulse this pass: the speed it wants
/// to separate at, as a `Softness` would write it.
const Target = struct {
    /// Added to the normal speed: negative asks for a push apart, positive
    /// allows an approach.
    bias: f32,
    mass_scale: f32,
    impulse_scale: f32,
};

fn targetOf(c: *const Constraint, s: f32, step: Step, comptime pass: Pass) Target {
    // Apart: it may close the gap by the end of the substep, and no more.
    if (s > 0) return .{ .bias = s * step.inv_h, .mass_scale = 1, .impulse_scale = 0 };
    if (pass == .relax) return .{ .bias = 0, .mass_scale = 1, .impulse_scale = 0 };
    return .{
        .bias = @max(c.softness.bias_rate * @min(0, s + step.slop), -step.max_push),
        .mass_scale = c.softness.mass_scale,
        .impulse_scale = c.softness.impulse_scale,
    };
}

/// One pass: the normal and then friction, Box2D v3's order - the friction
/// a point may have depends on how hard it is pressed, and that is best
/// known once the normal has had its say.
///
/// `pass` is `comptime`, so each kind is compiled as its own function, and
/// the loop that calls it a dozen times a step never asks which it is.
pub fn solve(c: *Constraint, a: *Body, b: *Body, step: Step, comptime pass: Pass) void {
    var v: Velocities = .of(a, b);

    if (pass == .relax and c.count == 2) {
        solvePair(c, &v, a, b, step);
    } else for (c.points[0..c.count]) |*p| {
        const t = targetOf(c, separationNow(c, p, a, b), step, pass);
        const vn = v.along(c.normal, p.ra, p.rb);
        var lambda = -p.normal_mass * t.mass_scale * (vn + t.bias) - t.impulse_scale * p.normal_impulse;

        // A contact can push and never pull: the accumulated impulse is
        // clamped, not this pass's, which is what lets a pass take back
        // what an earlier one over-applied - and what lets the relaxing
        // take back the push.
        const new_impulse = @max(p.normal_impulse + lambda, 0);
        lambda = new_impulse - p.normal_impulse;
        p.normal_impulse = new_impulse;
        p.max_normal_impulse = @max(p.max_normal_impulse, lambda);
        v.push(c, c.normal.scale(lambda), p.ra, p.rb);
    }

    for (c.points[0..c.count]) |*p| {
        const vt = v.along(c.tangent, p.ra, p.rb);
        var lambda = p.tangent_mass * (-vt);

        // Coulomb: friction can be no more than a fraction of how hard the
        // surfaces are pressed together, which is the normal impulse.
        const max_friction = c.friction * p.normal_impulse;
        const new_impulse = std.math.clamp(p.tangent_impulse + lambda, -max_friction, max_friction);
        lambda = new_impulse - p.tangent_impulse;
        p.tangent_impulse = new_impulse;
        v.push(c, c.tangent.scale(lambda), p.ra, p.rb);
    }

    v.store(c, a, b);
}

/// The normal impulses of a two-point contact, both at once, in the
/// relaxing pass.
///
/// **Why not one after the other**: a box resting on another touches it at
/// two corners, and solving the left corner and then the right leaves the
/// right with the last word in every pass. The left one is always a little
/// short, the box always leans a little that way, and in a stack the lean
/// adds up floor by floor until the stack falls. Solved together, the two
/// corners share the load the way the real ones do. The relaxing pass is
/// the one whose velocities the bodies keep, so it is the one that matters.
///
/// Together means a tiny linear complementarity problem: find impulses
/// `x >= 0` such that the speeds they leave, `vn = K x + b`, are at least
/// the targets, and each corner either pushes or is separating, never both.
/// With two unknowns there are four cases - both push, only the first,
/// only the second, neither - and the first that is consistent is the
/// answer. Box2D's block solver, due to Erin Catto and Dirk Gregorius.
fn solvePair(c: *Constraint, v: *Velocities, a: *const Body, b: *const Body, step: Step) void {
    const p1 = &c.points[0];
    const p2 = &c.points[1];
    const k = c.block;
    const a1 = p1.normal_impulse;
    const a2 = p2.normal_impulse;
    const t1 = targetOf(c, separationNow(c, p1, a, b), step, .relax);
    const t2 = targetOf(c, separationNow(c, p2, a, b), step, .relax);

    // The unknowns are the new totals, not this pass's change, so that
    // clamping them at zero clamps the totals: `b` moves to account for
    // what has already been applied.
    const b1 = v.along(c.normal, p1.ra, p1.rb) + t1.bias - (k.k11 * a1 + k.k12 * a2);
    const b2 = v.along(c.normal, p2.ra, p2.rb) + t2.bias - (k.k12 * a1 + k.k22 * a2);

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

    p1.normal_impulse = x1;
    p2.normal_impulse = x2;
    p1.max_normal_impulse = @max(p1.max_normal_impulse, x1 - a1);
    p2.max_normal_impulse = @max(p2.max_normal_impulse, x2 - a2);
    // Not `i1` and `i2`: those are type names in Zig - integers one and two
    // bits wide - and a variable may not shadow a type.
    const push1 = c.normal.scale(x1 - a1);
    const push2 = c.normal.scale(x2 - a2);
    v.push(c, push1, p1.ra, p1.rb);
    v.push(c, push2, p2.ra, p2.rb);
}

/// Bounce, after the substeps. Each point that was hit - approaching faster
/// than `threshold`, and pushed at some point in the step - is given what
/// it takes to leave at `restitution` times the speed it came in at. Slower
/// than the threshold is a rest, not a hit: a resting ball that bounced on
/// the speed gravity gives it each step would never lie still.
pub fn restitute(c: *Constraint, a: *Body, b: *Body, threshold: f32) void {
    if (c.restitution == 0) return;
    var v: Velocities = .of(a, b);
    for (c.points[0..c.count]) |*p| {
        if (p.approach > -threshold or p.max_normal_impulse == 0) continue;
        const vn = v.along(c.normal, p.ra, p.rb);
        var lambda = -p.normal_mass * (vn + c.restitution * p.approach);
        const new_impulse = @max(p.normal_impulse + lambda, 0);
        lambda = new_impulse - p.normal_impulse;
        p.normal_impulse = new_impulse;
        v.push(c, c.normal.scale(lambda), p.ra, p.rb);
    }
    v.store(c, a, b);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const test_step: Step = .{ .inv_h = 60, .slop = 0, .max_push = 3 };
/// Rigid, so a pass can be checked against numbers worked out by hand.
const rigid: Softness = .baumgarte(0.2, 60);

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
        var c = prepare(.{ .a = 1, .b = 2 }, &m, &wall, &mover, 0, 1, 0, e, null, rigid, rigid);
        for (0..4) |_| solve(&c, &wall, &mover, test_step, .solve);
        // Solving stops it dead; the bounce is its own pass, after.
        try testing.expectApproxEqAbs(@as(f32, 0), mover.linear_velocity.x, 1e-5);
        restitute(&c, &wall, &mover, 1);
        try testing.expectApproxEqAbs(-3 * e, mover.linear_velocity.x, 1e-5);
        // The wall was not written to.
        try testing.expect(wall.linear_velocity.eql(.zero));
    }
}

test "a sunk-in point is pushed out while solving, and the push is taken back relaxing" {
    var floor = ball(0, .zero);
    // Sunk a tenth into the floor and at rest.
    var m: Manifold = .{ .normal = .init(0, -1), .count = 1 };
    m.points[0] = .{ .point = .init(0, 0.5), .separation = -0.1, .id = 0 };
    var box = ball(1, .zero);
    var c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, 0.6, 0, null, rigid, rigid);
    for (0..4) |_| solve(&c, &floor, &box, test_step, .solve);
    // A fifth of the depth per step: 0.02 over a sixtieth, 1.2 m/s up.
    try testing.expectApproxEqAbs(@as(f32, -1.2), box.linear_velocity.y, 1e-4);
    for (0..2) |_| solve(&c, &floor, &box, test_step, .relax);
    try testing.expectApproxEqAbs(@as(f32, 0), box.linear_velocity.y, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0), c.points[0].normal_impulse, 1e-5);

    // Once it has moved out of the floor, the point measures itself from
    // where the bodies are now, and asks for nothing more.
    box.setTransform(.init(0, -0.2), 0);
    for (0..2) |_| solve(&c, &floor, &box, test_step, .solve);
    try testing.expectApproxEqAbs(@as(f32, 0), c.points[0].normal_impulse, 1e-5);
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
    var c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, 0.6, 0, null, rigid, rigid);
    try testing.expectEqual(@as(u32, 2), c.count);

    // One pass is enough: the block gets both corners exactly, where one
    // after the other would leave the second with more.
    solve(&c, &floor, &box, test_step, .relax);
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

    var c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, 0.5, 0, null, rigid, rigid);
    for (0..8) |_| solve(&c, &floor, &box, test_step, .solve);

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
    const c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, 0.6, 0, last, rigid, rigid);
    try testing.expectEqual(@as(f32, 1), c.points[0].normal_impulse);
    try testing.expectEqual(@as(f32, 2), c.points[1].normal_impulse);

    warmStart(&c, &floor, &box);
    // Three units of impulse along -y on a unit mass.
    try testing.expectApproxEqAbs(@as(f32, -3), box.linear_velocity.y, 1e-6);
}
