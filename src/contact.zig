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
//! **Position error is fed back as a velocity**, Baumgarte's way: a point
//! sunk in by more than the slop is given a bias velocity that pushes it
//! out over the next few steps. A separate position pass is the other
//! answer, and it is what Box2D does; this one is fewer lines and lets a
//! box sit a few thousandths into the floor, which nobody sees.
//!
//! Nothing in `solve` touches a body that cannot move, so two contacts
//! against one wall may be solved at once. See `solver`.

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
    /// How much of the penetration to remove per step, as a fraction.
    baumgarte: f32,
    /// One over the step, to turn a distance into a velocity.
    inv_dt: f32,
    /// How deep a point may sit before it is pushed. Some is necessary:
    /// pushing to exactly zero makes a contact flicker on and off.
    slop: f32,
    /// Slower than this, nothing bounces. Stops a resting ball from
    /// twitching on the tiny approach speed gravity gives it each step.
    restitution_threshold: f32,
    /// The most the position term may ask for.
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
    /// The velocity the point is asked to reach along the normal, from
    /// restitution or from position correction, whichever is larger.
    bias: f32,
    normal_impulse: f32,
    tangent_impulse: f32,
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

    pub fn pointSlice(self: *const Constraint) []const PointConstraint {
        return self.points[0..self.count];
    }
};

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
        const restitution_bias: f32 = if (vn < -bias.restitution_threshold) -c.restitution * vn else 0;
        const position_bias = @min(
            -bias.baumgarte * bias.inv_dt * @min(0, mp.separation + bias.slop),
            bias.max_push_speed,
        );

        var point: PointConstraint = .{
            .ra = ra,
            .rb = rb,
            .normal_mass = if (k_normal > 0) 1 / k_normal else 0,
            .tangent_mass = if (k_tangent > 0) 1 / k_tangent else 0,
            .bias = @max(restitution_bias, position_bias),
            .normal_impulse = 0,
            .tangent_impulse = 0,
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
    return c;
}

/// Apply last step's impulses before the first pass.
pub fn warmStart(c: *const Constraint, a: *Body, b: *Body) void {
    var va = a.linear_velocity;
    var wa = a.angular_velocity;
    var vb = b.linear_velocity;
    var wb = b.angular_velocity;

    for (c.pointSlice()) |p| {
        const impulse = c.normal.scale(p.normal_impulse).add(c.tangent.scale(p.tangent_impulse));
        va = va.mulAdd(impulse, -c.inv_mass_a);
        wa -= c.inv_inertia_a * cross(p.ra, impulse);
        vb = vb.mulAdd(impulse, c.inv_mass_b);
        wb += c.inv_inertia_b * cross(p.rb, impulse);
    }
    store(c, a, b, va, wa, vb, wb);
}

/// One pass. Friction first and then the normal, Box2D's order: the normal
/// impulse is the more important and going last lets it have the final say.
pub fn solve(c: *Constraint, a: *Body, b: *Body) void {
    var va = a.linear_velocity;
    var wa = a.angular_velocity;
    var vb = b.linear_velocity;
    var wb = b.angular_velocity;

    for (c.points[0..c.count]) |*p| {
        const dv = vb.add(crossSV(wb, p.rb)).sub(va).sub(crossSV(wa, p.ra));
        const vt = dv.dot(c.tangent);
        var lambda = p.tangent_mass * (-vt);

        // Coulomb: friction can be no more than a fraction of how hard the
        // surfaces are pressed together, which is the normal impulse.
        const max_friction = c.friction * p.normal_impulse;
        const new_impulse = std.math.clamp(p.tangent_impulse + lambda, -max_friction, max_friction);
        lambda = new_impulse - p.tangent_impulse;
        p.tangent_impulse = new_impulse;

        const impulse = c.tangent.scale(lambda);
        va = va.mulAdd(impulse, -c.inv_mass_a);
        wa -= c.inv_inertia_a * cross(p.ra, impulse);
        vb = vb.mulAdd(impulse, c.inv_mass_b);
        wb += c.inv_inertia_b * cross(p.rb, impulse);
    }

    for (c.points[0..c.count]) |*p| {
        const dv = vb.add(crossSV(wb, p.rb)).sub(va).sub(crossSV(wa, p.ra));
        const vn = dv.dot(c.normal);
        var lambda = -p.normal_mass * (vn - p.bias);

        // A contact can push and never pull: the accumulated impulse is
        // clamped, not this pass's, which is what lets a pass take back
        // what an earlier one over-applied.
        const new_impulse = @max(p.normal_impulse + lambda, 0);
        lambda = new_impulse - p.normal_impulse;
        p.normal_impulse = new_impulse;

        const impulse = c.normal.scale(lambda);
        va = va.mulAdd(impulse, -c.inv_mass_a);
        wa -= c.inv_inertia_a * cross(p.ra, impulse);
        vb = vb.mulAdd(impulse, c.inv_mass_b);
        wb += c.inv_inertia_b * cross(p.rb, impulse);
    }

    store(c, a, b, va, wa, vb, wb);
}

/// Write the velocities back - only to bodies that can move. A body with
/// no inverse mass had nothing added to its copy, and writing the same
/// number back would still be a write from two jobs at once.
inline fn store(c: *const Constraint, a: *Body, b: *Body, va: Vec2, wa: f32, vb: Vec2, wb: f32) void {
    if (c.inv_mass_a != 0) {
        a.linear_velocity = va;
        a.angular_velocity = wa;
    }
    if (c.inv_mass_b != 0) {
        b.linear_velocity = vb;
        b.angular_velocity = wb;
    }
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
        for (0..4) |_| solve(&c, &wall, &mover);
        try testing.expectApproxEqAbs(-3 * e, mover.linear_velocity.x, 1e-5);
        // The wall was not written to.
        try testing.expect(wall.linear_velocity.eql(.zero));
    }
}

test "friction slows a sliding box and cannot exceed its share of the normal impulse" {
    var floor = ball(0, .zero);
    var box = ball(1, .init(4, 1));
    // The floor is below (+y); the normal from floor to box points up (-y).
    var m: Manifold = .{ .normal = .init(0, -1), .count = 1 };
    m.points[0] = .{ .point = .init(0, 0.5), .separation = 0, .id = 0 };

    const rough: shape.Material = .{ .friction = 0.5 };
    var c = prepare(.{ .a = 1, .b = 2 }, &m, &floor, &box, 0, 1, rough, rough, null, no_bias);
    for (0..8) |_| solve(&c, &floor, &box);

    // The downward speed of 1 is stopped, which took a normal impulse of 1;
    // friction may then take at most 0.5 off the sideways speed.
    try testing.expectApproxEqAbs(@as(f32, 0), box.linear_velocity.y, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 3.5), box.linear_velocity.x, 1e-4);
}

test "warm starting picks up the impulse by id, not by position" {
    var floor = ball(0, .zero);
    var box = ball(1, .zero);
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
