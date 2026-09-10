// SPDX-License-Identifier: BSD-2-Clause

//! Joints: two bodies held to each other, or one body pulled to a point.
//!
//! ```zig
//! // A door on a hinge, swinging a quarter turn each way.
//! const hinge = try world.createJoint(.{ .revolute = .{
//!     .body_a = frame,
//!     .body_b = door,
//!     .anchor = .init(400, 300),
//!     .limit = .{ .lower = -std.math.pi / 2.0, .upper = std.math.pi / 2.0 },
//! } });
//!
//! // A crate on a rope: never further than three metres, never pushed.
//! _ = try world.createJoint(.{ .distance = .{
//!     .body_a = hook,
//!     .body_b = crate,
//!     .anchor_a = hook_point,
//!     .anchor_b = crate_top,
//!     .spring = .slack,
//!     .max_length = 300,
//! } });
//!
//! // The motor is a field; a throttle writes it every frame.
//! world.joint(hinge).?.kind.revolute.motor = .{ .speed = 2, .max_torque = 500 };
//! ```
//!
//! | Joint | What it holds | What it grows |
//! | --- | --- | --- |
//! | `distance` | two points a length apart | a rod, a spring, or a rope |
//! | `revolute` | two points together | a hinge: limits, a motor, a spring |
//! | `prismatic` | one body sliding along a line on the other, not turning | a piston, a lift: limits, a motor, a spring |
//! | `weld` | two bodies in one pose | rigid, or bending on a spring |
//! | `mouse` | a point of one body towards a target | dragging with the pointer |
//! | `wheel` | a wheel on a sprung axle | a car: suspension, limits, a motor |
//!
//! **Anchors are given in the world, once.** A definition says where the
//! pin goes as a point in the world, now, and the joint works out where
//! that is on each body and keeps it there. That is how a game places one -
//! the hinge is at the door's corner - and it means the pose the bodies are
//! in when the joint is made is its rest pose: the zero of a hinge's angle,
//! the length of a rod. Nothing is left to get wrong about whose frame a
//! number was in.
//!
//! **Solved beside the contacts, and the same way.** A joint is a few
//! velocity constraints between two bodies, solved by sequential impulses
//! and warm-started from what it pushed the substep before, exactly as a
//! contact is. The colouring treats it the same too: joints get colours of
//! their own, a colour's joints are solved at once, and every pass does the
//! joints colour by colour and then the contacts. The answer does not
//! depend on how many cores there are, for the same reason the contacts'
//! does not.
//!
//! **Measured again every substep.** A joint's lever arms turn with its
//! bodies, so `prepare` runs before the first substep and again after every
//! move: each pass sees the joint as it is, not as it was when the step
//! began. That, more than anything, is what lets a light chain hold a
//! heavy weight: see `World.step`, and the scenes under load at the end of
//! `joint_test`.
//!
//! **Drift is taken back the way a stiff spring would take it back**,
//! heavily damped, then taken back out of the velocity by a relaxing pass -
//! because the plainer way, a fixed fraction of the drift fed back as a
//! velocity, pumps energy into a light chain holding a heavy weight until
//! it flies apart. `Step.rigid` has the reasons, `Softness` the arithmetic,
//! `Settings.joint_hertz` the knob.
//!
//! **Springs are soft constraints**, Erin Catto's trick ("Soft
//! Constraints", GDC 2011, and Box2D since): a spring of a given frequency
//! and damping ratio is a constraint that is allowed to be violated by
//! exactly as much as the spring would stretch, so it is solved in the same
//! loop as everything rigid and cannot blow up however stiff it is set.
//! Given in hertz and not in newtons per metre, because a frequency means
//! the same for a crate as for a car, and a stiffness has to be retuned
//! whenever a mass changes.
//!
//! **Two bodies held by a joint do not collide with each other** unless the
//! definition says `collide_connected`. A joint usually holds two shapes
//! that overlap on purpose - an arm in its shoulder, a wheel in its arch -
//! and the contact would fight the joint for ever.

const std = @import("std");
const testing = std.testing;
const id = @import("fluxion_id");

const geometry = @import("geometry.zig");
const Vec2 = geometry.Vec2;
const cross = geometry.cross;
const crossSV = geometry.crossSV;
const Body = @import("body.zig");
const BodyId = Body.Id;

/// What names a joint. See `World.createJoint`.
pub const Id = id.Handle(Joint);

// -------------------------------------------------------------------------
// The pieces a joint is built from
// -------------------------------------------------------------------------

/// A spring, as a frequency and a damping ratio.
pub const Spring = struct {
    /// How many times a second it would bounce if nothing damped it. Zero
    /// is a spring that pushes nothing at all.
    hertz: f32 = 0,
    /// Zero bounces for ever, one comes to rest as fast as it can without
    /// overshooting, more than one creeps.
    damping_ratio: f32 = 0,

    /// Pushes nothing: a distance joint with a slack spring is held only by
    /// its limits, which is a rope.
    pub const slack: Spring = .{};

    /// As the solver steps it, `h` seconds at a time.
    pub fn softness(self: Spring, h: f32) Softness {
        return .of(self.hertz, self.damping_ratio, h);
    }
};

/// How far a joint may go each way from where it was made: radians for a
/// hinge, a distance along the axis for a slider or a wheel.
pub const Limit = struct {
    lower: f32,
    upper: f32,
};

/// Turns B against A.
pub const AngularMotor = struct {
    /// Radians per second, B relative to A. Positive turns `+x` towards
    /// `+y`, which is clockwise on this screen - and rolls a wheel on the
    /// ground towards `+x`.
    speed: f32 = 0,
    /// The most torque it may use to get there. What makes a motor a motor
    /// and not a gearbox: a heavy load slows it down.
    max_torque: f32,
};

/// Slides B along its axis.
pub const LinearMotor = struct {
    /// Units per second along the axis.
    speed: f32 = 0,
    max_force: f32,
};

// -------------------------------------------------------------------------
// Definitions
// -------------------------------------------------------------------------

/// What `World.createJoint` takes: one definition per kind of joint.
///
/// Every point in a definition is in the world, now; see the module comment.
pub const Def = union(enum) {
    distance: DistanceDef,
    revolute: RevoluteDef,
    prismatic: PrismaticDef,
    weld: WeldDef,
    mouse: MouseDef,
    wheel: WheelDef,

    /// The two bodies. A mouse joint's one body is both.
    pub fn bodies(self: Def) [2]BodyId {
        return switch (self) {
            .mouse => |d| .{ d.body, d.body },
            // `inline else` makes one prong per remaining kind at compile
            // time, so `d` has that kind's type in each and the field access
            // is checked for every one of them. One line for five kinds, and
            // a sixth without `body_a` would not compile.
            inline else => |d| .{ d.body_a, d.body_b },
        };
    }
};

/// Two points a length apart: a rod, a spring, or a rope.
///
/// ```zig
/// .{ .distance = .{ .body_a = a, .body_b = b, .anchor_a = p, .anchor_b = q } }                          // a rod
/// .{ .distance = .{ ..., .spring = .{ .hertz = 3, .damping_ratio = 0.2 } } }                             // a spring
/// .{ .distance = .{ ..., .spring = .slack, .max_length = 300 } }                                         // a rope
/// ```
pub const DistanceDef = struct {
    body_a: BodyId,
    body_b: BodyId,
    anchor_a: Vec2,
    anchor_b: Vec2,
    /// What it holds, or springs towards. Null is how far apart the anchors
    /// are now.
    length: ?f32 = null,
    /// Null holds `length` rigidly. A spring lets it stretch and squash
    /// about `length`, as far as the limits allow.
    spring: ?Spring = null,
    /// With a spring, the shortest and longest it may ever be.
    min_length: f32 = 0,
    max_length: f32 = std.math.inf(f32),
    collide_connected: bool = false,
    user_data: u64 = 0,
};

/// A pin. B turns about it, relative to A.
pub const RevoluteDef = struct {
    body_a: BodyId,
    body_b: BodyId,
    anchor: Vec2,
    /// Radians either side of the angle the bodies are at now.
    limit: ?Limit = null,
    motor: ?AngularMotor = null,
    /// Pulls back towards the angle the bodies are at now: a door closer,
    /// the stiffness of a ragdoll's joints.
    spring: ?Spring = null,
    collide_connected: bool = false,
    user_data: u64 = 0,
};

/// A slider. B moves along `axis`, which is fixed to A and turns with it,
/// and does not turn relative to A.
pub const PrismaticDef = struct {
    body_a: BodyId,
    body_b: BodyId,
    anchor: Vec2,
    /// The direction B slides in, in the world, now. Any length but zero.
    axis: Vec2,
    /// How far along the axis each way from where B is now.
    limit: ?Limit = null,
    motor: ?LinearMotor = null,
    /// Pulls back towards where B is now along the axis.
    spring: ?Spring = null,
    collide_connected: bool = false,
    user_data: u64 = 0,
};

/// Glue. Two bodies held in the pose they are in now.
pub const WeldDef = struct {
    body_a: BodyId,
    body_b: BodyId,
    /// Where the two are joined. For a rigid weld it hardly matters; a
    /// springy one bends about it.
    anchor: Vec2,
    /// Null for rigid.
    linear_spring: ?Spring = null,
    angular_spring: ?Spring = null,
    collide_connected: bool = false,
    user_data: u64 = 0,
};

/// A point of one body pulled towards a target on a spring. What dragging
/// something with the pointer is.
pub const MouseDef = struct {
    body: BodyId,
    /// Where it is pulled to. The point of the body under the target now is
    /// the point that is pulled; move the target every frame through
    /// `Joint.kind.mouse.target`.
    target: Vec2,
    spring: Spring = .{ .hertz = 5, .damping_ratio = 0.7 },
    /// The most force it may pull with. Null is a thousand times the body's
    /// mass in metres per second squared - a hundred times gravity, enough
    /// to drag anything and not enough to throw it through a wall.
    max_force: ?f32 = null,
    user_data: u64 = 0,
};

/// A wheel on a car: B turns freely about `anchor` and slides along `axis`
/// on a spring, the suspension.
pub const WheelDef = struct {
    /// The chassis.
    body_a: BodyId,
    /// The wheel.
    body_b: BodyId,
    /// The axle, usually the wheel's centre.
    anchor: Vec2,
    /// The way the suspension travels, in the world, now: up, for a car on
    /// level ground, which is `(0, -1)` on this screen.
    axis: Vec2,
    /// Null is a rigid axle, which is a hinge.
    spring: ?Spring = .{ .hertz = 4, .damping_ratio = 0.7 },
    /// How far the suspension may travel along the axis each way.
    limit: ?Limit = null,
    motor: ?AngularMotor = null,
    collide_connected: bool = false,
    user_data: u64 = 0,
};

// -------------------------------------------------------------------------
// The joint
// -------------------------------------------------------------------------

/// A joint as the world keeps it. Its parameters - a motor's speed, a
/// mouse joint's target, a limit - are fields, written directly between
/// steps.
pub const Joint = struct {
    /// The body the joint is measured from.
    body_a: BodyId,
    /// The body it moves. A mouse joint's one body is both.
    body_b: BodyId,
    collide_connected: bool,
    /// Yours. The joint never reads it.
    user_data: u64,
    kind: Kind,

    /// What the solver may push each body by, set every step before
    /// `prepare`. See `massesOf`.
    masses: Masses = .{},

    pub const Kind = union(enum) {
        distance: Distance,
        revolute: Revolute,
        prismatic: Prismatic,
        weld: Weld,
        mouse: Mouse,
        wheel: Wheel,
    };

    /// A joint from its definition, with `a` and `b` its two bodies where
    /// they are now.
    pub fn init(def: Def, a: *const Body, b: *const Body) Joint {
        return switch (def) {
            .distance => |d| base(d, .{ .distance = .{
                .local_anchor_a = a.transform.unapply(d.anchor_a),
                .local_anchor_b = b.transform.unapply(d.anchor_b),
                .length = d.length orelse d.anchor_a.dist(d.anchor_b),
                .spring = d.spring,
                .min_length = d.min_length,
                .max_length = d.max_length,
            } }),
            .revolute => |d| base(d, .{ .revolute = .{
                .local_anchor_a = a.transform.unapply(d.anchor),
                .local_anchor_b = b.transform.unapply(d.anchor),
                .reference_angle = b.angle - a.angle,
                .limit = d.limit,
                .motor = d.motor,
                .spring = d.spring,
            } }),
            .prismatic => |d| base(d, .{ .prismatic = .{
                .local_anchor_a = a.transform.unapply(d.anchor),
                .local_anchor_b = b.transform.unapply(d.anchor),
                .local_axis_a = a.transform.q.invRotate(d.axis.norm()),
                .reference_angle = b.angle - a.angle,
                .limit = d.limit,
                .motor = d.motor,
                .spring = d.spring,
            } }),
            .weld => |d| base(d, .{ .weld = .{
                .local_anchor_a = a.transform.unapply(d.anchor),
                .local_anchor_b = b.transform.unapply(d.anchor),
                .reference_angle = b.angle - a.angle,
                .linear_spring = d.linear_spring,
                .angular_spring = d.angular_spring,
            } }),
            .mouse => |d| .{
                .body_a = d.body,
                .body_b = d.body,
                .collide_connected = true,
                .user_data = d.user_data,
                .kind = .{ .mouse = .{
                    .target = d.target,
                    .local_anchor = b.transform.unapply(d.target),
                    .spring = d.spring,
                    .max_force = d.max_force,
                } },
            },
            .wheel => |d| base(d, .{ .wheel = .{
                .local_anchor_a = a.transform.unapply(d.anchor),
                .local_anchor_b = b.transform.unapply(d.anchor),
                .local_axis_a = a.transform.q.invRotate(d.axis.norm()),
                .spring = d.spring,
                .limit = d.limit,
                .motor = d.motor,
            } }),
        };
    }

    /// The fields every two-body definition shares, around a kind.
    fn base(def: anytype, kind: Kind) Joint {
        return .{
            .body_a = def.body_a,
            .body_b = def.body_b,
            .collide_connected = def.collide_connected,
            .user_data = def.user_data,
            .kind = kind,
        };
    }

    /// Where the joint is fixed to each body, in that body's frame. For a
    /// mouse joint the first is the target, in the world.
    pub fn localAnchors(self: *const Joint) [2]Vec2 {
        return switch (self.kind) {
            .mouse => |*m| .{ m.target, m.local_anchor },
            inline else => |*k| .{ k.local_anchor_a, k.local_anchor_b },
        };
    }

    /// The force and torque the joint put on body B over the last step,
    /// worked out from its impulses. A breakable joint is this compared
    /// with a limit, and `World.destroyJoint` when it is exceeded.
    pub fn reaction(self: *const Joint, inv_dt: f32) Reaction {
        return switch (self.kind) {
            inline else => |*k| k.reaction(inv_dt),
        };
    }
};

pub const Reaction = struct {
    force: Vec2,
    torque: f32,
};

/// What a joint may push each body by: its inverse mass and inertia, or
/// zeros for a body that cannot be moved - static, kinematic, or the half
/// of a mouse joint that is not there.
///
/// Worked out on one thread before anything is solved, because the
/// colouring needs to know which bodies a joint writes, and it writes
/// exactly the ones with a non-zero inverse mass here.
pub fn massesOf(j: *const Joint, a: *const Body, b: *const Body) Masses {
    if (j.kind == .mouse) return .{ .mb = b.inv_mass, .ib = b.inv_inertia };
    return .{ .ma = a.inv_mass, .mb = b.inv_mass, .ia = a.inv_inertia, .ib = b.inv_inertia };
}

/// Whether a joint is asking to move its bodies, whatever else is going
/// on: a motor with a speed, or a mouse joint whose target is somewhere
/// other than the point it pulls. Such a joint keeps its bodies awake, and
/// wakes them if they slept - which is how writing a throttle into a car's
/// motors gets a car that had come to rest going again.
pub fn isDriven(j: *const Joint, b: *const Body, slop: f32) bool {
    return switch (j.kind) {
        .revolute => |*r| if (r.motor) |m| m.speed != 0 else false,
        .prismatic => |*p| if (p.motor) |m| m.speed != 0 else false,
        .wheel => |*w| if (w.motor) |m| m.speed != 0 else false,
        .mouse => |*m| b.transform.apply(m.local_anchor).distSq(m.target) > slop * slop,
        .distance, .weld => false,
    };
}

/// A joint as the colouring sees it: which slot of the world's table, and
/// the two bodies it writes. See `solver.Colouring`.
pub const Ref = struct {
    joint: u32,
    body_a: u32,
    body_b: u32,
    inv_mass_a: f32,
    inv_mass_b: f32,
};

// -------------------------------------------------------------------------
// Solving, for every kind
// -------------------------------------------------------------------------

/// What a substep tells every joint, worked out once per step.
pub const Step = struct {
    /// The substep, not the whole step: every impulse a joint keeps is the
    /// impulse of one substep. See `World.step`.
    dt: f32,
    inv_dt: f32,
    /// How every rigid constraint takes back its drift: the joint spring
    /// while solving, `Softness.rigid` while relaxing. See `rigid`.
    stiffness: Softness,
    /// Shorter than this, a distance has no direction. World units.
    slop: f32,
    /// For the mouse joint's default force.
    units_per_metre: f32,

    /// The impulse a rigid constraint along one line asks for this pass:
    /// `mass` is its effective mass, `cdot` how fast it is being broken, `c`
    /// by how much it already is, and `total` what it has pushed so far.
    ///
    /// **Rigid, but with a spring's damping in how it takes back drift.**
    /// The plain way - ask for a fixed fraction of the drift back, as a
    /// velocity - puts the correction into the bodies as real speed, and
    /// when the solver has not converged each correction overshoots and the
    /// next is fed from the overshoot: a ball six times a link's mass on ten
    /// links pulled its pins 90 pixels apart in five seconds that way. So
    /// the drift is taken back the way a very stiff, heavily damped spring
    /// would (`Settings.joint_hertz`); see `Softness` for why that cannot
    /// add energy.
    ///
    /// **And never capped.** A contact's push is capped, so a body made
    /// inside another does not leave at the speed of a bullet; a joint's is
    /// not, because a joint that has been pulled apart - a chain yanked by
    /// the weight on its end - has to be able to come back together faster
    /// than the weight is pulling it apart. Capped, it could not, and the
    /// chain came apart for good. The spring's damping is what keeps an
    /// uncapped correction gentle.
    fn rigid(self: Step, mass: f32, cdot: f32, c: f32, total: f32) f32 {
        return self.stiffness.impulse(mass, cdot, c, total);
    }

    /// The same about an angle.
    fn rigidAngle(self: Step, mass: f32, cdot: f32, c: f32, total: f32) f32 {
        return self.stiffness.impulse(mass, cdot, c, total);
    }

    /// The same at a point, both directions at once. `matrix` is the
    /// point's effective mass matrix, not yet inverted.
    fn rigidPoint(self: Step, matrix: Sym22, cdot: Vec2, c: Vec2, total: Vec2) Vec2 {
        const s = self.stiffness;
        return matrix.solve(cdot.add(c.scale(s.bias_rate))).scale(-s.mass_scale).sub(total.scale(s.impulse_scale));
    }

    /// A one-sided limit. Short of the stop by `c > 0`, the gap may be
    /// closed this step and no more, exactly - so the joint arrives at its
    /// stop rather than going through and bouncing back, which is the
    /// speculative trick Box2D v3 plays. Past it, the drift is taken back
    /// like any other. The caller clamps the total at zero: a stop pushes
    /// and never pulls.
    fn limit(self: Step, mass: f32, cdot: f32, c: f32, total: f32) f32 {
        if (c > 0) return -mass * (cdot + c * self.inv_dt);
        return self.rigid(mass, cdot, c, total);
    }

    fn limitAngle(self: Step, mass: f32, cdot: f32, c: f32, total: f32) f32 {
        if (c > 0) return -mass * (cdot + c * self.inv_dt);
        return self.rigidAngle(mass, cdot, c, total);
    }
};

pub const Masses = struct {
    ma: f32 = 0,
    mb: f32 = 0,
    ia: f32 = 0,
    ib: f32 = 0,
};

/// The two bodies' velocities, copied out for a solve and written back
/// after, so a joint's passes work on locals and not through two pointers.
pub const Velocities = struct {
    va: Vec2,
    wa: f32,
    vb: Vec2,
    wb: f32,

    fn of(a: *const Body, b: *const Body) Velocities {
        return .{ .va = a.linear_velocity, .wa = a.angular_velocity, .vb = b.linear_velocity, .wb = b.angular_velocity };
    }

    /// How fast the point `rb` from B's centre moves away from `ra` from A's.
    inline fn relative(v: Velocities, ra: Vec2, rb: Vec2) Vec2 {
        return v.vb.add(crossSV(v.wb, rb)).sub(v.va).sub(crossSV(v.wa, ra));
    }

    /// An impulse `p` at the anchors: taken from A, given to B.
    inline fn push(v: *Velocities, m: Masses, p: Vec2, ra: Vec2, rb: Vec2) void {
        v.va = v.va.mulAdd(p, -m.ma);
        v.wa -= m.ia * cross(ra, p);
        v.vb = v.vb.mulAdd(p, m.mb);
        v.wb += m.ib * cross(rb, p);
    }

    /// An impulse `p` whose turning effect on A and on B has already been
    /// worked out: `la` and `lb`. What a slider's constraints are, whose
    /// lever arms are not simply the anchors'.
    inline fn pushWith(v: *Velocities, m: Masses, p: Vec2, la: f32, lb: f32) void {
        v.va = v.va.mulAdd(p, -m.ma);
        v.wa -= m.ia * la;
        v.vb = v.vb.mulAdd(p, m.mb);
        v.wb += m.ib * lb;
    }

    /// An angular impulse: taken from A, given to B.
    inline fn turn(v: *Velocities, m: Masses, impulse: f32) void {
        v.wa -= m.ia * impulse;
        v.wb += m.ib * impulse;
    }
};

/// A spring as the solver uses it. See `Softness` itself, which the
/// contacts share.
pub const Softness = @import("Softness.zig");

/// A symmetric 2x2 matrix: what one unit of impulse at a point does to the
/// velocity of that point, in each direction.
const Sym22 = struct {
    a11: f32 = 0,
    a12: f32 = 0,
    a22: f32 = 0,

    /// For a point constraint between `ra` from A's centre and `rb` from
    /// B's: the two masses, and each body's inertia at its lever arm.
    fn point(m: Masses, ra: Vec2, rb: Vec2) Sym22 {
        return .{
            .a11 = m.ma + m.mb + m.ia * ra.y * ra.y + m.ib * rb.y * rb.y,
            .a12 = -m.ia * ra.x * ra.y - m.ib * rb.x * rb.y,
            .a22 = m.ma + m.mb + m.ia * ra.x * ra.x + m.ib * rb.x * rb.x,
        };
    }

    /// `x` with `self x = b`, by Cramer's rule. Zero where the matrix is
    /// singular - two bodies neither of which can move - rather than
    /// infinity.
    fn solve(self: Sym22, b: Vec2) Vec2 {
        const det = self.a11 * self.a22 - self.a12 * self.a12;
        if (det == 0) return .zero;
        const inv = 1 / det;
        return .{
            .x = inv * (self.a22 * b.x - self.a12 * b.y),
            .y = inv * (self.a11 * b.y - self.a12 * b.x),
        };
    }
};

/// One over what a unit impulse along a line does to the speed along it,
/// with lever arms `la` and `lb`. Zero when nothing can move.
fn axialMass(m: Masses, la: f32, lb: f32) f32 {
    const k = m.ma + m.mb + m.ia * la * la + m.ib * lb * lb;
    return if (k > 0) 1 / k else 0;
}

/// From a body's centre of mass to a point fixed on it, in the world.
fn arm(b: *const Body, local_anchor: Vec2) Vec2 {
    return b.transform.q.rotate(local_anchor.sub(b.local_center));
}

/// Add `impulse` to an accumulated `total` that must stay in
/// `[lower, upper]`, and hand back how much was actually added.
///
/// The accumulated impulse is what is clamped, not this pass's, which is
/// what lets a later pass take back what an earlier one overdid. The same
/// rule as a contact's `max(0, ...)`.
fn clampedAdd(total: *f32, impulse: f32, lower: f32, upper: f32) f32 {
    const old = total.*;
    total.* = std.math.clamp(old + impulse, lower, upper);
    return total.* - old;
}

/// Get the joint ready for this step's passes: lever arms, effective
/// masses, errors, and the springs at this `dt`. `j.masses` must be set.
pub fn prepare(j: *Joint, a: *const Body, b: *const Body, step: Step) void {
    switch (j.kind) {
        // `|*k|` captures a pointer to the payload rather than a copy, which
        // needs the union itself to be addressable - `j.kind` through a
        // pointer is. Each prong calls its own kind's `prepare`.
        inline else => |*k| k.prepare(j.masses, a, b, step),
    }
}

/// Apply what the joint pushed last step before this step's first pass.
pub fn warmStart(j: *const Joint, a: *Body, b: *Body) void {
    var v: Velocities = .of(a, b);
    switch (j.kind) {
        inline else => |*k| k.warmStart(j.masses, &v),
    }
    store(j, a, b, v);
}

/// One pass.
pub fn solve(j: *Joint, a: *Body, b: *Body, step: Step) void {
    var v: Velocities = .of(a, b);
    switch (j.kind) {
        inline else => |*k| k.solve(j.masses, &v, step),
    }
    store(j, a, b, v);
}

/// Write the velocities back to the bodies this joint may move, and only
/// those - see `contact.store` for why a body that cannot move must not be
/// written even with its own numbers.
inline fn store(j: *const Joint, a: *Body, b: *Body, v: Velocities) void {
    if (j.masses.ma != 0) {
        a.linear_velocity = v.va;
        a.angular_velocity = v.wa;
    }
    if (j.masses.mb != 0) {
        b.linear_velocity = v.vb;
        b.angular_velocity = v.wb;
    }
}

// -------------------------------------------------------------------------
// Distance
// -------------------------------------------------------------------------

pub const Distance = struct {
    local_anchor_a: Vec2,
    local_anchor_b: Vec2,
    length: f32,
    spring: ?Spring = null,
    min_length: f32 = 0,
    max_length: f32 = std.math.inf(f32),

    /// What was pushed, kept for the next step's warm start: along the
    /// axis by the rod or the spring, and by each limit.
    impulse: f32 = 0,
    lower_impulse: f32 = 0,
    upper_impulse: f32 = 0,

    // Prepared each step.
    ra: Vec2 = .zero,
    rb: Vec2 = .zero,
    axis: Vec2 = .zero,
    current: f32 = 0,
    axial_mass: f32 = 0,
    softness: Softness = .{},

    fn prepare(self: *Distance, m: Masses, a: *const Body, b: *const Body, step: Step) void {
        self.ra = arm(a, self.local_anchor_a);
        self.rb = arm(b, self.local_anchor_b);
        const d = b.center.add(self.rb).sub(a.center.add(self.ra));
        self.current = d.len();
        // Two anchors on top of each other have no line between them, and
        // a rod of no length is a pin: see `revolute`.
        self.axis = if (self.current > step.slop) d.scale(1 / self.current) else .zero;
        self.axial_mass = axialMass(m, cross(self.ra, self.axis), cross(self.rb, self.axis));
        self.softness = if (self.spring) |s| s.softness(step.dt) else .{};
        // A limit that is not there any more pushes nothing, including in
        // the warm start.
        if (self.spring == null or self.min_length <= 0) self.lower_impulse = 0;
        if (self.spring == null or !std.math.isFinite(self.max_length)) self.upper_impulse = 0;
    }

    fn warmStart(self: *const Distance, m: Masses, v: *Velocities) void {
        const total = self.impulse + self.lower_impulse - self.upper_impulse;
        v.push(m, self.axis.scale(total), self.ra, self.rb);
    }

    fn solve(self: *Distance, m: Masses, v: *Velocities, step: Step) void {
        if (self.spring == null) {
            // A rod: the length, pushing and pulling.
            const cdot = self.axis.dot(v.relative(self.ra, self.rb));
            const impulse = step.rigid(self.axial_mass, cdot, self.current - self.length, self.impulse);
            self.impulse += impulse;
            v.push(m, self.axis.scale(impulse), self.ra, self.rb);
            return;
        }

        {
            const cdot = self.axis.dot(v.relative(self.ra, self.rb));
            const impulse = self.softness.impulse(self.axial_mass, cdot, self.current - self.length, self.impulse);
            self.impulse += impulse;
            v.push(m, self.axis.scale(impulse), self.ra, self.rb);
        }

        if (self.min_length > 0) {
            const cdot = self.axis.dot(v.relative(self.ra, self.rb));
            const wanted = step.limit(self.axial_mass, cdot, self.current - self.min_length, self.lower_impulse);
            const impulse = clampedAdd(&self.lower_impulse, wanted, 0, std.math.inf(f32));
            v.push(m, self.axis.scale(impulse), self.ra, self.rb);
        }

        if (std.math.isFinite(self.max_length)) {
            // Measured the other way, so the one-sided rule is the same: the
            // gap to the limit shrinks as the length grows.
            const cdot = -self.axis.dot(v.relative(self.ra, self.rb));
            const wanted = step.limit(self.axial_mass, cdot, self.max_length - self.current, self.upper_impulse);
            const impulse = clampedAdd(&self.upper_impulse, wanted, 0, std.math.inf(f32));
            v.push(m, self.axis.scale(-impulse), self.ra, self.rb);
        }
    }

    fn reaction(self: *const Distance, inv_dt: f32) Reaction {
        const total = self.impulse + self.lower_impulse - self.upper_impulse;
        return .{ .force = self.axis.scale(total * inv_dt), .torque = 0 };
    }
};

// -------------------------------------------------------------------------
// Revolute
// -------------------------------------------------------------------------

pub const Revolute = struct {
    local_anchor_a: Vec2,
    local_anchor_b: Vec2,
    /// B's angle less A's when the joint was made: the zero the limit and
    /// the spring measure from.
    reference_angle: f32,
    limit: ?Limit = null,
    motor: ?AngularMotor = null,
    spring: ?Spring = null,

    impulse: Vec2 = .zero,
    motor_impulse: f32 = 0,
    spring_impulse: f32 = 0,
    lower_impulse: f32 = 0,
    upper_impulse: f32 = 0,

    // Prepared each step.
    ra: Vec2 = .zero,
    rb: Vec2 = .zero,
    matrix: Sym22 = .{},
    separation: Vec2 = .zero,
    /// How far B has turned from where it started, relative to A, as of
    /// the start of the last step. Radians, and never wrapped: two turns
    /// is `4 pi`, so a limit may allow more than one.
    angle: f32 = 0,
    axial_mass: f32 = 0,
    softness: Softness = .{},

    fn prepare(self: *Revolute, m: Masses, a: *const Body, b: *const Body, step: Step) void {
        self.ra = arm(a, self.local_anchor_a);
        self.rb = arm(b, self.local_anchor_b);
        self.matrix = .point(m, self.ra, self.rb);
        self.separation = b.center.add(self.rb).sub(a.center.add(self.ra));
        self.angle = b.angle - a.angle - self.reference_angle;
        self.axial_mass = if (m.ia + m.ib > 0) 1 / (m.ia + m.ib) else 0;
        self.softness = if (self.spring) |s| s.softness(step.dt) else .{};

        // What has been switched off since last step pushes nothing now.
        if (self.limit == null) {
            self.lower_impulse = 0;
            self.upper_impulse = 0;
        }
        if (self.motor == null) self.motor_impulse = 0;
        if (self.spring == null) self.spring_impulse = 0;
    }

    fn warmStart(self: *const Revolute, m: Masses, v: *Velocities) void {
        v.push(m, self.impulse, self.ra, self.rb);
        v.turn(m, self.motor_impulse + self.spring_impulse + self.lower_impulse - self.upper_impulse);
    }

    /// The spring, the motor and the limits, then the pin. The pin goes last
    /// because it is the one that matters: a pass ends with the bodies held
    /// together, whatever the motor wanted.
    fn solve(self: *Revolute, m: Masses, v: *Velocities, step: Step) void {
        if (self.axial_mass > 0) {
            if (self.spring != null) {
                const impulse = self.softness.impulse(self.axial_mass, v.wb - v.wa, self.angle, self.spring_impulse);
                self.spring_impulse += impulse;
                v.turn(m, impulse);
            }
            if (self.motor) |motor| {
                const most = motor.max_torque * step.dt;
                const wanted = -self.axial_mass * (v.wb - v.wa - motor.speed);
                v.turn(m, clampedAdd(&self.motor_impulse, wanted, -most, most));
            }
            if (self.limit) |limit| {
                const lower = @min(limit.lower, limit.upper);
                const upper = @max(limit.lower, limit.upper);
                const up = step.limitAngle(self.axial_mass, v.wb - v.wa, self.angle - lower, self.lower_impulse);
                v.turn(m, clampedAdd(&self.lower_impulse, up, 0, std.math.inf(f32)));
                const down = step.limitAngle(self.axial_mass, v.wa - v.wb, upper - self.angle, self.upper_impulse);
                v.turn(m, -clampedAdd(&self.upper_impulse, down, 0, std.math.inf(f32)));
            }
        }

        const cdot = v.relative(self.ra, self.rb);
        const impulse = step.rigidPoint(self.matrix, cdot, self.separation, self.impulse);
        self.impulse = self.impulse.add(impulse);
        v.push(m, impulse, self.ra, self.rb);
    }

    fn reaction(self: *const Revolute, inv_dt: f32) Reaction {
        const axial = self.motor_impulse + self.spring_impulse + self.lower_impulse - self.upper_impulse;
        return .{ .force = self.impulse.scale(inv_dt), .torque = axial * inv_dt };
    }
};

// -------------------------------------------------------------------------
// Prismatic
// -------------------------------------------------------------------------

pub const Prismatic = struct {
    local_anchor_a: Vec2,
    local_anchor_b: Vec2,
    /// The direction B slides in, in A's frame. Unit length.
    local_axis_a: Vec2,
    reference_angle: f32,
    limit: ?Limit = null,
    motor: ?LinearMotor = null,
    spring: ?Spring = null,

    /// Across the axis, and against turning.
    impulse: Vec2 = .zero,
    motor_impulse: f32 = 0,
    spring_impulse: f32 = 0,
    lower_impulse: f32 = 0,
    upper_impulse: f32 = 0,

    // Prepared each step. `a1`, `a2` are the lever arms of a push along
    // the axis on A and B, and `s1`, `s2` of one across it: along the axis
    // the arm on A is measured from B's anchor, because A's axis turns with
    // A and sweeps B's anchor sideways as it does.
    axis: Vec2 = .zero,
    perp: Vec2 = .zero,
    a1: f32 = 0,
    a2: f32 = 0,
    s1: f32 = 0,
    s2: f32 = 0,
    axial_mass: f32 = 0,
    matrix: Sym22 = .{},
    /// How far along the axis B is from where it started.
    translation: f32 = 0,
    perp_error: f32 = 0,
    angle_error: f32 = 0,
    softness: Softness = .{},

    fn prepare(self: *Prismatic, m: Masses, a: *const Body, b: *const Body, step: Step) void {
        const ra = arm(a, self.local_anchor_a);
        const rb = arm(b, self.local_anchor_b);
        const d = b.center.add(rb).sub(a.center.add(ra));
        self.axis = a.transform.q.rotate(self.local_axis_a);
        self.perp = crossSV(1, self.axis);

        const da = d.add(ra);
        self.a1 = cross(da, self.axis);
        self.a2 = cross(rb, self.axis);
        self.s1 = cross(da, self.perp);
        self.s2 = cross(rb, self.perp);
        self.axial_mass = axialMass(m, self.a1, self.a2);

        // Across the axis and against turning are solved together, as one
        // two-by-two block, because each disturbs the other. Two bodies
        // that cannot turn leave the angular row empty; a one there keeps
        // the matrix invertible and the row does nothing.
        const k22 = m.ia + m.ib;
        self.matrix = .{
            .a11 = m.ma + m.mb + m.ia * self.s1 * self.s1 + m.ib * self.s2 * self.s2,
            .a12 = m.ia * self.s1 + m.ib * self.s2,
            .a22 = if (k22 == 0) 1 else k22,
        };

        self.translation = self.axis.dot(d);
        self.perp_error = self.perp.dot(d);
        self.angle_error = b.angle - a.angle - self.reference_angle;
        self.softness = if (self.spring) |s| s.softness(step.dt) else .{};

        if (self.limit == null) {
            self.lower_impulse = 0;
            self.upper_impulse = 0;
        }
        if (self.motor == null) self.motor_impulse = 0;
        if (self.spring == null) self.spring_impulse = 0;
    }

    fn axialSpeed(self: *const Prismatic, v: *const Velocities) f32 {
        return self.axis.dot(v.vb.sub(v.va)) + self.a2 * v.wb - self.a1 * v.wa;
    }

    fn pushAxial(self: *const Prismatic, m: Masses, v: *Velocities, impulse: f32) void {
        v.pushWith(m, self.axis.scale(impulse), impulse * self.a1, impulse * self.a2);
    }

    fn warmStart(self: *const Prismatic, m: Masses, v: *Velocities) void {
        const axial = self.motor_impulse + self.spring_impulse + self.lower_impulse - self.upper_impulse;
        const p = self.perp.scale(self.impulse.x).add(self.axis.scale(axial));
        const la = self.impulse.x * self.s1 + self.impulse.y + axial * self.a1;
        const lb = self.impulse.x * self.s2 + self.impulse.y + axial * self.a2;
        v.pushWith(m, p, la, lb);
    }

    fn solve(self: *Prismatic, m: Masses, v: *Velocities, step: Step) void {
        if (self.spring != null) {
            const impulse = self.softness.impulse(self.axial_mass, self.axialSpeed(v), self.translation, self.spring_impulse);
            self.spring_impulse += impulse;
            self.pushAxial(m, v, impulse);
        }
        if (self.motor) |motor| {
            const most = motor.max_force * step.dt;
            const wanted = -self.axial_mass * (self.axialSpeed(v) - motor.speed);
            self.pushAxial(m, v, clampedAdd(&self.motor_impulse, wanted, -most, most));
        }
        if (self.limit) |limit| {
            const lower = @min(limit.lower, limit.upper);
            const upper = @max(limit.lower, limit.upper);
            const out = step.limit(self.axial_mass, self.axialSpeed(v), self.translation - lower, self.lower_impulse);
            self.pushAxial(m, v, clampedAdd(&self.lower_impulse, out, 0, std.math.inf(f32)));
            const in = step.limit(self.axial_mass, -self.axialSpeed(v), upper - self.translation, self.upper_impulse);
            self.pushAxial(m, v, -clampedAdd(&self.upper_impulse, in, 0, std.math.inf(f32)));
        }

        // Across the axis and against turning, as one block, taking back
        // drift the way `Step.rigid` explains - written out here because
        // one row is a length and the other an angle.
        const s = step.stiffness;
        const cdot: Vec2 = .{
            .x = self.perp.dot(v.vb.sub(v.va)) + self.s2 * v.wb - self.s1 * v.wa,
            .y = v.wb - v.wa,
        };
        const bias: Vec2 = .{ .x = s.bias_rate * self.perp_error, .y = s.bias_rate * self.angle_error };
        const impulse = self.matrix.solve(cdot.add(bias)).scale(-s.mass_scale).sub(self.impulse.scale(s.impulse_scale));
        self.impulse = self.impulse.add(impulse);
        v.pushWith(m, self.perp.scale(impulse.x), impulse.x * self.s1 + impulse.y, impulse.x * self.s2 + impulse.y);
    }

    fn reaction(self: *const Prismatic, inv_dt: f32) Reaction {
        const axial = self.motor_impulse + self.spring_impulse + self.lower_impulse - self.upper_impulse;
        return .{
            .force = self.perp.scale(self.impulse.x).add(self.axis.scale(axial)).scale(inv_dt),
            .torque = self.impulse.y * inv_dt,
        };
    }
};

// -------------------------------------------------------------------------
// Weld
// -------------------------------------------------------------------------

pub const Weld = struct {
    local_anchor_a: Vec2,
    local_anchor_b: Vec2,
    reference_angle: f32,
    linear_spring: ?Spring = null,
    angular_spring: ?Spring = null,

    linear_impulse: Vec2 = .zero,
    angular_impulse: f32 = 0,

    // Prepared each step.
    ra: Vec2 = .zero,
    rb: Vec2 = .zero,
    matrix: Sym22 = .{},
    separation: Vec2 = .zero,
    angle_error: f32 = 0,
    axial_mass: f32 = 0,
    linear_softness: Softness = .{},
    angular_softness: Softness = .{},

    fn prepare(self: *Weld, m: Masses, a: *const Body, b: *const Body, step: Step) void {
        self.ra = arm(a, self.local_anchor_a);
        self.rb = arm(b, self.local_anchor_b);
        self.matrix = .point(m, self.ra, self.rb);
        self.separation = b.center.add(self.rb).sub(a.center.add(self.ra));
        self.angle_error = b.angle - a.angle - self.reference_angle;
        self.axial_mass = if (m.ia + m.ib > 0) 1 / (m.ia + m.ib) else 0;
        self.linear_softness = if (self.linear_spring) |s| s.softness(step.dt) else .{};
        self.angular_softness = if (self.angular_spring) |s| s.softness(step.dt) else .{};
    }

    fn warmStart(self: *const Weld, m: Masses, v: *Velocities) void {
        v.push(m, self.linear_impulse, self.ra, self.rb);
        v.turn(m, self.angular_impulse);
    }

    /// The angle, then the point: two small solves rather than one three by
    /// three, which is Box2D v3's choice. It bends a little between two
    /// very unequal masses, where a single block solve would not.
    fn solve(self: *Weld, m: Masses, v: *Velocities, step: Step) void {
        if (self.axial_mass > 0) {
            const cdot = v.wb - v.wa;
            const impulse = if (self.angular_spring != null)
                self.angular_softness.impulse(self.axial_mass, cdot, self.angle_error, self.angular_impulse)
            else
                step.rigidAngle(self.axial_mass, cdot, self.angle_error, self.angular_impulse);
            self.angular_impulse += impulse;
            v.turn(m, impulse);
        }

        const cdot = v.relative(self.ra, self.rb);
        const impulse = if (self.linear_spring != null) soft: {
            const s = self.linear_softness;
            const x = self.matrix.solve(cdot.add(self.separation.scale(s.bias_rate)));
            break :soft x.scale(-s.mass_scale).sub(self.linear_impulse.scale(s.impulse_scale));
        } else step.rigidPoint(self.matrix, cdot, self.separation, self.linear_impulse);
        self.linear_impulse = self.linear_impulse.add(impulse);
        v.push(m, impulse, self.ra, self.rb);
    }

    fn reaction(self: *const Weld, inv_dt: f32) Reaction {
        return .{ .force = self.linear_impulse.scale(inv_dt), .torque = self.angular_impulse * inv_dt };
    }
};

// -------------------------------------------------------------------------
// Mouse
// -------------------------------------------------------------------------

pub const Mouse = struct {
    /// Where the body is pulled to, in the world. Move it every frame.
    target: Vec2,
    /// The point of the body that is pulled, in its frame.
    local_anchor: Vec2,
    spring: Spring,
    max_force: ?f32 = null,

    impulse: Vec2 = .zero,
    angular_impulse: f32 = 0,

    // Prepared each step.
    rb: Vec2 = .zero,
    matrix: Sym22 = .{},
    separation: Vec2 = .zero,
    softness: Softness = .{},
    angular_softness: Softness = .{},
    angular_mass: f32 = 0,
    max_impulse: f32 = 0,

    /// A gentle brake on spin, with no target angle: without it, a body
    /// dragged by a corner windmills round the pointer. Box2D v3's numbers.
    const spin_damping: Spring = .{ .hertz = 0.5, .damping_ratio = 0.1 };

    fn prepare(self: *Mouse, m: Masses, a: *const Body, b: *const Body, step: Step) void {
        _ = a;
        self.rb = arm(b, self.local_anchor);
        self.matrix = .point(m, .zero, self.rb);
        self.separation = b.center.add(self.rb).sub(self.target);
        self.softness = self.spring.softness(step.dt);
        self.angular_softness = spin_damping.softness(step.dt);
        self.angular_mass = if (m.ib > 0) 1 / m.ib else 0;
        const force = self.max_force orelse 1000 * b.mass * step.units_per_metre;
        self.max_impulse = force * step.dt;
    }

    fn warmStart(self: *const Mouse, m: Masses, v: *Velocities) void {
        v.vb = v.vb.mulAdd(self.impulse, m.mb);
        v.wb += m.ib * (cross(self.rb, self.impulse) + self.angular_impulse);
    }

    fn solve(self: *Mouse, m: Masses, v: *Velocities, step: Step) void {
        _ = step;
        {
            const s = self.angular_softness;
            const impulse = -s.mass_scale * self.angular_mass * v.wb - s.impulse_scale * self.angular_impulse;
            self.angular_impulse += impulse;
            v.wb += m.ib * impulse;
        }

        const s = self.softness;
        const cdot = v.vb.add(crossSV(v.wb, self.rb));
        const x = self.matrix.solve(cdot.add(self.separation.scale(s.bias_rate)));
        const wanted = x.scale(-s.mass_scale).sub(self.impulse.scale(s.impulse_scale));
        // The cap is on the total, like a friction cone, so a pull that has
        // hit it can still change direction.
        const old = self.impulse;
        self.impulse = old.add(wanted).clampLen(self.max_impulse);
        const impulse = self.impulse.sub(old);
        v.vb = v.vb.mulAdd(impulse, m.mb);
        v.wb += m.ib * cross(self.rb, impulse);
    }

    fn reaction(self: *const Mouse, inv_dt: f32) Reaction {
        return .{ .force = self.impulse.scale(inv_dt), .torque = self.angular_impulse * inv_dt };
    }
};

// -------------------------------------------------------------------------
// Wheel
// -------------------------------------------------------------------------

pub const Wheel = struct {
    local_anchor_a: Vec2,
    local_anchor_b: Vec2,
    /// The suspension's direction, in A's frame. Unit length.
    local_axis_a: Vec2,
    spring: ?Spring,
    limit: ?Limit = null,
    motor: ?AngularMotor = null,

    /// Across the axis: what keeps the wheel on its line.
    perp_impulse: f32 = 0,
    /// Along it: the suspension, or a rigid axle.
    spring_impulse: f32 = 0,
    motor_impulse: f32 = 0,
    lower_impulse: f32 = 0,
    upper_impulse: f32 = 0,

    // Prepared each step, as for `Prismatic`.
    axis: Vec2 = .zero,
    perp: Vec2 = .zero,
    a1: f32 = 0,
    a2: f32 = 0,
    s1: f32 = 0,
    s2: f32 = 0,
    axial_mass: f32 = 0,
    perp_mass: f32 = 0,
    motor_mass: f32 = 0,
    translation: f32 = 0,
    perp_error: f32 = 0,
    softness: Softness = .{},

    fn prepare(self: *Wheel, m: Masses, a: *const Body, b: *const Body, step: Step) void {
        const ra = arm(a, self.local_anchor_a);
        const rb = arm(b, self.local_anchor_b);
        const d = b.center.add(rb).sub(a.center.add(ra));
        self.axis = a.transform.q.rotate(self.local_axis_a);
        self.perp = crossSV(1, self.axis);

        const da = d.add(ra);
        self.a1 = cross(da, self.axis);
        self.a2 = cross(rb, self.axis);
        self.s1 = cross(da, self.perp);
        self.s2 = cross(rb, self.perp);
        self.axial_mass = axialMass(m, self.a1, self.a2);
        self.perp_mass = axialMass(m, self.s1, self.s2);
        self.motor_mass = if (m.ia + m.ib > 0) 1 / (m.ia + m.ib) else 0;

        self.translation = self.axis.dot(d);
        self.perp_error = self.perp.dot(d);
        self.softness = if (self.spring) |s| s.softness(step.dt) else .{};

        if (self.limit == null) {
            self.lower_impulse = 0;
            self.upper_impulse = 0;
        }
        if (self.motor == null) self.motor_impulse = 0;
    }

    fn axialSpeed(self: *const Wheel, v: *const Velocities) f32 {
        return self.axis.dot(v.vb.sub(v.va)) + self.a2 * v.wb - self.a1 * v.wa;
    }

    fn pushAxial(self: *const Wheel, m: Masses, v: *Velocities, impulse: f32) void {
        v.pushWith(m, self.axis.scale(impulse), impulse * self.a1, impulse * self.a2);
    }

    fn warmStart(self: *const Wheel, m: Masses, v: *Velocities) void {
        const axial = self.spring_impulse + self.lower_impulse - self.upper_impulse;
        const p = self.perp.scale(self.perp_impulse).add(self.axis.scale(axial));
        const la = self.perp_impulse * self.s1 + axial * self.a1 + self.motor_impulse;
        const lb = self.perp_impulse * self.s2 + axial * self.a2 + self.motor_impulse;
        v.pushWith(m, p, la, lb);
    }

    fn solve(self: *Wheel, m: Masses, v: *Velocities, step: Step) void {
        {
            const impulse = if (self.spring != null)
                self.softness.impulse(self.axial_mass, self.axialSpeed(v), self.translation, self.spring_impulse)
            else
                step.rigid(self.axial_mass, self.axialSpeed(v), self.translation, self.spring_impulse);
            self.spring_impulse += impulse;
            self.pushAxial(m, v, impulse);
        }
        if (self.motor) |motor| {
            if (self.motor_mass > 0) {
                const most = motor.max_torque * step.dt;
                const wanted = -self.motor_mass * (v.wb - v.wa - motor.speed);
                v.turn(m, clampedAdd(&self.motor_impulse, wanted, -most, most));
            }
        }
        if (self.limit) |limit| {
            const lower = @min(limit.lower, limit.upper);
            const upper = @max(limit.lower, limit.upper);
            const out = step.limit(self.axial_mass, self.axialSpeed(v), self.translation - lower, self.lower_impulse);
            self.pushAxial(m, v, clampedAdd(&self.lower_impulse, out, 0, std.math.inf(f32)));
            const in = step.limit(self.axial_mass, -self.axialSpeed(v), upper - self.translation, self.upper_impulse);
            self.pushAxial(m, v, -clampedAdd(&self.upper_impulse, in, 0, std.math.inf(f32)));
        }

        const cdot = self.perp.dot(v.vb.sub(v.va)) + self.s2 * v.wb - self.s1 * v.wa;
        const impulse = step.rigid(self.perp_mass, cdot, self.perp_error, self.perp_impulse);
        self.perp_impulse += impulse;
        v.pushWith(m, self.perp.scale(impulse), impulse * self.s1, impulse * self.s2);
    }

    fn reaction(self: *const Wheel, inv_dt: f32) Reaction {
        const axial = self.spring_impulse + self.lower_impulse - self.upper_impulse;
        return .{
            .force = self.perp.scale(self.perp_impulse).add(self.axis.scale(axial)).scale(inv_dt),
            .torque = self.motor_impulse * inv_dt,
        };
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

/// Rigid, so a pass can be checked against numbers worked out by hand. The
/// world's joints give a little, on purpose; see `Step.rigid`.
const test_step: Step = .{
    .dt = 1.0 / 60.0,
    .inv_dt = 60,
    .stiffness = .baumgarte(0.2, 60),
    .slop = 0.005,
    .units_per_metre = 1,
};

/// A unit disc of the given mass, somewhere, moving.
fn disc(mass: f32, position: Vec2, velocity: Vec2) Body {
    var b: Body = .fromDef(.{ .position = position, .linear_velocity = velocity });
    if (mass == 0) {
        b.type = .static;
        b.linear_velocity = .zero;
        return b;
    }
    b.mass = mass;
    b.inv_mass = 1 / mass;
    b.inertia = 0.5 * mass;
    b.inv_inertia = 1 / b.inertia;
    return b;
}

/// Build, prepare, warm-start and solve one joint the way a step would.
fn run(def: Def, a: *Body, b: *Body, passes: usize) Joint {
    var j: Joint = .init(def, a, b);
    j.masses = massesOf(&j, a, b);
    prepare(&j, a, b, test_step);
    warmStart(&j, a, b);
    for (0..passes) |_| solve(&j, a, b, test_step);
    return j;
}

test "a pin stops two bodies moving apart at it, and leaves them free to turn" {
    var a = disc(1, .init(0, 0), .init(-1, 0));
    var b = disc(1, .init(2, 0), .init(1, 0));
    const j = run(.{ .revolute = .{ .body_a = .none, .body_b = .none, .anchor = .init(1, 0) } }, &a, &b, 10);
    const rel = Velocities.of(&a, &b).relative(j.kind.revolute.ra, j.kind.revolute.rb);
    try testing.expect(rel.len() < 1e-4);
    // Momentum is kept: they were moving apart equally, and now neither is.
    try testing.expect(a.linear_velocity.add(b.linear_velocity).len() < 1e-5);
}

test "a rod holds the length along it and nothing across it" {
    var anchor = disc(0, .zero, .zero);
    var b = disc(1, .init(2, 0), .init(3, 4));
    _ = run(.{ .distance = .{ .body_a = .none, .body_b = .none, .anchor_a = .zero, .anchor_b = .init(2, 0) } }, &anchor, &b, 8);
    try testing.expectApproxEqAbs(@as(f32, 0), b.linear_velocity.x, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 4), b.linear_velocity.y, 1e-4);
    try testing.expect(anchor.linear_velocity.eql(.zero));
}

test "a rope pulls only when taut" {
    var anchor = disc(0, .zero, .zero);
    // Slack: two units of rope, one unit away, moving further. Nothing.
    var slack = disc(1, .init(1, 0), .init(1, 0));
    const def: Def = .{ .distance = .{
        .body_a = .none,
        .body_b = .none,
        .anchor_a = .zero,
        .anchor_b = .init(1, 0),
        .spring = .slack,
        .max_length = 2,
    } };
    _ = run(def, &anchor, &slack, 8);
    try testing.expectApproxEqAbs(@as(f32, 1), slack.linear_velocity.x, 1e-5);

    // Taut: at the end of the rope and still going. Stopped.
    var taut = disc(1, .init(2, 0), .init(1, 0));
    var at_end = def;
    at_end.distance.anchor_b = .init(2, 0);
    _ = run(at_end, &anchor, &taut, 8);
    try testing.expectApproxEqAbs(@as(f32, 0), taut.linear_velocity.x, 1e-4);

    // Taut and coming back: a rope does not push.
    var returning = disc(1, .init(2, 0), .init(-1, 0));
    _ = run(at_end, &anchor, &returning, 8);
    try testing.expectApproxEqAbs(@as(f32, -1), returning.linear_velocity.x, 1e-5);
}

test "a motor reaches its speed, and a weak one only pushes as hard as it may" {
    var ground = disc(0, .zero, .zero);
    var wheel = disc(1, .zero, .zero);
    const strong: Def = .{ .revolute = .{
        .body_a = .none,
        .body_b = .none,
        .anchor = .zero,
        .motor = .{ .speed = 3, .max_torque = 1000 },
    } };
    _ = run(strong, &ground, &wheel, 8);
    try testing.expectApproxEqAbs(@as(f32, 3), wheel.angular_velocity, 1e-4);

    // Inertia 0.5, a torque of 6 for a sixtieth: 0.2 radians per second.
    var heavy = disc(1, .zero, .zero);
    var weak = strong;
    weak.revolute.motor.?.max_torque = 6;
    const j = run(weak, &ground, &heavy, 8);
    try testing.expectApproxEqAbs(@as(f32, 0.2), heavy.angular_velocity, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 6), j.reaction(60).torque, 1e-3);
}

test "a mouse joint pulls towards the target and no harder than it is allowed" {
    var b = disc(2, .zero, .zero);
    const def: Def = .{ .mouse = .{ .body = .none, .target = .zero, .max_force = 12 } };
    var j: Joint = .init(def, &b, &b);
    j.kind.mouse.target = .init(10, 0);
    j.masses = massesOf(&j, &b, &b);
    try testing.expectEqual(@as(f32, 0), j.masses.ma);
    prepare(&j, &b, &b, test_step);
    for (0..8) |_| solve(&j, &b, &b, test_step);
    // Twelve newtons on two kilograms for a sixtieth of a second.
    try testing.expectApproxEqAbs(@as(f32, 0.1), b.linear_velocity.x, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0), b.linear_velocity.y, 1e-5);
}
