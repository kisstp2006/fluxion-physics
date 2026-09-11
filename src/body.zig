// SPDX-License-Identifier: BSD-2-Clause

//! A rigid body: where it is, how it moves, and what it weighs.
//!
//! ```zig
//! const crate = try world.createBody(.{ .position = .init(4, -3) });
//! _ = try world.addShape(crate, .box(0.5, 0.5));
//! world.body(crate).?.applyImpulse(.init(0, -5), world.body(crate).?.center);
//! ```
//!
//! **Three kinds.** A *static* body never moves and has infinite mass: the
//! ground, the walls, the level. A *kinematic* body moves the way it is told
//! and nothing pushes it back: a moving platform, a door. A *dynamic* body
//! is the one physics is for: it falls, bounces, and is stopped by the other
//! two. Only dynamic bodies collide with each other; two walls never need to
//! know they overlap.
//!
//! **The body moves about its centre of mass, not its origin.** The origin is
//! where a game put it - the corner of a sprite, the feet of a character -
//! and the centre of mass is where the shapes say it is. Velocity and
//! rotation act on the centre; the origin is worked out from it afterwards.
//! Forgetting this is why a crate with a lopsided shape spins when pushed
//! straight, and why it is right that it does.
//!
//! **A body owns nothing**, so it is handed out by pointer from the world's
//! table and written to directly. The pointer is good until the next
//! `createBody`; the handle is good for ever.
//!
//! **A dynamic body falls asleep** when it and everything it touches have
//! been still for a while, and a step skips it until something wakes it.
//! Writing a velocity, applying a force or an impulse, and `setTransform`
//! all wake it, because a sleeping body has all of those at zero and the
//! step notices when one is not. See `World` for the rest of the rules.

const std = @import("std");
const testing = std.testing;
const id = @import("fluxion_id");

const geometry = @import("geometry.zig");
const Vec2 = geometry.Vec2;
const Transform = geometry.Transform;
const shape = @import("shape.zig");

const Body = @This();

/// What names a body. See `World.createBody`.
pub const Id = id.Handle(Body);

/// What names a shape on a body: a handle into the world's table of them.
/// Declared here because a body holds the head of its list of shapes, and
/// the import is circular on purpose - Zig resolves those lazily, and the
/// alternative is a third file for one line.
pub const ShapeId = id.Handle(@import("World.zig").ShapeEntry);

pub const Type = enum(u8) {
    static,
    kinematic,
    dynamic,
};

/// What `World.createBody` takes. Every field has a default, so a dynamic
/// body at the origin is `.{}`.
pub const Def = struct {
    type: Type = .dynamic,
    /// Where the origin starts.
    position: Vec2 = .zero,
    /// Radians. See `geometry` for which way is positive.
    angle: f32 = 0,
    linear_velocity: Vec2 = .zero,
    angular_velocity: f32 = 0,
    /// How quickly speed leaks away with nothing touching: zero is space,
    /// a fraction of one is air. Per second, roughly.
    linear_damping: f32 = 0,
    angular_damping: f32 = 0,
    /// One is the world's gravity, zero is floating, minus one is a balloon.
    gravity_scale: f32 = 1,
    /// A body that never turns, however it is hit. A character capsule.
    fixed_rotation: bool = false,
    /// Whether it may fall asleep. Turn it off for a body a game steers by
    /// setting its velocity only when a key is down: asleep, it would not
    /// notice the first frame of the next press. A body that may not sleep
    /// keeps everything it touches awake too.
    allow_sleep: bool = true,
    /// Swept against other moving bodies too, not only the level, so it
    /// cannot pass through a crate or a plank between one step and the
    /// next. For small, fast things meant to hit other moving things: a
    /// bullet, a thrown knife. Every fast body is swept against the level
    /// whatever this says; see `World` for what continuous collision does.
    /// Two bullets do not see each other.
    bullet: bool = false,
    /// Yours. The body never reads it.
    user_data: u64 = 0,
};

type: Type,
/// Where the origin is and which way the body faces. Read this to draw.
transform: Transform,
/// The angle the rotation in `transform` was built from. Kept beside it
/// because a rotation stored as a cosine and a sine cannot be integrated
/// without drifting off unit length, and an angle can.
angle: f32,
/// The centre of mass, in the world. What the velocity is the velocity of.
center: Vec2,
/// The centre of mass in the body's own frame.
local_center: Vec2 = .zero,
linear_velocity: Vec2,
angular_velocity: f32,
/// Accumulated until the next step, then cleared.
force: Vec2 = .zero,
torque: f32 = 0,
mass: f32 = 0,
inv_mass: f32 = 0,
/// Rotational inertia about the centre of mass.
inertia: f32 = 0,
inv_inertia: f32 = 0,
linear_damping: f32,
angular_damping: f32,
gravity_scale: f32,
fixed_rotation: bool,
allow_sleep: bool,
/// See `Def.bullet`. May be changed at any time.
bullet: bool,
user_data: u64,
/// False while asleep. Only ever false for a dynamic body.
awake: bool = true,
/// How long it has been still, in seconds. Reset by any movement; see
/// `World.Settings.time_to_sleep`.
sleep_time: f32 = 0,
/// Moved by `setTransform` since the last step. Contacts that were asleep
/// against it are looked at again rather than kept, so a platform moved by
/// hand does not leave what slept on it floating.
teleported: bool = false,
/// How far its shapes reach from its centre of mass. What a spin moves its
/// furthest edge by, for deciding whether it is still. Kept by
/// `World.updateMass`.
extent: f32 = 0,
/// How thin its thinnest shape is, from that shape's middle to its nearest
/// edge. A body that moves more than half of this in a step is swept for
/// what it could have passed through. Kept by `World.updateMass`.
min_extent: f32 = 0,
/// Where the centre of mass was, and the angle, when the step began: the
/// start of the path the continuous pass sweeps. Written by the step.
center0: Vec2 = .zero,
angle0: f32 = 0,
/// The continuous pass stopped it short last step, where it first touched
/// something, still moving towards it. This step looks a little ahead of
/// it for what it touches, so the solver sees what it is about to hit.
stopped_short: bool = false,
/// The head of the list of shapes on this body. The list is threaded
/// through the shapes themselves - see `World.ShapeEntry.next`.
first_shape: ShapeId = .none,
shape_count: u32 = 0,

pub fn fromDef(def: Def) Body {
    const xf: Transform = .init(def.position, def.angle);
    return .{
        .type = def.type,
        .transform = xf,
        .angle = def.angle,
        .center = def.position,
        .linear_velocity = if (def.type == .static) .zero else def.linear_velocity,
        .angular_velocity = if (def.type == .static) 0 else def.angular_velocity,
        .linear_damping = def.linear_damping,
        .angular_damping = def.angular_damping,
        .gravity_scale = def.gravity_scale,
        .fixed_rotation = def.fixed_rotation,
        .allow_sleep = def.allow_sleep,
        .bullet = def.bullet,
        .user_data = def.user_data,
        .center0 = def.position,
        .angle0 = def.angle,
    };
}

/// Where the origin is.
pub fn position(self: *const Body) Vec2 {
    return self.transform.p;
}

/// Put the body somewhere, at once and without a velocity. For placing
/// things at load time and for teleporting; anything else should be a
/// velocity or a force, which the solver can reason about.
///
/// Wakes it, and whatever was asleep against it.
pub fn setTransform(self: *Body, p: Vec2, radians: f32) void {
    self.transform = .init(p, radians);
    self.angle = radians;
    self.center = self.transform.apply(self.local_center);
    self.teleported = true;
    self.wake();
}

/// Whether a step will move it. Static and kinematic bodies never sleep.
pub fn isAwake(self: *const Body) bool {
    return self.awake;
}

/// Wake it. Everything it touches wakes with it at the end of the next
/// step, because an island sleeps and wakes as one.
///
/// Needed only for a change the world cannot see - gravity turned round, a
/// joint's limit or spring moved. Velocities, forces, impulses, moving it,
/// new shapes and a joint's motor or pointer are all seen.
pub fn wake(self: *Body) void {
    if (self.type != .dynamic) return;
    self.awake = true;
    self.sleep_time = 0;
}

/// Put it to sleep now, still. The world does this to whole islands; a
/// game may do it to what it has just placed, so a level starts at rest
/// rather than settling. Asleep with anything awake touching it, it is
/// woken again at the end of the next step.
pub fn sleep(self: *Body) void {
    if (self.type != .dynamic) return;
    self.awake = false;
    // Longer than any time to sleep: a body put to sleep by hand does not
    // hold its island awake while its timer catches up.
    self.sleep_time = std.math.inf(f32);
    self.linear_velocity = .zero;
    self.angular_velocity = 0;
    self.force = .zero;
    self.torque = 0;
}

/// Whether something from outside has pushed it since it fell asleep.
/// Sleeping zeroed all four, so any of them not zero is news.
pub fn isPushed(self: *const Body) bool {
    return self.linear_velocity.x != 0 or self.linear_velocity.y != 0 or self.angular_velocity != 0 or
        self.force.x != 0 or self.force.y != 0 or self.torque != 0;
}

/// Rebuild the origin's transform from the centre and angle after a step
/// has moved them. The one place the two are reconciled.
pub fn syncTransform(self: *Body) void {
    self.transform.q = .fromAngle(self.angle);
    self.transform.p = self.center.sub(self.transform.q.rotate(self.local_center));
}

/// The velocity of a world point on the body: the linear part plus what
/// the spin does at that distance from the centre.
pub fn velocityAt(self: *const Body, world_point: Vec2) Vec2 {
    return self.linear_velocity.add(geometry.crossSV(self.angular_velocity, world_point.sub(self.center)));
}

/// A force through the centre of mass, until the next step. No torque.
pub fn applyForce(self: *Body, force: Vec2) void {
    if (self.type != .dynamic) return;
    self.force = self.force.add(force);
}

/// A force at a world point: the same push, and a turn if the point is
/// off centre.
pub fn applyForceAt(self: *Body, force: Vec2, world_point: Vec2) void {
    if (self.type != .dynamic) return;
    self.force = self.force.add(force);
    self.torque += geometry.cross(world_point.sub(self.center), force);
}

pub fn applyTorque(self: *Body, torque: f32) void {
    if (self.type != .dynamic) return;
    self.torque += torque;
}

/// An instant change of momentum at a world point. A kick, a bullet, an
/// explosion - anything that happens rather than pushes.
pub fn applyImpulse(self: *Body, impulse: Vec2, world_point: Vec2) void {
    if (self.type != .dynamic) return;
    self.linear_velocity = self.linear_velocity.mulAdd(impulse, self.inv_mass);
    self.angular_velocity += self.inv_inertia * geometry.cross(world_point.sub(self.center), impulse);
}

pub fn applyAngularImpulse(self: *Body, impulse: f32) void {
    if (self.type != .dynamic) return;
    self.angular_velocity += self.inv_inertia * impulse;
}

/// Half m v squared plus half I w squared, for the log line that says what hit what how hard.
pub fn kineticEnergy(self: *const Body) f32 {
    return 0.5 * (self.mass * self.linear_velocity.lenSq() +
        self.inertia * self.angular_velocity * self.angular_velocity);
}

/// Whether the solver may change this body's velocity. Static and kinematic
/// bodies are never written to during a step, which is what lets two jobs
/// solve two contacts against the same wall at the same time.
pub inline fn isDynamic(self: *const Body) bool {
    return self.type == .dynamic;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "a static body ignores the velocity in its definition" {
    const b: Body = .fromDef(.{ .type = .static, .linear_velocity = .init(1, 1) });
    try testing.expect(b.linear_velocity.eql(.zero));
    try testing.expect(!b.isDynamic());
}

test "an impulse off centre spins as well as pushes" {
    var b: Body = .fromDef(.{});
    b.mass = 2;
    b.inv_mass = 0.5;
    b.inertia = 4;
    b.inv_inertia = 0.25;

    b.applyImpulse(.init(0, 2), .init(1, 0));
    try testing.expect(b.linear_velocity.approxEql(.init(0, 1)));
    // r = (1, 0), J = (0, 2): r x J = 2, times 1/4.
    try testing.expectApproxEqAbs(@as(f32, 0.5), b.angular_velocity, 1e-6);

    // A point on the far side moves the other way from the spin.
    const at = b.velocityAt(.init(0, -2));
    try testing.expect(at.approxEql(.init(1, 1)));
}

test "moving the origin moves the centre with it" {
    var b: Body = .fromDef(.{});
    b.local_center = .init(1, 0);
    b.setTransform(.init(10, 10), std.math.pi / 2.0);
    try testing.expect(b.center.approxEql(.init(10, 11)));

    // And going back the other way: a centre and an angle give the origin.
    b.center = .init(0, 1);
    b.angle = std.math.pi / 2.0;
    b.syncTransform();
    try testing.expect(b.transform.p.approxEql(.zero));
}
