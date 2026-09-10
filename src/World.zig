// SPDX-License-Identifier: BSD-2-Clause

//! The bodies, the shapes on them, and one step of time.
//!
//! ```zig
//! var world: World = .init(gpa, .{ .units_per_metre = 100 });
//! defer world.deinit();
//!
//! const ground = try world.createBody(.{ .type = .static, .position = .init(0, 500) });
//! _ = try world.addShape(ground, .box(1000, 10));
//!
//! const crate = try world.createBody(.{ .position = .init(100, 0) });
//! _ = try world.addShape(crate, .box(20, 20));
//!
//! try world.step(1.0 / 60.0, &jobs);
//! const where = world.body(crate).?.position();
//! ```
//!
//! **A step is thirteen phases**, and eight of them run on every core:
//!
//! | Phase | What | On |
//! | --- | --- | --- |
//! | wake | sleepers a joint's motor or pointer pulls on | one |
//! | integrate velocities | gravity, forces, damping; sleepers pushed from outside wake | every core |
//! | update boxes | each moving shape's box in the world | every core |
//! | broad phase | which boxes overlap, of pairs with something moving | one |
//! | narrow phase | where each pair touches | every core |
//! | prepare contacts | into constraints, warm-started; sleeping ones kept | one |
//! | prepare joints | lever arms, masses, springs | every core |
//! | colour | contacts and joints into runs that share no body | one |
//! | solve | warm start, then the passes: joints, then contacts | every core, per colour |
//! | push | sunk-in bodies pushed apart, on a velocity that is then forgotten | every core, per colour |
//! | integrate positions | move, turn, rebuild transforms | every core |
//! | remember | impulses for next step, begin and end events | one |
//! | sleep | who has been still, and which islands rest or wake | one |
//!
//! With a scheduler that has no workers - a browser - the same phases run
//! on the calling thread in the same order. Nothing is compiled out.
//!
//! **Handles, not pointers.** A body is named by a `Body.Id`, a shape by a
//! `ShapeId` and a joint by a `JointId`, all generational: a handle to
//! something destroyed answers null, for ever, however many times its slot
//! is reused. `body(id)` hands out a pointer for writing velocity and
//! applying forces; hold the handle and ask again next frame.
//!
//! **Units are yours.** Set `Settings.units_per_metre` to what one metre is
//! in your coordinates - a hundred, for a game that thinks in pixels - and
//! the tolerances the solver needs, which are in metres, are scaled by it.
//! Gravity defaults to Earth's, downwards, in those units.
//!
//! **What rests, sleeps.** An *island* is a set of dynamic bodies joined by
//! touching and by joints - what one does, the others feel. When every body
//! in an island has been still for `Settings.time_to_sleep`, the whole
//! island sleeps: a step no longer moves it, tests its pairs or solves its
//! contacts, which keep their impulses for the moment it wakes. It wakes as
//! a whole too, when anything in it is disturbed: a velocity, a force or an
//! impulse from outside; `setTransform`; a shape added or taken away; a
//! contact that ends, which is how a stack whose bottom crate is destroyed
//! comes down; an awake body touching it; a kinematic body moving against
//! it; a joint's motor given a speed, or a mouse joint's target moved.
//! What the world cannot see - gravity turned round, a joint's limit moved
//! - wants a `Body.wake`. Islands are found again every step, which costs a
//! pass over the contacts and never goes stale.
//!
//! **What is not here yet**: continuous collision, and a tree for queries.
//! Each is listed in the README with what it would take.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const id = @import("fluxion_id");
const Jobs = @import("fluxion_jobs").Jobs;

const geometry = @import("geometry.zig");
const Vec2 = geometry.Vec2;
const Transform = geometry.Transform;
const Aabb = geometry.Aabb;
const shape_mod = @import("shape.zig");
const Shape = shape_mod.Shape;
const Filter = shape_mod.Filter;
const Body = @import("body.zig");
const collide = @import("collide.zig");
const Manifold = collide.Manifold;
const broadphase = @import("broadphase.zig");
const contact = @import("contact.zig");
const joint_mod = @import("joint.zig");
const Joint = joint_mod.Joint;
const solver = @import("solver.zig");

const World = @This();

pub const BodyId = Body.Id;
pub const ShapeId = Body.ShapeId;
pub const JointId = joint_mod.Id;

pub const Settings = struct {
    /// Null is Earth's, pulling towards `+y`, which is down on this screen.
    /// Set it to anything, including zero for space.
    gravity: ?Vec2 = null,
    /// What a metre is in your coordinates. Everything below that is a
    /// length or a speed is in metres and scaled by this.
    units_per_metre: f32 = 1,
    /// Passes the solver makes over the contacts and joints. Eight is
    /// Box2D's default; a deep pile wants more, a game of loose balls fewer.
    velocity_iterations: u32 = 8,
    /// Passes over the contacts pushing sunk-in bodies apart, on a velocity
    /// that moves them and is then forgotten. None, and nothing is ever
    /// pushed out; see `contact`.
    push_iterations: u32 = 3,
    /// How much of a penetration is removed per step. Higher is stiffer and
    /// jumpier; lower is softer and sinks.
    baumgarte: f32 = 0.2,
    /// How far a shape may sink before it is pushed out. Metres.
    linear_slop: f32 = 0.005,
    /// Slower than this, nothing bounces. Metres per second.
    restitution_threshold: f32 = 1,
    /// The fastest a penetration is pushed apart. Metres per second. What
    /// keeps a body that was placed inside another from leaving at the
    /// speed of a bullet, which is what an unbounded Baumgarte term does.
    max_push_speed: f32 = 3,
    /// How a rigid joint takes back the little it drifts: the way a spring
    /// would that rang this many times a second, damped this many times
    /// past the point of bouncing. Stiff enough that nobody sees the give,
    /// and a spring rather than a fixed fraction per step because the
    /// fraction, fed back as speed, pumps energy into a light chain holding
    /// a heavy weight. Zero hertz holds joints rigid and lets them drift.
    /// See `joint`.
    joint_hertz: f32 = 60,
    joint_damping_ratio: f32 = 2,
    /// Whether anything sleeps at all. Turning it off wakes what sleeps.
    enable_sleep: bool = true,
    /// Slower than this counts as still. Metres per second, of the body's
    /// centre plus what its spin does to its furthest edge - one number for
    /// a pebble and a boulder, which a separate angular one would not be.
    sleep_threshold: f32 = 0.05,
    /// How long an island must be still before it sleeps. Seconds.
    time_to_sleep: f32 = 0.5,
};

/// A shape as the world keeps it: the definition, and where it hangs.
pub const ShapeEntry = struct {
    def: Shape,
    body: BodyId,
    /// The body's slot, so a step reads it without resolving the handle.
    body_index: u32,
    /// The next shape on the same body. See `Body.first_shape`.
    next: ShapeId,
};

/// Two shapes that started or stopped touching during the last step.
pub const ContactEvent = struct {
    shape_a: ShapeId,
    shape_b: ShapeId,
    body_a: BodyId,
    body_b: BodyId,
    /// True when either shape is a sensor: nothing was pushed, only seen.
    sensor: bool,
};

/// What a ray found.
pub const RayHit = struct {
    shape: ShapeId,
    body: BodyId,
    point: Vec2,
    /// Of the surface hit, pointing out of it.
    normal: Vec2,
    /// How far along the ray, from zero at the origin to one at its end.
    fraction: f32,
};

pub const Error = Allocator.Error;

/// What `addShape` can fail with besides memory.
pub const ShapeError = Allocator.Error || error{
    /// The body has been destroyed, or the handle was never one.
    NoSuchBody,
};

/// What `createJoint` can fail with. `||` joins two error sets into one,
/// so this is every error of `ShapeError` and one more.
pub const JointError = ShapeError || error{
    /// A joint between a body and itself holds nothing.
    SameBody,
};

gpa: Allocator,
settings: Settings,
/// The acceleration every dynamic body feels, times its `gravity_scale`.
/// Change it whenever.
gravity: Vec2,

bodies: id.Table(Body) = .empty,
shapes: id.Table(ShapeEntry) = .empty,
joints: id.Table(Joint) = .empty,
sweep: broadphase.Sweep(*World) = .empty,

/// Pairs of bodies a joint says must not collide, by slot, with how many
/// joints say so. Asked by the broad phase for every pair it finds while
/// there is anything in it; see `bodyPairKey`.
no_collide: std.AutoHashMapUnmanaged(u64, u32) = .empty,

// Per-step scratch, kept so a step allocates only when the scene grows.
aabbs: std.ArrayList(Aabb) = .empty,
pairs: std.ArrayList(broadphase.Pair) = .empty,
manifolds: std.ArrayList(Manifold) = .empty,
constraints: std.ArrayList(contact.Constraint) = .empty,
colouring: solver.Colouring(contact.Constraint) = .empty,
joint_refs: std.ArrayList(joint_mod.Ref) = .empty,
joint_colouring: solver.Colouring(joint_mod.Ref) = .empty,
/// Per body slot: the island it is in, as a union-find forest, and how
/// long that island has been still. See `updateSleep`.
island_parent: std.ArrayList(u32) = .empty,
island_rest: std.ArrayList(f32) = .empty,
/// Every contact that pushes, asleep or awake, as the islands see it:
/// written while the contacts are prepared, read by `updateSleep`.
links: std.ArrayList(Link) = .empty,
/// One over the last step's length, for turning impulses into forces.
inv_dt: f32 = 0,

/// What was touching at the end of the last step, with the impulses that
/// held it there, and the same for the step before. Swapped every step.
contacts: ContactMap = .empty,
previous: ContactMap = .empty,
begin_events: std.ArrayList(ContactEvent) = .empty,
end_events: std.ArrayList(ContactEvent) = .empty,

/// How many steps have run. A frame counter for the log.
step_count: u64 = 0,

const ContactMap = std.AutoHashMapUnmanaged(contact.PairKey, Stored);

/// Two bodies that push on each other, by slot, with what kind each is.
///
/// Eleven bytes a contact, in an array, because the islands are found by
/// walking every contact every step: walking the contact map instead means
/// a hash map's gaps and two bodies' worth of cache lines per contact, and
/// on a heap of six hundred bodies that was most of what sleeping cost.
const Link = struct {
    a: u32,
    b: u32,
    type_a: Body.Type,
    type_b: Body.Type,
};

const Stored = struct {
    shape_a: ShapeId,
    shape_b: ShapeId,
    body_a: BodyId,
    body_b: BodyId,
    sensor: bool,
    impulses: [2]contact.Impulse = .{ .{}, .{} },
};

pub fn init(gpa: Allocator, settings: Settings) World {
    return .{
        .gpa = gpa,
        .settings = settings,
        .gravity = settings.gravity orelse .init(0, 9.81 * settings.units_per_metre),
    };
}

pub fn deinit(self: *World) void {
    const gpa = self.gpa;
    self.bodies.deinit(gpa);
    self.shapes.deinit(gpa);
    self.joints.deinit(gpa);
    self.sweep.deinit(gpa);
    self.no_collide.deinit(gpa);
    self.aabbs.deinit(gpa);
    self.pairs.deinit(gpa);
    self.manifolds.deinit(gpa);
    self.constraints.deinit(gpa);
    self.colouring.deinit(gpa);
    self.joint_refs.deinit(gpa);
    self.joint_colouring.deinit(gpa);
    self.island_parent.deinit(gpa);
    self.island_rest.deinit(gpa);
    self.links.deinit(gpa);
    self.contacts.deinit(gpa);
    self.previous.deinit(gpa);
    self.begin_events.deinit(gpa);
    self.end_events.deinit(gpa);
    self.* = undefined;
}

// -------------------------------------------------------------------------
// Bodies
// -------------------------------------------------------------------------

pub fn createBody(self: *World, def: Body.Def) Error!BodyId {
    return self.bodies.add(self.gpa, .fromDef(def));
}

/// Take a body, every shape on it and every joint to it out of the world.
/// Contacts it was in end next step, and every handle to it answers null
/// from now on.
pub fn destroyBody(self: *World, handle: BodyId) void {
    const doomed = self.bodies.get(handle) orelse return;

    // A linear walk, because a body does not keep a list of its joints:
    // destroying is rare beside stepping, and a list would be two more
    // links in every joint to keep straight.
    for (self.joints.slots.items, 0..) |*slot, i| {
        const j = &(slot.value orelse continue);
        if (j.body_a.eql(handle) or j.body_b.eql(handle)) {
            self.destroyJoint(.{ .index = @intCast(i), .generation = slot.generation });
        }
    }

    var next = doomed.first_shape;
    while (self.shapes.get(next)) |entry| {
        const after = entry.next;
        self.sweep.remove(next.index);
        _ = self.shapes.remove(next);
        next = after;
    }
    _ = self.bodies.remove(handle);
}

/// The body, to read or to push. Null once destroyed. The pointer is good
/// until the next `createBody`.
pub fn body(self: *World, handle: BodyId) ?*Body {
    return self.bodies.get(handle);
}

pub fn bodyConst(self: *const World, handle: BodyId) ?*const Body {
    return self.bodies.getConst(handle);
}

pub fn bodyCount(self: *const World) usize {
    return self.bodies.count();
}

/// Every live body, in slot order, for drawing.
pub fn bodyIterator(self: *World) id.Table(Body).Iterator {
    return self.bodies.iterator();
}

/// The body in a slot. Only a step calls this, with slots it knows are live.
inline fn bodyAt(self: *World, index: u32) *Body {
    return &self.bodies.slots.items[index].value.?;
}

// -------------------------------------------------------------------------
// Shapes
// -------------------------------------------------------------------------

/// Put a shape on a body. The body's mass is recomputed from all its
/// shapes, so add the shapes before reading the mass.
pub fn addShape(self: *World, body_handle: BodyId, def: Shape) ShapeError!ShapeId {
    const owner = self.bodies.get(body_handle) orelse return error.NoSuchBody;
    const handle = try self.shapes.add(self.gpa, .{
        .def = def,
        .body = body_handle,
        .body_index = body_handle.index,
        .next = owner.first_shape,
    });
    errdefer _ = self.shapes.remove(handle);
    try self.sweep.add(self.gpa, handle.index);
    owner.first_shape = handle;
    owner.shape_count += 1;
    self.updateMass(body_handle);
    return handle;
}

/// Take one shape off its body. The body stays, lighter.
pub fn removeShape(self: *World, handle: ShapeId) void {
    const entry = self.shapes.get(handle) orelse return;
    const owner_handle = entry.body;
    const after = entry.next;
    if (self.bodies.get(owner_handle)) |owner| {
        if (owner.first_shape.eql(handle)) {
            owner.first_shape = after;
        } else {
            var cursor = owner.first_shape;
            while (self.shapes.get(cursor)) |link| : (cursor = link.next) {
                if (link.next.eql(handle)) {
                    link.next = after;
                    break;
                }
            }
        }
        owner.shape_count -= 1;
    }
    self.sweep.remove(handle.index);
    _ = self.shapes.remove(handle);
    self.updateMass(owner_handle);
}

/// The shape, to read or to change its material or filter. Changing its
/// geometry or density is allowed too; call `updateMass` on the body after.
pub fn shape(self: *World, handle: ShapeId) ?*ShapeEntry {
    return self.shapes.get(handle);
}

pub fn shapeCount(self: *const World) usize {
    return self.shapes.count();
}

/// Every live shape, in slot order, for drawing. The body it hangs from is
/// in the entry.
pub fn shapeIterator(self: *World) id.Table(ShapeEntry).Iterator {
    return self.shapes.iterator();
}

/// Where a shape is in the world right now.
pub fn shapeTransform(self: *World, entry: *const ShapeEntry) Transform {
    return self.bodyAt(entry.body_index).transform;
}

/// Add up what a body's shapes weigh and where. Called by `addShape` and
/// `removeShape`; call it yourself after changing a shape's density.
///
/// A dynamic body with no mass - no shapes, or shapes of zero density -
/// gets a mass of one rather than infinity, because a body somebody made
/// dynamic was meant to move. And it wakes: heavier or lighter, it has to
/// find its feet again.
pub fn updateMass(self: *World, handle: BodyId) void {
    const b = self.bodies.get(handle) orelse return;
    b.mass = 0;
    b.inv_mass = 0;
    b.inertia = 0;
    b.inv_inertia = 0;
    b.local_center = .zero;

    if (b.type != .dynamic) {
        b.center = b.transform.p;
        b.extent = self.extentOf(b);
        return;
    }

    var local_center: Vec2 = .zero;
    var inertia: f32 = 0;
    var cursor = b.first_shape;
    while (self.shapes.get(cursor)) |entry| : (cursor = entry.next) {
        const md = entry.def.geometry.massData(entry.def.material.density);
        if (md.mass <= 0) continue;
        b.mass += md.mass;
        local_center = local_center.mulAdd(md.center, md.mass);
        inertia += md.inertia;
    }

    if (b.mass > 0) {
        b.inv_mass = 1 / b.mass;
        local_center = local_center.scale(b.inv_mass);
    } else {
        b.mass = 1;
        b.inv_mass = 1;
    }

    if (inertia > 0 and !b.fixed_rotation) {
        // From the origin to the centre of mass: the parallel axis theorem
        // the other way round.
        inertia -= b.mass * local_center.lenSq();
        if (inertia > 0) {
            b.inertia = inertia;
            b.inv_inertia = 1 / inertia;
        }
    }

    // The centre moved; the velocity is of the centre, so a spinning body
    // whose centre moved has a different linear velocity now.
    const old_center = b.center;
    b.local_center = local_center;
    b.center = b.transform.apply(local_center);
    b.linear_velocity = b.linear_velocity.add(geometry.crossSV(b.angular_velocity, b.center.sub(old_center)));
    b.extent = self.extentOf(b);
    b.wake();
}

/// How far a body's shapes reach from its centre of mass.
fn extentOf(self: *World, b: *const Body) f32 {
    var reach: f32 = 0;
    var cursor = b.first_shape;
    while (self.shapes.get(cursor)) |entry| : (cursor = entry.next) {
        switch (entry.def.geometry) {
            .circle => |c| reach = @max(reach, c.center.dist(b.local_center) + c.radius),
            .polygon => |*p| {
                for (p.vertexSlice()) |v| reach = @max(reach, v.dist(b.local_center));
            },
        }
    }
    return reach;
}

/// How many dynamic bodies are awake. For the log line that says whether
/// sleeping is earning its keep.
pub fn awakeCount(self: *const World) usize {
    var n: usize = 0;
    for (self.bodies.slots.items) |*slot| {
        const b = &(slot.value orelse continue);
        if (b.type == .dynamic and b.awake) n += 1;
    }
    return n;
}

// -------------------------------------------------------------------------
// Joints
// -------------------------------------------------------------------------

/// Hold two bodies together, or one to a point. See `joint` for the kinds.
///
/// Every point in the definition is in the world, now, and the pose the
/// bodies are in now is the joint's rest pose - so make the bodies, put
/// them where they belong, then join them.
pub fn createJoint(self: *World, def: joint_mod.Def) JointError!JointId {
    const pair = def.bodies();
    const a = self.bodies.getConst(pair[0]) orelse return error.NoSuchBody;
    const b = self.bodies.getConst(pair[1]) orelse return error.NoSuchBody;
    // Comparing a tagged union with an enum literal compares its tag.
    if (def != .mouse and pair[0].eql(pair[1])) return error.SameBody;

    const made: Joint = .init(def, a, b);
    const handle = try self.joints.add(self.gpa, made);
    errdefer _ = self.joints.remove(handle);
    if (!made.collide_connected) {
        const entry = try self.no_collide.getOrPut(self.gpa, bodyPairKey(pair[0].index, pair[1].index));
        entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
    }
    // What holds them has changed; they settle again.
    self.bodyAt(pair[0].index).wake();
    self.bodyAt(pair[1].index).wake();
    return handle;
}

/// Take a joint out. The bodies it held stay where they are, wake, and may
/// collide with each other again if nothing else says they may not.
pub fn destroyJoint(self: *World, handle: JointId) void {
    const gone = self.joints.remove(handle) orelse return;
    if (!gone.collide_connected) {
        const key = bodyPairKey(gone.body_a.index, gone.body_b.index);
        if (self.no_collide.getPtr(key)) |count| {
            count.* -= 1;
            if (count.* == 0) _ = self.no_collide.remove(key);
        }
    }
    self.bodyAt(gone.body_a.index).wake();
    self.bodyAt(gone.body_b.index).wake();
}

/// The joint, to read or to steer: a motor's speed, a mouse joint's
/// target. Null once destroyed, including by destroying either body.
pub fn joint(self: *World, handle: JointId) ?*Joint {
    return self.joints.get(handle);
}

pub fn jointCount(self: *const World) usize {
    return self.joints.count();
}

/// Every live joint, in slot order, for drawing.
pub fn jointIterator(self: *World) id.Table(Joint).Iterator {
    return self.joints.iterator();
}

/// Where a joint is fixed to each of its bodies, in the world, now. What
/// a debug view draws a line between. For a mouse joint, the target and
/// the point being pulled.
pub fn jointAnchors(self: *World, j: *const Joint) [2]Vec2 {
    const local = j.localAnchors();
    const b = self.bodyAt(j.body_b.index).transform.apply(local[1]);
    if (j.kind == .mouse) return .{ local[0], b };
    return .{ self.bodyAt(j.body_a.index).transform.apply(local[0]), b };
}

/// The force and torque a joint put on body B during the last step.
pub fn jointReaction(self: *World, handle: JointId) ?joint_mod.Reaction {
    const j = self.joints.getConst(handle) orelse return null;
    return j.reaction(self.inv_dt);
}

/// The key two body slots are known by in `no_collide`, the same whichever
/// way round they are given.
fn bodyPairKey(a: u32, b: u32) u64 {
    return @as(u64, @min(a, b)) << 32 | @max(a, b);
}

// -------------------------------------------------------------------------
// Stepping
// -------------------------------------------------------------------------

/// How many of each thing one job takes.
///
/// Large on purpose. A job costs a lock and, when a worker is asleep, a
/// wake-up - tens of microseconds on the one-queue scheduler this runs
/// on - and a body integrates in tens of nanoseconds. A colour with fewer
/// contacts than `grain_contacts` is solved on the calling thread, and a
/// scene of a few hundred bodies never spawns a job for its solver at all;
/// that is the right answer for it, because the join would cost more than
/// the work. The demo prints both timings so the crossover can be seen on
/// the machine at hand.
const grain_bodies = 512;
const grain_pairs = 256;
const grain_contacts = 256;
const grain_joints = 256;

/// What every phase of a step is handed.
const Step = struct {
    world: *World,
    dt: f32,
    inv_dt: f32,
    joint: joint_mod.Step,
};

/// Advance the world by `dt` seconds, on `jobs`.
///
/// Call it with the same `dt` every time - a fixed step - because a solver
/// tuned by a bias per step behaves differently at a different step, and
/// because the same inputs then give the same world on every machine.
pub fn step(self: *World, dt: f32, jobs: *Jobs) Error!void {
    if (dt <= 0) return;
    const gpa = self.gpa;
    const units = self.settings.units_per_metre;
    const ctx: Step = .{
        .world = self,
        .dt = dt,
        .inv_dt = 1 / dt,
        .joint = .{
            .dt = dt,
            .inv_dt = 1 / dt,
            .stiffness = if (self.settings.joint_hertz > 0)
                .of(.{ .hertz = self.settings.joint_hertz, .damping_ratio = self.settings.joint_damping_ratio }, dt)
            else
                .baumgarte(0, 1 / dt),
            .max_push = self.settings.max_push_speed * units,
            .slop = self.settings.linear_slop * units,
            .units_per_metre = units,
        },
    };
    self.inv_dt = ctx.inv_dt;

    const body_slots = self.bodies.slotCount();
    const shape_slots = self.shapes.slotCount();
    try self.aabbs.resize(gpa, shape_slots);

    // 0. Sleepers a joint is asking to move, woken before anything looks.
    self.wakeDrivenJoints();

    // 1. Forces into velocities.
    solver.forRange(jobs, body_slots, grain_bodies, &ctx, integrateVelocities);

    // 2. Where every moving shape is.
    solver.forRange(jobs, shape_slots, grain_bodies, &ctx, updateAabbs);

    // 3. Which boxes overlap.
    self.pairs.clearRetainingCapacity();
    try self.sweep.update(gpa, self.aabbs.items, self, acceptPair, &self.pairs);

    // 4. Where each pair touches.
    try self.manifolds.resize(gpa, self.pairs.items.len);
    solver.forRange(jobs, self.pairs.items.len, grain_pairs, &ctx, narrowPhase);

    // 5. Contacts into constraints, remembering what is touching.
    try self.prepareContacts(ctx);

    // 6. The joints that can move something, each made ready.
    try self.gatherJoints();
    solver.forRange(jobs, self.joint_refs.items.len, grain_joints, &ctx, prepareJoints);

    // 7. Both into runs that share no movable body.
    try self.colouring.assign(gpa, self.constraints.items, body_slots);
    try self.joint_colouring.assign(gpa, self.joint_refs.items, body_slots);

    // 8. Warm start, then the passes: each is the joints colour by colour,
    //    then the contacts. Every colour is a fork and a join; the
    //    overflows are walked here.
    self.warmStartAll(jobs, &ctx);
    for (0..self.settings.velocity_iterations) |_| self.solveAll(jobs, &ctx);

    // 9. The pushing apart, on the push velocities. See `contact`.
    for (0..self.settings.push_iterations) |_| self.pushAll(jobs, &ctx);

    // 10. Velocities and pushes into positions.
    solver.forRange(jobs, body_slots, grain_bodies, &ctx, integratePositions);

    // 11. What to remember, and what changed.
    try self.rememberContacts();

    // 12. Which islands rest, and which wake.
    try self.updateSleep(dt);
    self.step_count += 1;
}

/// Whether a body moves this step: an awake dynamic body, or a kinematic
/// one with somewhere to go. Only pairs with something active in them are
/// looked at; see `acceptPair`.
fn isActive(b: *const Body) bool {
    return switch (b.type) {
        .static => false,
        .kinematic => b.linear_velocity.x != 0 or b.linear_velocity.y != 0 or b.angular_velocity != 0,
        .dynamic => b.awake,
    };
}

/// Wake the bodies of every joint that is asking to move them. On one
/// thread, because two joints may share a body and both would write it.
fn wakeDrivenJoints(self: *World) void {
    const slop = self.settings.linear_slop * self.settings.units_per_metre;
    for (self.joints.slots.items) |*slot| {
        const j = &(slot.value orelse continue);
        const a = self.bodyAt(j.body_a.index);
        const b = self.bodyAt(j.body_b.index);
        if (a.awake and b.awake) continue;
        if (!joint_mod.isDriven(j, b, slop)) continue;
        a.wake();
        b.wake();
    }
}

fn integrateVelocities(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    const dt = ctx.dt;
    for (world.bodies.slots.items[begin..end]) |*slot| {
        const b = &(slot.value orelse continue);
        if (b.type != .dynamic) continue;
        // Every step's pushing starts from nothing - a sleeper's too, since
        // a contact with something awake may push it before it wakes.
        b.push_velocity = .zero;
        b.push_angular = 0;
        if (!b.awake) {
            // Asleep - unless somebody has pushed it since, or sleeping is
            // no longer allowed. A body wakes alone here; its island follows
            // at the end of the step.
            if (b.isPushed() or !b.allow_sleep or !world.settings.enable_sleep) b.wake() else continue;
        }

        const acceleration = world.gravity.scale(b.gravity_scale).mulAdd(b.force, b.inv_mass);
        b.linear_velocity = b.linear_velocity.mulAdd(acceleration, dt);
        b.angular_velocity += dt * b.inv_inertia * b.torque;

        // Damping as `v /= 1 + c dt`, which is stable for any `c` and any
        // `dt`, where `v *= 1 - c dt` goes negative for a big enough step.
        b.linear_velocity = b.linear_velocity.scale(1 / (1 + dt * b.linear_damping));
        b.angular_velocity *= 1 / (1 + dt * b.angular_damping);

        b.force = .zero;
        b.torque = 0;
    }
}

fn updateAabbs(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    for (world.shapes.slots.items[begin..end], begin..) |*slot, i| {
        const entry = &(slot.value orelse continue);
        const b = world.bodyAt(entry.body_index);
        // Asleep, it has not moved, and its box is still the one it had.
        // Everything that can wake a body without a step - a new shape,
        // `setTransform` - wakes it first, so no box here is ever stale.
        if (!b.awake) continue;
        world.aabbs.items[i] = entry.def.geometry.aabb(b.transform);
    }
}

/// Whether a pair of overlapping boxes is worth the narrow phase this step.
fn acceptPair(world: *World, a: u32, b: u32) bool {
    const ea = &world.shapes.slots.items[a].value.?;
    const eb = &world.shapes.slots.items[b].value.?;
    const ba = world.bodyAt(ea.body_index);
    const bb = world.bodyAt(eb.body_index);
    // Two sleepers, or a sleeper and a wall, keep whatever they had; see
    // `prepareContacts`.
    if (!isActive(ba) and !isActive(bb)) return false;
    return world.mayTouch(ea, eb, ba, bb);
}

/// Whether two shapes may touch at all, whatever they are doing.
fn mayTouch(world: *World, ea: *const ShapeEntry, eb: *const ShapeEntry, ba: *const Body, bb: *const Body) bool {
    if (ea.body_index == eb.body_index) return false;
    // Two things that cannot move have nothing to say to each other.
    if (ba.type != .dynamic and bb.type != .dynamic) return false;
    if (!ea.def.filter.shouldCollide(eb.def.filter)) return false;
    // Nor do two a joint holds, unless it says they should.
    if (world.no_collide.count() != 0 and world.no_collide.contains(bodyPairKey(ea.body_index, eb.body_index))) return false;
    return true;
}

fn narrowPhase(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    for (world.pairs.items[begin..end], world.manifolds.items[begin..end]) |pair, *out| {
        const ea = &world.shapes.slots.items[pair.a].value.?;
        const eb = &world.shapes.slots.items[pair.b].value.?;
        const xa = world.bodyAt(ea.body_index).transform;
        const xb = world.bodyAt(eb.body_index).transform;
        out.* = manifoldOf(&ea.def.geometry, xa, &eb.def.geometry, xb);
    }
}

/// The right pairing for two geometries, with the normal always from A to
/// B whichever way round the pairing was written.
pub fn manifoldOf(a: *const shape_mod.Geometry, xa: Transform, b: *const shape_mod.Geometry, xb: Transform) Manifold {
    return switch (a.*) {
        .circle => |ca| switch (b.*) {
            .circle => |cb| collide.circles(ca, xa, cb, xb),
            .polygon => |*pb| blk: {
                var m = collide.polygonCircle(pb, xb, ca, xa);
                m.normal = m.normal.neg();
                break :blk m;
            },
        },
        .polygon => |*pa| switch (b.*) {
            .circle => |cb| collide.polygonCircle(pa, xa, cb, xb),
            .polygon => |*pb| collide.polygons(pa, xa, pb, xb),
        },
    };
}

/// Every touching pair into `contacts`, and every one that pushes into a
/// constraint, warm-started from `previous`. Then what is still asleep.
fn prepareContacts(self: *World, ctx: Step) Error!void {
    const gpa = self.gpa;
    self.constraints.clearRetainingCapacity();
    self.contacts.clearRetainingCapacity();
    self.links.clearRetainingCapacity();

    const units = self.settings.units_per_metre;
    const bias: contact.Bias = .{
        .baumgarte = self.settings.baumgarte,
        .inv_dt = ctx.inv_dt,
        .slop = self.settings.linear_slop * units,
        .restitution_threshold = self.settings.restitution_threshold * units,
        .max_push_speed = self.settings.max_push_speed * units,
    };

    for (self.pairs.items, self.manifolds.items) |pair, *m| {
        if (m.count == 0) continue;
        const slot_a = &self.shapes.slots.items[pair.a];
        const slot_b = &self.shapes.slots.items[pair.b];
        const ea = &slot_a.value.?;
        const eb = &slot_b.value.?;
        const handle_a: ShapeId = .{ .index = pair.a, .generation = slot_a.generation };
        const handle_b: ShapeId = .{ .index = pair.b, .generation = slot_b.generation };
        const key: contact.PairKey = .{ .a = handle_a.toInt(), .b = handle_b.toInt() };
        const sensor = ea.def.sensor or eb.def.sensor;

        try self.contacts.put(gpa, key, .{
            .shape_a = handle_a,
            .shape_b = handle_b,
            .body_a = ea.body,
            .body_b = eb.body,
            .sensor = sensor,
        });
        if (sensor) continue;

        const ba = self.bodyAt(ea.body_index);
        const bb = self.bodyAt(eb.body_index);
        try self.links.append(gpa, .{ .a = ea.body_index, .b = eb.body_index, .type_a = ba.type, .type_b = bb.type });
        const warm: ?[2]contact.Impulse = if (self.previous.get(key)) |last| last.impulses else null;
        try self.constraints.append(gpa, contact.prepare(
            key,
            m,
            ba,
            bb,
            ea.body_index,
            eb.body_index,
            ea.def.material,
            eb.def.material,
            warm,
            bias,
        ));
    }

    // What was touching and is asleep - neither body moving this step -
    // stays touching, impulses and all, without being looked at: that is
    // most of what sleeping saves. It cannot also have been found above,
    // because a pair with nothing active in it is never accepted. Every
    // contact has a dynamic body in it, so with none asleep there is
    // nothing to keep and the walk is skipped.
    if (!self.anyAsleep()) return;
    var then = self.previous.iterator();
    while (then.next()) |entry| {
        const stored = entry.value_ptr;
        if (!self.keepsSleeping(stored)) continue;
        try self.contacts.put(gpa, entry.key_ptr.*, stored.*);
        if (stored.sensor) continue;
        try self.links.append(gpa, .{
            .a = stored.body_a.index,
            .b = stored.body_b.index,
            .type_a = self.bodyAt(stored.body_a.index).type,
            .type_b = self.bodyAt(stored.body_b.index).type,
        });
    }
}

/// Whether any dynamic body is asleep. A walk over the bodies rather than a
/// count kept up to date, because `Body.sleep` and `Body.wake` are the
/// body's own and cannot tell the world.
fn anyAsleep(self: *World) bool {
    for (self.bodies.slots.items) |*slot| {
        const b = &(slot.value orelse continue);
        if (!b.awake) return true;
    }
    return false;
}

/// Whether a contact from last step is asleep and still true. Anything that
/// has changed under it - a shape gone, a body moved by hand, a filter or a
/// joint that now says no - leaves it out, so it ends, and the end wakes
/// what it held.
fn keepsSleeping(self: *World, stored: *const Stored) bool {
    // The bodies first, because most contacts have something awake in
    // them and that is two loads to find out.
    const ba = self.bodies.getConst(stored.body_a) orelse return false;
    const bb = self.bodies.getConst(stored.body_b) orelse return false;
    if (isActive(ba) or isActive(bb)) return false;
    if (ba.teleported or bb.teleported) return false;
    const ea = self.shapes.getConst(stored.shape_a) orelse return false;
    const eb = self.shapes.getConst(stored.shape_b) orelse return false;
    return self.mayTouch(ea, eb, ba, bb);
}

/// Every joint that can move something this step, into `joint_refs`, in
/// slot order - which is what makes their colouring, and so the step, come
/// out the same on every run.
fn gatherJoints(self: *World) Error!void {
    self.joint_refs.clearRetainingCapacity();
    for (self.joints.slots.items, 0..) |*slot, i| {
        const j = &(slot.value orelse continue);
        // Both bodies are alive: destroying one destroys its joints.
        const index_a = j.body_a.index;
        const index_b = j.body_b.index;
        const a = self.bodyAt(index_a);
        const b = self.bodyAt(index_b);
        j.masses = joint_mod.massesOf(j, a, b);
        // Two walls joined together have nothing to say to each other, and
        // a joint asleep at both ends keeps its impulses for when it wakes.
        if (j.masses.ma == 0 and j.masses.mb == 0) continue;
        if (!isActive(a) and !isActive(b)) continue;
        try self.joint_refs.append(self.gpa, .{
            .joint = @intCast(i),
            .body_a = index_a,
            .body_b = index_b,
            .inv_mass_a = j.masses.ma,
            .inv_mass_b = j.masses.mb,
        });
    }
}

fn prepareJoints(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    for (world.joint_refs.items[begin..end]) |ref| {
        joint_mod.prepare(world.jointAt(ref.joint), world.bodyAt(ref.body_a), world.bodyAt(ref.body_b), ctx.joint);
    }
}

/// The joint in a slot. Only a step calls this, with slots it knows are live.
inline fn jointAt(self: *World, index: u32) *Joint {
    return &self.joints.slots.items[index].value.?;
}

/// One colour's constraints, for the jobs that solve them.
const Run = struct {
    step: *const Step,
    constraints: []contact.Constraint,
};

/// One colour's joints, the same.
const JointRun = struct {
    step: *const Step,
    refs: []joint_mod.Ref,
};

fn warmStartAll(self: *World, jobs: *Jobs, ctx: *const Step) void {
    for (0..solver.max_colours) |colour| {
        const run: JointRun = .{ .step = ctx, .refs = self.joint_colouring.run(colour) };
        solver.forRange(jobs, run.refs.len, grain_joints, &run, warmStartJointRun);
    }
    self.warmStartJointsHere(self.joint_colouring.run(solver.overflow));
    for (0..solver.max_colours) |colour| {
        const run: Run = .{ .step = ctx, .constraints = self.colouring.run(colour) };
        solver.forRange(jobs, run.constraints.len, grain_contacts, &run, warmStartRun);
    }
    self.warmStartRunHere(self.colouring.run(solver.overflow));
}

fn solveAll(self: *World, jobs: *Jobs, ctx: *const Step) void {
    for (0..solver.max_colours) |colour| {
        const run: JointRun = .{ .step = ctx, .refs = self.joint_colouring.run(colour) };
        solver.forRange(jobs, run.refs.len, grain_joints, &run, solveJointRun);
    }
    self.solveJointsHere(self.joint_colouring.run(solver.overflow), ctx.joint);
    self.contactPass(jobs, ctx, .solve);
}

/// A push pass: contacts only. A joint's drift goes back through its own
/// bias, which a pendulum shows costs it almost nothing - see `joint`.
fn pushAll(self: *World, jobs: *Jobs, ctx: *const Step) void {
    self.contactPass(jobs, ctx, .push);
}

fn contactPass(self: *World, jobs: *Jobs, ctx: *const Step, comptime pass: contact.Pass) void {
    for (0..solver.max_colours) |colour| {
        const run: Run = .{ .step = ctx, .constraints = self.colouring.run(colour) };
        solver.forRange(jobs, run.constraints.len, grain_contacts, &run, solveRun(pass));
    }
    self.solveRunHere(self.colouring.run(solver.overflow), pass);
}

fn warmStartRun(run: *const Run, begin: usize, end: usize) void {
    run.step.world.warmStartRunHere(run.constraints[begin..end]);
}

/// The job that solves part of a run, for one kind of pass.
///
/// A job is a plain function, and `forRange` wants it at compile time, but
/// which pass it makes is a parameter. So this is a function that *makes*
/// the function: the struct inside is a fresh type for each `pass`, its
/// `go` sees `pass` as a constant, and `solveRun(.push)` is as much a
/// compile-time value as a function written out by hand would be.
fn solveRun(comptime pass: contact.Pass) fn (*const Run, usize, usize) void {
    return struct {
        fn go(run: *const Run, begin: usize, end: usize) void {
            run.step.world.solveRunHere(run.constraints[begin..end], pass);
        }
    }.go;
}

fn warmStartRunHere(self: *World, constraints: []contact.Constraint) void {
    for (constraints) |*c| contact.warmStart(c, self.bodyAt(c.body_a), self.bodyAt(c.body_b));
}

fn solveRunHere(self: *World, constraints: []contact.Constraint, comptime pass: contact.Pass) void {
    for (constraints) |*c| contact.solve(c, self.bodyAt(c.body_a), self.bodyAt(c.body_b), pass);
}

fn warmStartJointRun(run: *const JointRun, begin: usize, end: usize) void {
    run.step.world.warmStartJointsHere(run.refs[begin..end]);
}

fn solveJointRun(run: *const JointRun, begin: usize, end: usize) void {
    run.step.world.solveJointsHere(run.refs[begin..end], run.step.joint);
}

fn warmStartJointsHere(self: *World, refs: []const joint_mod.Ref) void {
    for (refs) |ref| joint_mod.warmStart(self.jointAt(ref.joint), self.bodyAt(ref.body_a), self.bodyAt(ref.body_b));
}

fn solveJointsHere(self: *World, refs: []const joint_mod.Ref, joint_step: joint_mod.Step) void {
    for (refs) |ref| joint_mod.solve(self.jointAt(ref.joint), self.bodyAt(ref.body_a), self.bodyAt(ref.body_b), joint_step);
}

fn integratePositions(ctx: *const Step, begin: usize, end: usize) void {
    const dt = ctx.dt;
    for (ctx.world.bodies.slots.items[begin..end]) |*slot| {
        const b = &(slot.value orelse continue);
        // Whatever `setTransform` changed has been looked at by now.
        b.teleported = false;
        if (b.type == .static or !b.awake) continue;
        // The push moves it this once, and is not kept.
        b.center = b.center.mulAdd(b.linear_velocity.add(b.push_velocity), dt);
        b.angle += dt * (b.angular_velocity + b.push_angular);
        b.syncTransform();
    }
}

/// Impulses into the contact map for next step's warm start, and the
/// difference between this step's touching set and the last one's into
/// the event lists.
fn rememberContacts(self: *World) Error!void {
    const gpa = self.gpa;
    for (self.colouring.sorted.items) |*c| {
        const stored = self.contacts.getPtr(c.key) orelse continue;
        for (c.pointSlice(), 0..) |p, i| {
            stored.impulses[i] = .{ .id = p.id, .normal = p.normal_impulse, .tangent = p.tangent_impulse };
        }
    }

    self.begin_events.clearRetainingCapacity();
    self.end_events.clearRetainingCapacity();
    var now = self.contacts.iterator();
    while (now.next()) |entry| {
        if (self.previous.contains(entry.key_ptr.*)) continue;
        try self.begin_events.append(gpa, eventOf(entry.value_ptr.*));
    }
    var then = self.previous.iterator();
    while (then.next()) |entry| {
        if (self.contacts.contains(entry.key_ptr.*)) continue;
        const ended = eventOf(entry.value_ptr.*);
        try self.end_events.append(gpa, ended);
        // Something stopped holding something up, or leaning on it:
        // whatever was asleep on either side has to look again.
        if (!ended.sensor) {
            if (self.bodies.get(ended.body_a)) |b| b.wake();
            if (self.bodies.get(ended.body_b)) |b| b.wake();
        }
    }

    std.mem.swap(ContactMap, &self.contacts, &self.previous);
}

/// Find the islands, and put to sleep every one that has been still for
/// long enough, and wake every one with anything in it that is not.
///
/// An island is found with a union-find forest over the body slots: every
/// slot starts as its own root, and each contact or joint between two
/// dynamic bodies joins their trees. It is rebuilt from nothing every step
/// - a pass over the bodies, the `links` and the joints, a few nanoseconds
/// each - so there is no island to keep in step with what touches what,
/// and nothing to go stale when a contact ends. Static and kinematic bodies
/// join nothing, or one floor would make every crate on it a single island
/// that could never sleep while any of them moved.
fn updateSleep(self: *World, dt: f32) Error!void {
    if (!self.settings.enable_sleep) return;
    const gpa = self.gpa;

    // How long each moving body has been still: the speed of its centre,
    // plus the speed its spin gives its furthest edge. The push is not
    // counted - a body being eased out of another is not moving.
    const threshold = self.settings.sleep_threshold * self.settings.units_per_metre;
    for (self.bodies.slots.items) |*slot| {
        const b = &(slot.value orelse continue);
        if (b.type == .static or !b.awake) continue;
        const speed = b.linear_velocity.len() + b.extent * @abs(b.angular_velocity);
        if (!b.allow_sleep or speed > threshold) b.sleep_time = 0 else b.sleep_time += dt;
    }

    const slots = self.bodies.slotCount();
    try self.island_parent.resize(gpa, slots);
    try self.island_rest.resize(gpa, slots);
    const parent = self.island_parent.items;
    const rest = self.island_rest.items;
    for (parent, 0..) |*p, i| p.* = @intCast(i);
    @memset(rest, std.math.inf(f32));

    for (self.links.items) |l| {
        if (l.type_a == .dynamic and l.type_b == .dynamic) join(parent, l.a, l.b);
    }
    for (self.joints.slots.items) |*slot| {
        const j = &(slot.value orelse continue);
        const a = j.body_a.index;
        const b = j.body_b.index;
        if (self.bodyAt(a).type == .dynamic and self.bodyAt(b).type == .dynamic) join(parent, a, b);
    }

    // An island is as rested as its least rested member...
    for (self.bodies.slots.items, 0..) |*slot, i| {
        const b = &(slot.value orelse continue);
        if (b.type != .dynamic) continue;
        const root = find(parent, @intCast(i));
        rest[root] = @min(rest[root], b.sleep_time);
    }

    // ...and a kinematic body moving against it, or a joint driving it,
    // is a member that has not rested at all.
    const slop = self.settings.linear_slop * self.settings.units_per_metre;
    for (self.links.items) |l| {
        if (l.type_a == .kinematic or l.type_b == .kinematic) self.restAgainst(parent, rest, l.a, l.b);
    }
    for (self.joints.slots.items) |*slot| {
        const j = &(slot.value orelse continue);
        self.restAgainst(parent, rest, j.body_a.index, j.body_b.index);
        const b = self.bodyAt(j.body_b.index);
        if (joint_mod.isDriven(j, b, slop)) {
            const moved = if (b.type == .dynamic) j.body_b.index else j.body_a.index;
            rest[find(parent, moved)] = 0;
        }
    }

    for (self.bodies.slots.items, 0..) |*slot, i| {
        const b = &(slot.value orelse continue);
        if (b.type != .dynamic) continue;
        if (rest[find(parent, @intCast(i))] >= self.settings.time_to_sleep) {
            if (b.awake) b.sleep();
        } else if (!b.awake) {
            b.wake();
        }
    }
}

/// Join the islands of two body slots.
fn join(parent: []u32, a: u32, b: u32) void {
    const root_a = find(parent, a);
    const root_b = find(parent, b);
    if (root_a == root_b) return;
    // The lower slot becomes the root. Either would do; always the same one
    // keeps the forest - and so nothing at all - depending on thread timing.
    if (root_a < root_b) parent[root_b] = root_a else parent[root_a] = root_b;
}

/// The root of a slot's tree, halving the path on the way up so the next
/// walk is shorter: every node passed is pointed at its grandparent.
fn find(parent: []u32, slot: u32) u32 {
    var x = slot;
    while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    return x;
}

/// A kinematic body against a dynamic one counts, for the dynamic one's
/// island, as a member as rested as the kinematic body is - so a platform
/// that moves keeps what rides it awake, and one that has stopped does not.
fn restAgainst(self: *World, parent: []u32, rest: []f32, a: u32, b: u32) void {
    const body_a = self.bodyAt(a);
    const body_b = self.bodyAt(b);
    const island, const kinematic = if (body_a.type == .kinematic and body_b.type == .dynamic)
        .{ b, body_a }
    else if (body_b.type == .kinematic and body_a.type == .dynamic)
        .{ a, body_b }
    else
        return;
    const root = find(parent, island);
    rest[root] = @min(rest[root], kinematic.sleep_time);
}

fn eventOf(stored: Stored) ContactEvent {
    return .{
        .shape_a = stored.shape_a,
        .shape_b = stored.shape_b,
        .body_a = stored.body_a,
        .body_b = stored.body_b,
        .sensor = stored.sensor,
    };
}

/// Pairs that started touching during the last step.
pub fn beginEvents(self: *const World) []const ContactEvent {
    return self.begin_events.items;
}

/// Pairs that stopped touching during the last step - including because
/// one of them was destroyed.
pub fn endEvents(self: *const World) []const ContactEvent {
    return self.end_events.items;
}

/// How many pairs of shapes are touching right now, sensors included.
pub fn touchingCount(self: *const World) usize {
    return self.previous.count();
}

/// How many colours the last step's contacts needed. For the log.
pub fn colourCount(self: *const World) usize {
    return self.colouring.colourCount();
}

// -------------------------------------------------------------------------
// Queries
// -------------------------------------------------------------------------

/// The first thing a ray hits, from `origin` along `translation`, among
/// shapes whose filter agrees with `filter`. Null if nothing.
///
/// Every shape is asked, so this is linear in the scene; see `broadphase`
/// for why, and for what would change it.
pub fn castRay(self: *World, origin: Vec2, translation: Vec2, filter: Filter) ?RayHit {
    var best: ?RayHit = null;
    var max_fraction: f32 = 1;
    var it = self.shapes.iterator();
    while (it.next()) |entry| {
        if (!filter.shouldCollide(entry.value.def.filter)) continue;
        const b = self.bodyAt(entry.value.body_index);
        const xf = b.transform;
        const hit = rayAgainst(&entry.value.def.geometry, xf, origin, translation, max_fraction) orelse continue;
        max_fraction = hit.fraction;
        best = .{
            .shape = entry.handle,
            .body = entry.value.body,
            .point = origin.mulAdd(translation, hit.fraction),
            .normal = hit.normal,
            .fraction = hit.fraction,
        };
    }
    return best;
}

const LocalHit = struct { fraction: f32, normal: Vec2 };

fn rayAgainst(g: *const shape_mod.Geometry, xf: Transform, origin: Vec2, translation: Vec2, max_fraction: f32) ?LocalHit {
    // Into the shape's frame, where a circle is at its centre and a polygon
    // has its stored normals.
    const p1 = xf.unapply(origin);
    const d = xf.q.invRotate(translation);
    switch (g.*) {
        .circle => |c| {
            const s = p1.sub(c.center);
            const b = s.lenSq() - c.radius * c.radius;
            const rr = d.lenSq();
            const cc = s.dot(d);
            const sigma = cc * cc - rr * b;
            if (sigma < 0 or rr < std.math.floatEps(f32)) return null;
            const a = -(cc + @sqrt(sigma));
            if (a < 0 or a > max_fraction * rr) return null;
            const fraction = a / rr;
            return .{ .fraction = fraction, .normal = xf.q.rotate(s.mulAdd(d, fraction).norm()) };
        },
        .polygon => |*poly| {
            var lower: f32 = 0;
            var upper: f32 = max_fraction;
            var index: ?usize = null;
            for (poly.vertexSlice(), poly.normalSlice(), 0..) |v, n, i| {
                const numerator = n.dot(v.sub(p1));
                const denominator = n.dot(d);
                if (denominator == 0) {
                    // Parallel to this face and outside it: no way in.
                    if (numerator < 0) return null;
                } else if (denominator < 0 and numerator < lower * denominator) {
                    // Entering through this face, later than any so far.
                    lower = numerator / denominator;
                    index = i;
                } else if (denominator > 0 and numerator < upper * denominator) {
                    // Leaving through this face, earlier than any so far.
                    upper = numerator / denominator;
                }
                if (upper < lower) return null;
            }
            const i = index orelse return null;
            return .{ .fraction = lower, .normal = xf.q.rotate(poly.normals[i]) };
        },
    }
}

/// The first shape under a point, or null. Slot order, so a game that
/// wants the topmost of several should ask `overlapAabb` and choose.
pub fn overlapPoint(self: *World, point: Vec2) ?ShapeId {
    var it = self.shapes.iterator();
    while (it.next()) |entry| {
        const xf = self.bodyAt(entry.value.body_index).transform;
        const local = xf.unapply(point);
        const inside = switch (entry.value.def.geometry) {
            .circle => |c| local.distSq(c.center) <= c.radius * c.radius,
            .polygon => |*p| p.containsLocal(local),
        };
        if (inside) return entry.handle;
    }
    return null;
}

/// Call `visit(context, shape)` for every shape whose box overlaps `box`,
/// until it returns false.
pub fn overlapAabb(
    self: *World,
    box: Aabb,
    context: anytype,
    comptime visit: fn (@TypeOf(context), ShapeId) bool,
) void {
    var it = self.shapes.iterator();
    while (it.next()) |entry| {
        const xf = self.bodyAt(entry.value.body_index).transform;
        if (!entry.value.def.geometry.aabb(xf).overlaps(box)) continue;
        if (!visit(context, entry.handle)) return;
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "a body's mass comes from its shapes, and is one when they weigh nothing" {
    var world: World = .init(testing.allocator, .{});
    defer world.deinit();

    const b = try world.createBody(.{ .position = .init(3, 4) });
    try testing.expectEqual(@as(f32, 0), world.body(b).?.mass);

    const s = try world.addShape(b, .box(1, 1));
    try testing.expectApproxEqAbs(@as(f32, 4), world.body(b).?.mass, 1e-5);
    try testing.expect(world.body(b).?.center.approxEql(.init(3, 4)));

    // A second shape off to one side moves the centre of mass.
    const s2 = try world.addShape(b, .{ .geometry = .{ .circle = .{ .center = .init(4, 0), .radius = 1 } } });
    try testing.expect(world.body(b).?.local_center.x > 0);
    try testing.expectEqual(@as(u32, 2), world.body(b).?.shape_count);

    world.removeShape(s2);
    try testing.expectApproxEqAbs(@as(f32, 4), world.body(b).?.mass, 1e-5);
    try testing.expectEqual(@as(u32, 1), world.body(b).?.shape_count);
    try testing.expect(world.shape(s2) == null);

    world.shape(s).?.def.material.density = 0;
    world.updateMass(b);
    try testing.expectEqual(@as(f32, 1), world.body(b).?.mass);

    world.destroyBody(b);
    try testing.expect(world.body(b) == null);
    try testing.expect(world.shape(s) == null);
    try testing.expectEqual(@as(usize, 0), world.shapeCount());
}
