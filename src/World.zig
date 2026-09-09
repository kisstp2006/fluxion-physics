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
//! **A step is nine phases**, and six of them run on every core:
//!
//! | Phase | What | On |
//! | --- | --- | --- |
//! | integrate velocities | gravity, forces, damping | every core |
//! | update boxes | each shape's box in the world | every core |
//! | broad phase | which boxes overlap | one |
//! | narrow phase | where each pair touches | every core |
//! | prepare | contacts into constraints, warm-started | one |
//! | colour | contacts into runs that share no body | one |
//! | solve | warm start, then the velocity passes | every core, per colour |
//! | integrate positions | move, turn, rebuild transforms | every core |
//! | remember | impulses for next step, begin and end events | one |
//!
//! With a scheduler that has no workers - a browser - the same phases run
//! on the calling thread in the same order. Nothing is compiled out.
//!
//! **Handles, not pointers.** A body is named by a `Body.Id` and a shape by
//! a `ShapeId`, both generational: a handle to something destroyed answers
//! null, for ever, however many times its slot is reused. `body(id)` hands
//! out a pointer for writing velocity and applying forces; hold the handle
//! and ask again next frame.
//!
//! **Units are yours.** Set `Settings.units_per_metre` to what one metre is
//! in your coordinates - a hundred, for a game that thinks in pixels - and
//! the tolerances the solver needs, which are in metres, are scaled by it.
//! Gravity defaults to Earth's, downwards, in those units.
//!
//! **What is not here yet**: joints, sleeping, continuous collision, and a
//! tree for queries. Each is listed in the README with what it would take.

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
const solver = @import("solver.zig");

const World = @This();

pub const BodyId = Body.Id;
pub const ShapeId = Body.ShapeId;

pub const Settings = struct {
    /// Null is Earth's, pulling towards `+y`, which is down on this screen.
    /// Set it to anything, including zero for space.
    gravity: ?Vec2 = null,
    /// What a metre is in your coordinates. Everything below that is a
    /// length or a speed is in metres and scaled by this.
    units_per_metre: f32 = 1,
    /// Passes the solver makes over the contacts. Eight is Box2D's default;
    /// a tall stack wants more, a game of loose balls fewer.
    velocity_iterations: u32 = 8,
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

gpa: Allocator,
settings: Settings,
/// The acceleration every dynamic body feels, times its `gravity_scale`.
/// Change it whenever.
gravity: Vec2,

bodies: id.Table(Body) = .empty,
shapes: id.Table(ShapeEntry) = .empty,
sweep: broadphase.Sweep(*World) = .empty,

// Per-step scratch, kept so a step allocates only when the scene grows.
aabbs: std.ArrayList(Aabb) = .empty,
pairs: std.ArrayList(broadphase.Pair) = .empty,
manifolds: std.ArrayList(Manifold) = .empty,
constraints: std.ArrayList(contact.Constraint) = .empty,
colouring: solver.Colouring = .empty,

/// What was touching at the end of the last step, with the impulses that
/// held it there, and the same for the step before. Swapped every step.
contacts: ContactMap = .empty,
previous: ContactMap = .empty,
begin_events: std.ArrayList(ContactEvent) = .empty,
end_events: std.ArrayList(ContactEvent) = .empty,

/// How many steps have run. A frame counter for the log.
step_count: u64 = 0,

const ContactMap = std.AutoHashMapUnmanaged(contact.PairKey, Stored);

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
    self.sweep.deinit(gpa);
    self.aabbs.deinit(gpa);
    self.pairs.deinit(gpa);
    self.manifolds.deinit(gpa);
    self.constraints.deinit(gpa);
    self.colouring.deinit(gpa);
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

/// Take a body and every shape on it out of the world. Contacts it was in
/// end next step, and every handle to it answers null from now on.
pub fn destroyBody(self: *World, handle: BodyId) void {
    const doomed = self.bodies.get(handle) orelse return;
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
pub fn addShape(self: *World, body_handle: BodyId, def: Shape) Error!ShapeId {
    const owner = self.bodies.get(body_handle) orelse return error.OutOfMemory;
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
/// dynamic was meant to move.
pub fn updateMass(self: *World, handle: BodyId) void {
    const b = self.bodies.get(handle) orelse return;
    b.mass = 0;
    b.inv_mass = 0;
    b.inertia = 0;
    b.inv_inertia = 0;
    b.local_center = .zero;

    if (b.type != .dynamic) {
        b.center = b.transform.p;
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

/// What every phase of a step is handed.
const Step = struct {
    world: *World,
    dt: f32,
    inv_dt: f32,
};

/// Advance the world by `dt` seconds, on `jobs`.
///
/// Call it with the same `dt` every time - a fixed step - because a solver
/// tuned by a bias per step behaves differently at a different step, and
/// because the same inputs then give the same world on every machine.
pub fn step(self: *World, dt: f32, jobs: *Jobs) Error!void {
    if (dt <= 0) return;
    const gpa = self.gpa;
    const ctx: Step = .{ .world = self, .dt = dt, .inv_dt = 1 / dt };

    const body_slots = self.bodies.slotCount();
    const shape_slots = self.shapes.slotCount();
    try self.aabbs.resize(gpa, shape_slots);

    // 1. Forces into velocities.
    solver.forRange(jobs, body_slots, grain_bodies, &ctx, integrateVelocities);

    // 2. Where every shape is.
    solver.forRange(jobs, shape_slots, grain_bodies, &ctx, updateAabbs);

    // 3. Which boxes overlap.
    self.pairs.clearRetainingCapacity();
    try self.sweep.update(gpa, self.aabbs.items, self, acceptPair, &self.pairs);

    // 4. Where each pair touches.
    try self.manifolds.resize(gpa, self.pairs.items.len);
    solver.forRange(jobs, self.pairs.items.len, grain_pairs, &ctx, narrowPhase);

    // 5. Contacts into constraints, remembering what is touching.
    try self.prepareContacts(ctx);

    // 6. Into runs that share no movable body.
    try self.colouring.assign(gpa, self.constraints.items, body_slots);

    // 7. Warm start, then the passes. Each colour is a fork and a join;
    //    the overflow is walked here.
    for (0..solver.max_colours) |colour| {
        const run = self.colouring.run(colour);
        const run_ctx: Run = .{ .step = &ctx, .constraints = run };
        solver.forRange(jobs, run.len, grain_contacts, &run_ctx, warmStartRun);
    }
    self.warmStartRunHere(self.colouring.run(solver.overflow));
    for (0..self.settings.velocity_iterations) |_| {
        for (0..solver.max_colours) |colour| {
            const run = self.colouring.run(colour);
            const run_ctx: Run = .{ .step = &ctx, .constraints = run };
            solver.forRange(jobs, run.len, grain_contacts, &run_ctx, solveRun);
        }
        self.solveRunHere(self.colouring.run(solver.overflow));
    }

    // 8. Velocities into positions.
    solver.forRange(jobs, body_slots, grain_bodies, &ctx, integratePositions);

    // 9. What to remember, and what changed.
    try self.rememberContacts();
    self.step_count += 1;
}

fn integrateVelocities(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    const dt = ctx.dt;
    for (world.bodies.slots.items[begin..end]) |*slot| {
        const b = &(slot.value orelse continue);
        if (b.type != .dynamic) continue;

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
        const xf = world.bodyAt(entry.body_index).transform;
        world.aabbs.items[i] = entry.def.geometry.aabb(xf);
    }
}

/// Whether a pair of overlapping boxes is worth the narrow phase.
fn acceptPair(world: *World, a: u32, b: u32) bool {
    const ea = &world.shapes.slots.items[a].value.?;
    const eb = &world.shapes.slots.items[b].value.?;
    if (ea.body_index == eb.body_index) return false;
    const ba = world.bodyAt(ea.body_index);
    const bb = world.bodyAt(eb.body_index);
    // Two things that cannot move have nothing to say to each other.
    if (ba.type != .dynamic and bb.type != .dynamic) return false;
    return ea.def.filter.shouldCollide(eb.def.filter);
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
/// constraint, warm-started from `previous`.
fn prepareContacts(self: *World, ctx: Step) Error!void {
    const gpa = self.gpa;
    self.constraints.clearRetainingCapacity();
    self.contacts.clearRetainingCapacity();

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

        const warm: ?[2]contact.Impulse = if (self.previous.get(key)) |last| last.impulses else null;
        try self.constraints.append(gpa, contact.prepare(
            key,
            m,
            self.bodyAt(ea.body_index),
            self.bodyAt(eb.body_index),
            ea.body_index,
            eb.body_index,
            ea.def.material,
            eb.def.material,
            warm,
            bias,
        ));
    }
}

/// One colour's constraints, for the jobs that solve them.
const Run = struct {
    step: *const Step,
    constraints: []contact.Constraint,
};

fn warmStartRun(run: *const Run, begin: usize, end: usize) void {
    run.step.world.warmStartRunHere(run.constraints[begin..end]);
}

fn solveRun(run: *const Run, begin: usize, end: usize) void {
    run.step.world.solveRunHere(run.constraints[begin..end]);
}

fn warmStartRunHere(self: *World, constraints: []contact.Constraint) void {
    for (constraints) |*c| contact.warmStart(c, self.bodyAt(c.body_a), self.bodyAt(c.body_b));
}

fn solveRunHere(self: *World, constraints: []contact.Constraint) void {
    for (constraints) |*c| contact.solve(c, self.bodyAt(c.body_a), self.bodyAt(c.body_b));
}

fn integratePositions(ctx: *const Step, begin: usize, end: usize) void {
    const dt = ctx.dt;
    for (ctx.world.bodies.slots.items[begin..end]) |*slot| {
        const b = &(slot.value orelse continue);
        if (b.type == .static) continue;
        b.center = b.center.mulAdd(b.linear_velocity, dt);
        b.angle += dt * b.angular_velocity;
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
        try self.end_events.append(gpa, eventOf(entry.value_ptr.*));
    }

    std.mem.swap(ContactMap, &self.contacts, &self.previous);
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
