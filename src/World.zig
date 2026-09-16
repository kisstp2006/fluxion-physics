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
//! **A step finds what touches once, and then solves it in substeps**
//! (`Settings.substeps`, four by default). Most phases run on every core:
//!
//! | Phase | What | On |
//! | --- | --- | --- |
//! | wake | sleepers a joint drives, or somebody pushed; where each starts | one, then every core |
//! | update boxes | each moving shape's box in the world | every core |
//! | broad phase | which boxes overlap, of pairs with something moving: the sweep, then the level's tree | one |
//! | narrow phase | where each pair touches | every core |
//! | prepare contacts | into constraints, warm-started; sleeping ones kept | one |
//! | colour | contacts and joints into runs that share no body | one |
//! | *each substep:* integrate velocities | gravity, forces, damping, a speed limit | every core |
//! | *each substep:* warm start and solve | joints, then contacts, pushing back towards whole | every core, per colour |
//! | *each substep:* integrate positions | move, turn; then the joints measured again | every core |
//! | *each substep:* relax | the pushing taken back out of the velocities | every core, per colour |
//! | restitution | bounces | every core, per colour |
//! | sweep | fast bodies along their paths, back to where they first touched | every core |
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
//! **The level is in a tree.** Every shape on a static body is a leaf of
//! `static_tree`, and everything else is in the sweep. Each awake dynamic
//! shape asks the tree what is near it, and the queries ask the tree for the
//! level and walk only the shapes that move. So a level of thousands of
//! tiles costs a step what is near the moving bodies - and the passes over
//! bodies walk only `movers`, so it costs nothing for being thousands of
//! bodies either. Its seams are smooth: an edge of a piece that another
//! piece covers makes no contacts, so a box slides across a floor of tiles
//! as across one slab. See `coverEdges`.
//!
//! **Fast bodies are swept.** A step that only looks at where bodies are
//! lets one pass through a wall thinner than its step. So after the
//! substeps, every body that moved more than half its thinnest extent is
//! swept along its path, against the level and - for a `Body.Def.bullet` -
//! the other moving bodies, and put back where it first touched anything it
//! should not have gone into. See `continuous` for how, and for why a touch
//! that is only a graze, such as the next tile of a floor, is left alone.

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
const Tree = @import("tree.zig");
const contact = @import("contact.zig");
const continuous = @import("continuous.zig");
const joint_mod = @import("joint.zig");
const Joint = joint_mod.Joint;
const solver = @import("solver.zig");
const Softness = @import("Softness.zig");

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
    /// How many pieces a step is cut into. Each is a whole small step -
    /// forces, one pass, movement, one relaxing pass - so four substeps are
    /// eight passes, and cost about what eight passes over one step would,
    /// but every one of them starts from where the bodies really are. That
    /// is what holds a heavy crate on a light bridge. Box2D v3's default;
    /// more for a wrecking ball on a thin chain, fewer for loose balls.
    substeps: u32 = 4,
    /// How far a shape may sink before it is pushed out. Metres.
    linear_slop: f32 = 0.005,
    /// Slower than this, nothing bounces. Metres per second.
    restitution_threshold: f32 = 1,
    /// The fastest a penetration is pushed apart. Metres per second. What
    /// keeps a body made inside another from leaving at the speed of a
    /// bullet.
    max_push_speed: f32 = 3,
    /// How a contact takes back sinking in: the way a spring would that
    /// rang this many times a second, damped this many times past bouncing.
    /// Against something that cannot move, twice as stiff. Box2D v3's
    /// numbers, and capped at a quarter of the substep rate, past which a
    /// substep cannot follow the spring. See `Softness`.
    contact_hertz: f32 = 30,
    contact_damping_ratio: f32 = 10,
    /// The same for a rigid joint's drift. Stiff enough that nobody sees
    /// the give; a spring rather than a fixed fraction per step because the
    /// fraction, fed back as speed, pumps energy into a light chain holding
    /// a heavy weight. Zero hertz holds joints rigid and lets them drift.
    joint_hertz: f32 = 60,
    joint_damping_ratio: f32 = 2,
    /// The fastest anything may move. Metres per second. Not a speed a game
    /// should reach - a sanity cap, so a body flung by something that went
    /// wrong leaves the level rather than turning to infinity.
    max_speed: f32 = 400,
    /// Whether anything sleeps at all. Turning it off wakes what sleeps.
    enable_sleep: bool = true,
    /// Slower than this counts as still. Metres per second, of the body's
    /// centre plus what its spin does to its furthest edge - one number for
    /// a pebble and a boulder, which a separate angular one would not be.
    sleep_threshold: f32 = 0.05,
    /// How long an island must be still before it sleeps. Seconds.
    time_to_sleep: f32 = 0.5,
    /// Whether a body that moved more than half its thinnest extent in a
    /// step is swept along its path for the level it went through - and a
    /// `Body.Def.bullet` for the moving bodies too. Off, a fast ball passes
    /// through a wall thinner than its step. See `continuous`.
    enable_continuous: bool = true,
    /// Which bits two shapes that push need to touch: both masks having the
    /// other's category, Box2D's, or either, Godot 3's. A sensor's pair
    /// always takes either; see `Filter.shouldSense`.
    filter_rule: shape_mod.FilterRule = .both,
    /// How a contact's friction and restitution come from its two surfaces'.
    /// Box2D's by default; Godot 3's are `.minimum` and `.sum_clamped`.
    friction_mix: shape_mod.Mix = .geometric_mean,
    restitution_mix: shape_mod.Mix = .maximum,
};

/// A shape as the world keeps it: the definition, and where it hangs.
pub const ShapeEntry = struct {
    def: Shape,
    body: BodyId,
    /// The body's slot, so a step reads it without resolving the handle.
    body_index: u32,
    /// The next shape on the same body. See `Body.first_shape`.
    next: ShapeId,
    /// Its leaf in `static_tree`, for a shape on a static body; `null_node`
    /// for one that moves, which is in the sweep instead.
    proxy: u32 = Tree.null_node,
    /// For a polygon of the level, the edges another piece of the level
    /// covers, which make no contacts. See `coverEdges`.
    hidden: collide.Hidden = collide.none_hidden,
    /// Whether `hidden` has been worked out since the level around this
    /// shape last changed. Worked out when something first comes near.
    hidden_known: bool = false,
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
/// The shapes that move - on dynamic and kinematic bodies - sorted along x.
sweep: broadphase.Sweep(*World) = .empty,
/// The shapes that do not: the level. Each moving shape asks it what is
/// near, and the queries ask it first. See `Tree`.
static_tree: Tree = .empty,

/// Pairs of bodies a joint says must not collide, by slot, with how many
/// joints say so. Asked by the broad phase for every pair it finds while
/// there is anything in it; see `bodyPairKey`.
no_collide: std.AutoHashMapUnmanaged(u64, u32) = .empty,

/// Pairs of bodies told never to touch - Godot's collision exceptions - by
/// slot, with how many times each was asked. Apart from `no_collide` so a
/// destroyed body takes its exceptions with it, where a joint goes anyway.
exceptions: std.AutoHashMapUnmanaged(u64, u32) = .empty,

/// The slots of every body that is not static, in slot order: what the
/// passes over bodies walk. A level built of a body per tile is thousands
/// of bodies that never move, and a step that visited each of them a dozen
/// times - forces, movement, sleep - spent more time on the level than on
/// the game. Rebuilt when a body comes or goes, which `movers_dirty` says.
movers: std.ArrayList(u32) = .empty,
movers_dirty: bool = false,
/// Static bodies moved by `setTransform` since the last step, whose flag
/// is cleared once the step has looked at it.
moved_level: std.ArrayList(u32) = .empty,

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
/// One over the last step's substep, for turning impulses into forces: an
/// impulse a joint keeps is one substep's.
inv_h: f32 = 0,

/// What was touching at the end of the last step, with the impulses that
/// held it there, and the same for the step before. Swapped every step.
contacts: ContactMap = .empty,
previous: ContactMap = .empty,
/// The pairs a one-way shape is letting through, this step and the last,
/// swapped with `contacts`: overlapping, pushing nothing, reporting
/// nothing, until they stop overlapping. See `holdsOneWay`.
one_way_off: ContactMap = .empty,
one_way_before: ContactMap = .empty,
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
    self.static_tree.deinit(gpa);
    self.movers.deinit(gpa);
    self.moved_level.deinit(gpa);
    self.no_collide.deinit(gpa);
    self.exceptions.deinit(gpa);
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
    self.one_way_off.deinit(gpa);
    self.one_way_before.deinit(gpa);
    self.begin_events.deinit(gpa);
    self.end_events.deinit(gpa);
    self.* = undefined;
}

// -------------------------------------------------------------------------
// Bodies
// -------------------------------------------------------------------------

pub fn createBody(self: *World, def: Body.Def) Error!BodyId {
    if (def.type != .static) self.movers_dirty = true;
    return self.bodies.add(self.gpa, .fromDef(def));
}

/// The slots of the bodies that move, fresh. See `movers`.
fn refreshMovers(self: *World) Error!void {
    if (!self.movers_dirty) return;
    self.movers.clearRetainingCapacity();
    for (self.bodies.slots.items, 0..) |*slot, i| {
        const b = &(slot.value orelse continue);
        if (b.type != .static) try self.movers.append(self.gpa, @intCast(i));
    }
    self.movers_dirty = false;
}

/// Take a body, every shape on it and every joint to it out of the world.
/// Contacts it was in end next step, and every handle to it answers null
/// from now on.
pub fn destroyBody(self: *World, handle: BodyId) void {
    const doomed = self.bodies.get(handle) orelse return;
    if (doomed.type != .static) self.movers_dirty = true;

    // A linear walk, because a body does not keep a list of its joints:
    // destroying is rare beside stepping, and a list would be two more
    // links in every joint to keep straight.
    for (self.joints.slots.items, 0..) |*slot, i| {
        const j = &(slot.value orelse continue);
        if (j.body_a.eql(handle) or j.body_b.eql(handle)) {
            self.destroyJoint(.{ .index = @intCast(i), .generation = slot.generation });
        }
    }

    // Its exceptions go with it, or the next body in its slot would have
    // them.
    if (self.exceptions.count() != 0) {
        var it = self.exceptions.keyIterator();
        while (it.next()) |key| {
            if (key.* >> 32 == handle.index or key.* & 0xFFFF_FFFF == handle.index) self.exceptions.removeByPtr(key);
        }
    }

    var next = doomed.first_shape;
    while (self.shapes.get(next)) |entry| {
        const after = entry.next;
        self.unenrol(next.index, entry);
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
    try self.enrol(handle.index, self.shapes.get(handle).?, owner);
    owner.first_shape = handle;
    owner.shape_count += 1;
    self.measure(body_handle);
    return handle;
}

/// Put a shape where the broad phase will find it: a static body's in the
/// level's tree, anything that moves in the sweep.
fn enrol(self: *World, index: u32, entry: *ShapeEntry, owner: *const Body) Allocator.Error!void {
    if (owner.type == .static) {
        const box = entry.def.geometry.aabb(owner.transform);
        // The pieces it now touches may have edges it covers.
        self.forgetCoverNear(box);
        entry.proxy = try self.static_tree.insert(self.gpa, box, index);
    } else {
        try self.sweep.add(self.gpa, index);
    }
}

/// And take it out again.
fn unenrol(self: *World, index: u32, entry: *const ShapeEntry) void {
    if (entry.proxy != Tree.null_node) {
        // The pieces it touched may have edges it covered.
        const box = self.static_tree.boxOf(entry.proxy);
        self.static_tree.remove(entry.proxy);
        self.forgetCoverNear(box);
    } else {
        self.sweep.remove(index);
    }
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
    self.unenrol(handle.index, entry);
    _ = self.shapes.remove(handle);
    self.measure(owner_handle);
}

/// The shape, to read or to change its material or filter. Changing its
/// geometry or density is allowed too; call `updateMass` on the body after,
/// and after changing the filter of a shape of the level.
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
/// `removeShape`; call it yourself after changing a shape's density - or,
/// on a static body, a shape's geometry or filter, which moves it in the
/// level's tree and changes the seams around it: both are worked out again
/// at the next step.
///
/// A dynamic body with no mass - no shapes, or shapes of zero density -
/// gets a mass of one rather than infinity, because a body somebody made
/// dynamic was meant to move. And it wakes: heavier or lighter, it has to
/// find its feet again.
pub fn updateMass(self: *World, handle: BodyId) void {
    self.measure(handle);
    // As though it had been moved by hand; see `refitMovedLevel`.
    if (self.bodies.get(handle)) |b| {
        if (b.type == .static) b.teleported = true;
    }
}

/// `updateMass` without the level's refitting, for `addShape` and
/// `removeShape`, which put the shape in the tree or take it out themselves.
fn measure(self: *World, handle: BodyId) void {
    const b = self.bodies.get(handle) orelse return;
    b.mass = 0;
    b.inv_mass = 0;
    b.inertia = 0;
    b.inv_inertia = 0;
    b.local_center = .zero;

    if (b.type != .dynamic) {
        b.center = b.transform.p;
        // A static body is never swept, and never asked about by the circle
        // round it - its shapes' boxes are in the level's tree. And a level
        // built as one body of thousands of tiles would add them all up
        // again for every tile added.
        if (b.type == .kinematic) {
            const extents = self.extentsOf(b);
            b.extent = extents.reach;
            b.min_extent = extents.thinnest;
        }
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
    const extents = self.extentsOf(b);
    b.extent = extents.reach;
    b.min_extent = extents.thinnest;
    b.wake();
}

/// How far a body's shapes reach from its centre of mass, and how thin the
/// thinnest of them is. See `Body.extent` and `Body.min_extent`.
fn extentsOf(self: *World, b: *const Body) struct { reach: f32, thinnest: f32 } {
    var reach: f32 = 0;
    var thinnest: ?f32 = null;
    var cursor = b.first_shape;
    while (self.shapes.get(cursor)) |entry| : (cursor = entry.next) {
        reach = @max(reach, entry.def.geometry.reach(b.local_center));
        const thin = entry.def.geometry.minExtent();
        thinnest = if (thinnest) |t| @min(t, thin) else thin;
    }
    return .{ .reach = reach, .thinnest = thinnest orelse 0 };
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
    return j.reaction(self.inv_h);
}

/// Keep two bodies from ever touching, whatever their filters say, until
/// `removeCollisionException`: Godot's `add_collision_exception_with`.
/// Counted, so two calls take two removals. What they touch now ends at
/// the next step, and both wake.
pub fn addCollisionException(self: *World, a: BodyId, b: BodyId) JointError!void {
    if (self.bodies.getConst(a) == null or self.bodies.getConst(b) == null) return error.NoSuchBody;
    if (a.eql(b)) return error.SameBody;
    const entry = try self.exceptions.getOrPut(self.gpa, bodyPairKey(a.index, b.index));
    entry.value_ptr.* = if (entry.found_existing) entry.value_ptr.* + 1 else 1;
    self.bodyAt(a.index).wake();
    self.bodyAt(b.index).wake();
}

/// Take one `addCollisionException` back. The two may touch again once
/// none is left and nothing else keeps them apart.
pub fn removeCollisionException(self: *World, a: BodyId, b: BodyId) void {
    if (self.bodies.getConst(a) == null or self.bodies.getConst(b) == null) return;
    const key = bodyPairKey(a.index, b.index);
    const count = self.exceptions.getPtr(key) orelse return;
    count.* -= 1;
    if (count.* == 0) _ = self.exceptions.remove(key);
    self.bodyAt(a.index).wake();
    self.bodyAt(b.index).wake();
}

/// Whether an `addCollisionException` keeps the two apart.
pub fn hasCollisionException(self: *const World, a: BodyId, b: BodyId) bool {
    if (self.bodies.getConst(a) == null or self.bodies.getConst(b) == null) return false;
    return self.exceptions.contains(bodyPairKey(a.index, b.index));
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
    /// One substep, and one over it.
    h: f32,
    inv_h: f32,
    contact: contact.Step,
    /// The spring a contact takes back sinking in with, and the stiffer one
    /// against something that cannot move.
    contact_softness: Softness,
    static_softness: Softness,
    /// What joints are solved with: their spring while solving, rigid and
    /// asking nothing back while relaxing.
    joint: joint_mod.Step,
    joint_relax: joint_mod.Step,
    /// The fastest anything may move, in world units per second, and turn,
    /// in radians per substep. See `integrateVelocities`.
    max_speed: f32,
    max_turn: f32,
    /// How far ahead a body stopped short looks for what it touches.
    speculative: f32,
    /// The continuous pass: how close a first touch is, give or take, and
    /// how deep the rest of a path may sink in and still be a graze. See
    /// `continuous`. All in world units.
    toi_target: f32,
    toi_tolerance: f32,
    graze: f32,
    /// How deep a contact against the level may be and still be one made
    /// at a seam between two of its pieces. See `collide.Seams`.
    seam_depth: f32,
};

/// Advance the world by `dt` seconds, on `jobs`.
///
/// Call it with the same `dt` every time - a fixed step - because a solver
/// tuned by springs of so many hertz behaves differently at a different
/// step, and because the same inputs then give the same world on every
/// machine.
///
/// **A step is cut into substeps** (`Settings.substeps`), and each substep
/// is a whole small step of its own: forces into velocities, the joints
/// measured again from where the bodies are, one pass pushing everything
/// that is broken back towards whole, the bodies moved, and one relaxing
/// pass taking the pushing back out of the velocities. Which pairs touch,
/// and where, is found once per step, before the substeps - that is the
/// expensive part, and a substep's worth of movement does not change it.
///
/// Why substeps and not more passes: a pass works on a picture of the
/// world, and the picture is taken when the pass starts. Eight passes over
/// one picture of a chain holding a heavy weight solve, very precisely, the
/// chain as it was - and the weight has moved on. Four substeps take four
/// pictures, each of the world as it is. A crate ten times a plank's weight
/// dropped on a light bridge tore the pins 60 pixels apart the first way;
/// see `joint_test` for what it does now. Erin Catto's "soft step", Box2D
/// v3's solver.
pub fn step(self: *World, dt: f32, jobs: *Jobs) Error!void {
    if (dt <= 0) return;
    const gpa = self.gpa;
    const units = self.settings.units_per_metre;
    const substeps = @max(self.settings.substeps, 1);
    const h = dt / @as(f32, @floatFromInt(substeps));
    const inv_h = 1 / h;
    // A spring faster than a quarter of the substep rate is one the
    // substeps cannot follow; Box2D v3's cap.
    const fastest = 0.25 * inv_h;
    const contact_hertz = @min(self.settings.contact_hertz, fastest);
    const joint_hertz = @min(self.settings.joint_hertz, fastest);
    const slop = self.settings.linear_slop * units;
    const joint_step: joint_mod.Step = .{
        .dt = h,
        .inv_dt = inv_h,
        .stiffness = if (joint_hertz > 0) .of(joint_hertz, self.settings.joint_damping_ratio, h) else .rigid,
        .slop = slop,
        .units_per_metre = units,
    };
    var joint_relax = joint_step;
    joint_relax.stiffness = .rigid;
    const ctx: Step = .{
        .world = self,
        .dt = dt,
        .h = h,
        .inv_h = inv_h,
        .contact = .{
            .inv_h = inv_h,
            .slop = slop,
            .max_push = self.settings.max_push_speed * units,
        },
        .contact_softness = .of(contact_hertz, self.settings.contact_damping_ratio, h),
        .static_softness = .of(2 * contact_hertz, self.settings.contact_damping_ratio, h),
        .joint = joint_step,
        .joint_relax = joint_relax,
        .max_speed = self.settings.max_speed * units,
        .max_turn = 0.25 * std.math.pi,
        // Box2D v3's numbers: a first touch a slop short, found to within a
        // quarter of one, and a look four slops ahead.
        .speculative = 4 * slop,
        .toi_target = slop,
        .toi_tolerance = 0.25 * slop,
        .graze = 4 * slop,
        // A box sliding over a seam has sunk the slop a resting box sinks,
        // a little more under a load; four is room for both.
        .seam_depth = 4 * slop,
    };
    self.inv_h = inv_h;

    const body_slots = self.bodies.slotCount();
    try self.aabbs.resize(gpa, self.shapes.slotCount());
    try self.refreshMovers();
    const movers = self.movers.items.len;

    // 0. Sleepers a joint is asking to move, or somebody has pushed, woken
    //    before anything looks, and where everything starts from noted.
    self.wakeDrivenJoints();
    solver.forRange(jobs, movers, grain_bodies, &ctx, beginStep);

    // 1. Where every moving shape is, and where a level moved by hand now
    //    is in its tree.
    solver.forRange(jobs, self.sweep.order.items.len, grain_bodies, &ctx, updateAabbs);
    try self.refitMovedLevel();

    // 2. Which boxes overlap: moving shapes against each other, by the
    //    sweep, and against the level, by asking its tree.
    self.pairs.clearRetainingCapacity();
    try self.sweep.update(gpa, self.aabbs.items, self, acceptPair, &self.pairs);
    try self.findLevelPairs();

    // 3. Where each pair touches.
    try self.manifolds.resize(gpa, self.pairs.items.len);
    solver.forRange(jobs, self.pairs.items.len, grain_pairs, &ctx, narrowPhase);

    // 4. Contacts into constraints, remembering what is touching. A level
    //    moved by hand has been looked at by now, sleeping contacts and all.
    try self.prepareContacts(ctx);
    for (self.moved_level.items) |index| self.bodyAt(index).teleported = false;
    self.moved_level.clearRetainingCapacity();

    // 5. The joints that can move something.
    try self.gatherJoints();

    // 6. Both into runs that share no movable body.
    try self.colouring.assign(gpa, self.constraints.items, body_slots);
    try self.joint_colouring.assign(gpa, self.joint_refs.items, body_slots);

    // 7. The substeps. Each pass is the joints colour by colour, then the
    //    contacts; every colour is a fork and a join, and the overflows are
    //    walked here. The joints are measured before the first pass and
    //    again after every move, so the relaxing and the next substep both
    //    see where the bodies are.
    solver.forRange(jobs, self.joint_refs.items.len, grain_joints, &ctx, prepareJoints);
    for (0..substeps) |_| {
        solver.forRange(jobs, movers, grain_bodies, &ctx, integrateVelocities);
        self.warmStartAll(jobs, &ctx);
        self.solveAll(jobs, &ctx, .solve);
        solver.forRange(jobs, movers, grain_bodies, &ctx, integratePositions);
        solver.forRange(jobs, self.joint_refs.items.len, grain_joints, &ctx, prepareJoints);
        self.solveAll(jobs, &ctx, .relax);
    }

    // 8. Bounce, and forget the forces, which pushed for the whole step.
    self.restituteAll(jobs, &ctx);
    solver.forRange(jobs, movers, grain_bodies, &ctx, clearForces);

    // 9. Every body that moved fast swept along its path, and put back
    //    where it first touched anything it went into: the rest first, then
    //    the bullets, which sweep against the rest where they now are.
    solver.forRange(jobs, movers, grain_bodies, &ctx, sweepRun(false));
    solver.forRange(jobs, movers, grain_bodies, &ctx, sweepRun(true));

    // 10. What to remember, and what changed.
    try self.rememberContacts();

    // 11. Which islands rest, and which wake.
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

/// A sleeper that somebody has pushed since it fell asleep - or that may
/// not sleep any more - wakes. Alone: its island follows at the end of the
/// step. And every dynamic body notes where it starts from, which is where
/// the continuous pass sweeps it from.
fn beginStep(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    for (world.movers.items[begin..end]) |index| {
        const b = world.bodyAt(index);
        if (b.type != .dynamic) continue;
        if (!b.awake and (b.isPushed() or !b.allow_sleep or !world.settings.enable_sleep)) b.wake();
        b.center0 = b.center;
        b.angle0 = b.angle;
    }
}

/// One substep's worth of gravity, forces and damping.
fn integrateVelocities(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    const h = ctx.h;
    for (world.movers.items[begin..end]) |index| {
        const b = world.bodyAt(index);
        if (b.type != .dynamic or !b.awake) continue;

        const acceleration = world.gravity.scale(b.gravity_scale).mulAdd(b.force, b.inv_mass);
        b.linear_velocity = b.linear_velocity.mulAdd(acceleration, h);
        b.angular_velocity += h * b.inv_inertia * b.torque;

        // Damping as `v /= 1 + c h`, which is stable for any `c` and any
        // `h`, where `v *= 1 - c h` goes negative for a big enough step.
        b.linear_velocity = b.linear_velocity.scale(1 / (1 + h * b.linear_damping));
        b.angular_velocity *= 1 / (1 + h * b.angular_damping);

        // Past a quarter turn a substep, a joint's or a contact's lever arm
        // has swung too far for the solver's straight-line picture of it to
        // be any use, and a light link whipped by a heavy weight spins
        // itself apart. Box2D's limit; at four substeps of a sixtieth it is
        // thirty turns a second, which nothing a game means to do reaches.
        const turn = h * b.angular_velocity;
        if (@abs(turn) > ctx.max_turn) b.angular_velocity *= ctx.max_turn / @abs(turn);
        const speed_sq = b.linear_velocity.lenSq();
        if (speed_sq > ctx.max_speed * ctx.max_speed) {
            b.linear_velocity = b.linear_velocity.scale(ctx.max_speed / @sqrt(speed_sq));
        }
    }
}

/// Forces push for a whole step, every substep of it, and are then gone.
fn clearForces(ctx: *const Step, begin: usize, end: usize) void {
    for (ctx.world.movers.items[begin..end]) |index| {
        const b = ctx.world.bodyAt(index);
        b.force = .zero;
        b.torque = 0;
    }
}

/// The box of every moving shape - the sweep's, which are only ever those.
fn updateAabbs(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    for (world.sweep.order.items[begin..end]) |index| {
        const entry = &world.shapes.slots.items[index].value.?;
        const b = world.bodyAt(entry.body_index);
        // Asleep, a body has not moved, and its box is still the one it
        // had: everything that can wake a body without a step - a new
        // shape, `setTransform` - wakes it first, so no box here is stale.
        if (!b.awake) continue;
        const box = entry.def.geometry.aabb(b.transform);
        // Stopped short, it looks a little ahead; see `narrowPhase`.
        world.aabbs.items[index] = if (b.stopped_short) box.grow(ctx.speculative) else box;
    }
}

/// A static body moved by `setTransform` since the last step: its leaves in
/// the level's tree move with it, and it is noted in `moved_level`, whose
/// flags are cleared once the contacts have seen them. On one thread,
/// because a tree is one structure.
///
/// The one walk a step still makes over every body, because `setTransform`
/// is the body's own and cannot tell the world. It reads a few bytes of
/// each: over seven thousand tiles, a few hundredths of a millisecond.
fn refitMovedLevel(self: *World) Error!void {
    for (self.bodies.slots.items, 0..) |*slot, i| {
        const b = &(slot.value orelse continue);
        if (b.type != .static or !b.teleported) continue;
        try self.moved_level.append(self.gpa, @intCast(i));
        var cursor = b.first_shape;
        while (self.shapes.get(cursor)) |entry| : (cursor = entry.next) {
            // What it covered where it was, and what it covers now, are
            // both to be worked out again - and so is what covers it.
            const box = entry.def.geometry.aabb(b.transform);
            self.forgetCoverNear(self.static_tree.boxOf(entry.proxy));
            try self.static_tree.move(self.gpa, entry.proxy, box);
            self.forgetCoverNear(box);
        }
    }
}

// -------------------------------------------------------------------------
// Seams in the level
// -------------------------------------------------------------------------

/// Work out which edges of a polygon of the level another piece of the
/// level covers, into `hidden`. See `collide` for why it matters: a box
/// sliding across a floor of tiles is otherwise stopped at every seam.
///
/// An edge is covered when the strip just outside it - a slop out, short of
/// its corners by a slop - lies wholly inside other solid polygons of the
/// level with the same filter. Tiles that meet exactly cover each other's
/// sides; so do tiles of different sizes, several along one edge; so do
/// tiles a little apart or a little overlapping, which is how a level
/// placed by hand comes out. A sensor covers nothing, and neither does a
/// piece with another filter, which something may pass through while it
/// stops at this one.
///
/// Lazily: when something first comes near, from `findLevelPairs`, and
/// again whenever a piece of the level near it comes, goes or moves - see
/// `forgetCoverNear`. A level of thousands of tiles costs nothing for the
/// ones nobody walks on.
fn coverEdges(self: *World, index: u32) void {
    const entry = &self.shapes.slots.items[index].value.?;
    entry.hidden_known = true;
    entry.hidden = collide.none_hidden;
    if (entry.def.sensor) return;
    const poly = switch (entry.def.geometry) {
        .polygon => |*p| p,
        .circle => return,
    };
    const xf = self.bodyAt(entry.body_index).transform;
    const reach = self.settings.linear_slop * self.settings.units_per_metre;

    var near: Near = .{ .world = self, .self_index = index, .filter = entry.def.filter };
    self.static_tree.query(entry.def.geometry.aabb(xf).grow(2 * reach), &near, Near.visit);
    if (near.count == 0) return;

    for (0..poly.count) |i| {
        const v0 = xf.apply(poly.vertices[i]);
        const v1 = xf.apply(poly.vertices[(i + 1) % poly.count]);
        const out = xf.q.rotate(poly.normals[i]).scale(reach);
        const along = v1.sub(v0);
        const length = along.len();
        // The strip a slop outside the edge, a slop short of each corner so
        // the corners' rounding does not decide it. An edge too short for
        // that is asked about at its middle.
        const in = if (length > 2 * reach) along.scale(reach / length) else along.scale(0.5);
        if (near.covers(v0.add(in).add(out), v1.sub(in).add(out))) {
            entry.hidden |= @as(collide.Hidden, 1) << @intCast(i);
        }
    }
}

/// The solid polygons of the level near one being worked out, and whether
/// they cover a strip beside it.
const Near = struct {
    world: *World,
    self_index: u32,
    filter: Filter,
    /// Sixty-four is eight times what a tile in a grid has around it; past
    /// that the rest are not asked, which can only leave an edge showing.
    shapes: [64]u32 = undefined,
    count: usize = 0,

    fn visit(self: *Near, index: u32) bool {
        if (index == self.self_index) return true;
        const entry = &self.world.shapes.slots.items[index].value.?;
        if (entry.def.sensor or entry.def.geometry != .polygon) return true;
        const f = entry.def.filter;
        if (f.category != self.filter.category or f.mask != self.filter.mask or f.group != self.filter.group) return true;
        self.shapes[self.count] = index;
        self.count += 1;
        return self.count < self.shapes.len;
    }

    /// Whether every point from `a` to `b` is inside one or other of them.
    fn covers(self: *const Near, a: Vec2, b: Vec2) bool {
        // Each polygon holds one stretch of the segment, being convex; the
        // stretches, in order, must leave no gap.
        var stretches: [64][2]f32 = undefined;
        var n: usize = 0;
        for (self.shapes[0..self.count]) |index| {
            const entry = &self.world.shapes.slots.items[index].value.?;
            const xf = self.world.bodyAt(entry.body_index).transform;
            if (stretchInside(&entry.def.geometry.polygon, xf.unapply(a), xf.unapply(b))) |s| {
                stretches[n] = s;
                n += 1;
            }
        }
        std.sort.pdq([2]f32, stretches[0..n], {}, struct {
            fn before(_: void, x: [2]f32, y: [2]f32) bool {
                return x[0] < y[0];
            }
        }.before);
        // A thousandth of the strip is a hair, not a gap.
        const hair = 1e-3;
        var reached: f32 = 0;
        for (stretches[0..n]) |s| {
            if (s[0] > reached + hair) return false;
            reached = @max(reached, s[1]);
        }
        return reached >= 1 - hair;
    }
};

/// The part of the segment from `a` to `b` inside a convex polygon, as
/// fractions of the way along it - clipped against one face after another,
/// Cyrus and Beck's way - or null if none of it is. In the polygon's frame.
fn stretchInside(poly: *const shape_mod.Polygon, a: Vec2, b: Vec2) ?[2]f32 {
    const d = b.sub(a);
    var lo: f32 = 0;
    var hi: f32 = 1;
    for (poly.vertexSlice(), poly.normalSlice()) |v, n| {
        // Inside this face's side where n . (a + t d - v) <= 0.
        const room = n.dot(v.sub(a));
        const rate = n.dot(d);
        if (rate == 0) {
            if (room < 0) return null;
        } else if (rate > 0) {
            hi = @min(hi, room / rate);
        } else {
            lo = @max(lo, room / rate);
        }
        if (lo > hi) return null;
    }
    return .{ lo, hi };
}

/// Every piece of the level near `box` has its covered edges worked out
/// again the next time something comes near it: a piece next to it has
/// come, gone or moved.
fn forgetCoverNear(self: *World, box: Aabb) void {
    const Forget = struct {
        world: *World,
        fn visit(f: *@This(), index: u32) bool {
            f.world.shapes.slots.items[index].value.?.hidden_known = false;
            return true;
        }
    };
    const reach = self.settings.linear_slop * self.settings.units_per_metre;
    var forget: Forget = .{ .world = self };
    self.static_tree.query(box.grow(2 * reach), &forget, Forget.visit);
}

/// Every pair of a moving shape and a piece of the level whose boxes
/// overlap, into `pairs`, after the sweep's. Each dynamic shape that is
/// awake asks the tree about its own box, and so does every kinematic one,
/// moving or not: a kinematic body has nothing to say to a wall, but a
/// sensor on either side has - see `mayTouch`. A sleeper keeps what it had.
///
/// In the sweep's order and then the tree's, both of which depend only on
/// what was added and where, so the pairs - and the step - come out the
/// same on every run.
fn findLevelPairs(self: *World) Error!void {
    if (self.static_tree.root == Tree.null_node) return;
    var visitor: LevelVisitor = .{ .world = self };
    for (self.sweep.order.items) |index| {
        const entry = &self.shapes.slots.items[index].value.?;
        const b = self.bodyAt(entry.body_index);
        const asks = switch (b.type) {
            .dynamic => b.awake,
            .kinematic => true,
            .static => false,
        };
        if (!asks) continue;
        visitor.shape = index;
        self.static_tree.query(self.aabbs.items[index], &visitor, LevelVisitor.visit);
        if (visitor.failed) return error.OutOfMemory;
    }
}

const LevelVisitor = struct {
    world: *World,
    shape: u32 = 0,
    failed: bool = false,

    fn visit(self: *LevelVisitor, level_shape: u32) bool {
        if (!acceptPair(self.world, self.shape, level_shape)) return true;
        // Here, on one thread, before the narrow phase reads it on many.
        if (!self.world.shapes.slots.items[level_shape].value.?.hidden_known) self.world.coverEdges(level_shape);
        self.world.pairs.append(self.world.gpa, .{
            .a = @min(self.shape, level_shape),
            .b = @max(self.shape, level_shape),
        }) catch {
            self.failed = true;
            return false;
        };
        return true;
    }
};

/// Whether a pair of overlapping boxes is worth the narrow phase this step.
fn acceptPair(world: *World, a: u32, b: u32) bool {
    const ea = &world.shapes.slots.items[a].value.?;
    const eb = &world.shapes.slots.items[b].value.?;
    const ba = world.bodyAt(ea.body_index);
    const bb = world.bodyAt(eb.body_index);
    // Two sleepers, or a sleeper and a wall, keep whatever they had; see
    // `prepareContacts`. A sensor's pair is looked at whether anything in it
    // moves or not, unless both bodies sleep: nothing else would keep it.
    if (!isActive(ba) and !isActive(bb) and !watched(ea, eb, ba, bb)) return false;
    return world.mayTouch(ea, eb, ba, bb);
}

/// Whether a pair is a sensor's to watch every step, still or not: a sensor
/// in it, and a body that is not dynamic - a trigger a character stands in,
/// an area set down over a wall or on a body that sleeps. Two dynamic
/// sleepers keep their contact asleep instead.
fn watched(ea: *const ShapeEntry, eb: *const ShapeEntry, ba: *const Body, bb: *const Body) bool {
    if (!ea.def.sensor and !eb.def.sensor) return false;
    return ba.type != .dynamic or bb.type != .dynamic;
}

/// Whether two shapes may touch at all, whatever they are doing.
fn mayTouch(world: *World, ea: *const ShapeEntry, eb: *const ShapeEntry, ba: *const Body, bb: *const Body) bool {
    if (ea.body_index == eb.body_index) return false;
    // Two things that cannot move have nothing to say to each other - unless
    // one of them is a sensor, which is there to say what it is over: a
    // trigger a kinematic character walks into, an area over the level.
    const sensing = ea.def.sensor or eb.def.sensor;
    if (ba.type != .dynamic and bb.type != .dynamic and !sensing) return false;
    // A sensor is seen when either side asks for the other - a hitbox by the
    // hurtbox that watches for it - and a push takes both, or with Godot's
    // rule either.
    const either = sensing or world.settings.filter_rule == .either;
    const filtered = if (either) ea.def.filter.shouldSense(eb.def.filter) else ea.def.filter.shouldCollide(eb.def.filter);
    if (!filtered) return false;
    // Nor do two a joint holds, unless it says they should, or two told
    // never to.
    if (world.no_collide.count() != 0 or world.exceptions.count() != 0) {
        const key = bodyPairKey(ea.body_index, eb.body_index);
        if (world.no_collide.contains(key) or world.exceptions.contains(key)) return false;
    }
    return true;
}

fn narrowPhase(ctx: *const Step, begin: usize, end: usize) void {
    const world = ctx.world;
    for (world.pairs.items[begin..end], world.manifolds.items[begin..end]) |pair, *out| {
        const ea = &world.shapes.slots.items[pair.a].value.?;
        const eb = &world.shapes.slots.items[pair.b].value.?;
        const ba = world.bodyAt(ea.body_index);
        const bb = world.bodyAt(eb.body_index);
        // A body stopped short last step looks ahead; see `Body.stopped_short`.
        const margin = if (ba.stopped_short or bb.stopped_short) ctx.speculative else 0;
        // The level's covered edges, from `coverEdges`; nothing that moves
        // has any.
        const seams: collide.Seams = .{ .a = ea.hidden, .b = eb.hidden, .depth = ctx.seam_depth };
        out.* = manifoldOf(&ea.def.geometry, ba.transform, &eb.def.geometry, bb.transform, margin, seams);
    }
}

/// The right pairing for two geometries, with the normal always from A to
/// B whichever way round the pairing was written. `margin`, how far past
/// touching to look, and `seams`, the edges of the level another piece of
/// it covers: see `collide` for both.
pub fn manifoldOf(a: *const shape_mod.Geometry, xa: Transform, b: *const shape_mod.Geometry, xb: Transform, margin: f32, seams: collide.Seams) Manifold {
    return switch (a.*) {
        .circle => |ca| switch (b.*) {
            .circle => |cb| collide.circles(ca, xa, cb, xb, margin),
            .polygon => |*pb| blk: {
                var m = collide.polygonCircle(pb, xb, ca, xa, margin, seams.swapped());
                m.normal = m.normal.neg();
                break :blk m;
            },
        },
        .polygon => |*pa| switch (b.*) {
            .circle => |cb| collide.polygonCircle(pa, xa, cb, xb, margin, seams),
            .polygon => |*pb| collide.polygons(pa, xa, pb, xb, margin, seams),
        },
    };
}

/// Every touching pair into `contacts`, and every one that pushes into a
/// constraint, warm-started from `previous`. Then what is still asleep.
fn prepareContacts(self: *World, ctx: Step) Error!void {
    const gpa = self.gpa;
    self.constraints.clearRetainingCapacity();
    self.contacts.clearRetainingCapacity();
    self.one_way_off.clearRetainingCapacity();
    self.links.clearRetainingCapacity();

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
        const ba = self.bodyAt(ea.body_index);
        const bb = self.bodyAt(eb.body_index);
        const stored: Stored = .{
            .shape_a = handle_a,
            .shape_b = handle_b,
            .body_a = ea.body,
            .body_b = eb.body,
            .sensor = sensor,
        };

        // A pair with a one-way shape in it is decided when it first
        // touches, and stays as it was decided while it goes on touching.
        if (!sensor and (ea.def.one_way != null or eb.def.one_way != null)) {
            const held = self.previous.contains(key) or
                (!self.one_way_before.contains(key) and holdsOneWay(m, ea, eb, ba, bb));
            if (!held) {
                try self.one_way_off.put(gpa, key, stored);
                continue;
            }
        }

        try self.contacts.put(gpa, key, stored);
        if (sensor) continue;

        try self.links.append(gpa, .{ .a = ea.body_index, .b = eb.body_index, .type_a = ba.type, .type_b = bb.type });
        const warm: ?[2]contact.Impulse = if (self.previous.get(key)) |last| last.impulses else null;
        try self.constraints.append(gpa, contact.prepare(
            key,
            m,
            ba,
            bb,
            ea.body_index,
            eb.body_index,
            self.settings.friction_mix.of(ea.def.material.friction, eb.def.material.friction),
            self.settings.restitution_mix.of(ea.def.material.restitution, eb.def.material.restitution),
            warm,
            ctx.contact_softness,
            ctx.static_softness,
        ));
    }

    // What was touching and is asleep - neither body moving this step -
    // stays touching, impulses and all, without being looked at: that is
    // most of what sleeping saves. A sensor's pair may have been looked at
    // above all the same, and then what was found stands; see `watched`.
    // A contact with no dynamic body in it is a sensor's, looked at every
    // step, so with no dynamic body asleep there is nothing to keep and the
    // walk is skipped.
    if (!self.anyAsleep()) return;
    // A pair being let through sleeps as it was, too: woken, a body resting
    // inside a platform it came up through must not be lifted onto it.
    var off = self.one_way_before.iterator();
    while (off.next()) |entry| {
        if (!self.keepsSleeping(entry.value_ptr)) continue;
        if (self.one_way_off.contains(entry.key_ptr.*)) continue;
        try self.one_way_off.put(gpa, entry.key_ptr.*, entry.value_ptr.*);
    }
    var then = self.previous.iterator();
    while (then.next()) |entry| {
        const stored = entry.value_ptr;
        if (!self.keepsSleeping(stored)) continue;
        if (self.contacts.contains(entry.key_ptr.*)) continue;
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

/// Whether a pair with a one-way shape in it holds, at its first touch:
/// for each one-way side, the contact's normal says the other shape is on
/// the side its arrow comes from. Godot 3.6's rule for bodies, which asks
/// nothing of their speeds or depths. See `shape.OneWay`.
fn holdsOneWay(m: *const Manifold, ea: *const ShapeEntry, eb: *const ShapeEntry, ba: *const Body, bb: *const Body) bool {
    const eps = 1e-5;
    // The manifold's normal is from A to B.
    if (ea.def.one_way) |way| {
        if (m.normal.dot(ba.transform.q.rotate(way.direction)) > -eps) return false;
    }
    if (eb.def.one_way) |way| {
        if (m.normal.neg().dot(bb.transform.q.rotate(way.direction)) > -eps) return false;
    }
    return true;
}

/// Whether any dynamic body is asleep. A walk over the bodies that move
/// rather than a count kept up to date, because `Body.sleep` and
/// `Body.wake` are the body's own and cannot tell the world.
fn anyAsleep(self: *World) bool {
    for (self.movers.items) |index| {
        if (!self.bodyAt(index).awake) return true;
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

/// One colour's joints, the same, with the step they are solved at -
/// their spring, or rigid while relaxing.
const JointRun = struct {
    step: *const Step,
    joint_step: joint_mod.Step,
    refs: []joint_mod.Ref,
};

fn warmStartAll(self: *World, jobs: *Jobs, ctx: *const Step) void {
    for (0..solver.max_colours) |colour| {
        const run: JointRun = .{ .step = ctx, .joint_step = ctx.joint, .refs = self.joint_colouring.run(colour) };
        solver.forRange(jobs, run.refs.len, grain_joints, &run, warmStartJointRun);
    }
    self.warmStartJointsHere(self.joint_colouring.run(solver.overflow));
    for (0..solver.max_colours) |colour| {
        const run: Run = .{ .step = ctx, .constraints = self.colouring.run(colour) };
        solver.forRange(jobs, run.constraints.len, grain_contacts, &run, warmStartRun);
    }
    self.warmStartRunHere(self.colouring.run(solver.overflow));
}

/// One pass over everything: the joints colour by colour, then the
/// contacts. `pass` says whether it is the pushing pass or the relaxing one.
fn solveAll(self: *World, jobs: *Jobs, ctx: *const Step, comptime pass: contact.Pass) void {
    const joint_step = if (pass == .solve) ctx.joint else ctx.joint_relax;
    for (0..solver.max_colours) |colour| {
        const run: JointRun = .{ .step = ctx, .joint_step = joint_step, .refs = self.joint_colouring.run(colour) };
        solver.forRange(jobs, run.refs.len, grain_joints, &run, solveJointRun);
    }
    self.solveJointsHere(self.joint_colouring.run(solver.overflow), joint_step);
    for (0..solver.max_colours) |colour| {
        const run: Run = .{ .step = ctx, .constraints = self.colouring.run(colour) };
        solver.forRange(jobs, run.constraints.len, grain_contacts, &run, solveRun(pass));
    }
    self.solveRunHere(self.colouring.run(solver.overflow), ctx.contact, pass);
}

/// Restitution, contact by contact. Joints do not bounce.
fn restituteAll(self: *World, jobs: *Jobs, ctx: *const Step) void {
    for (0..solver.max_colours) |colour| {
        const run: Run = .{ .step = ctx, .constraints = self.colouring.run(colour) };
        solver.forRange(jobs, run.constraints.len, grain_contacts, &run, restituteRun);
    }
    self.restituteRunHere(self.colouring.run(solver.overflow));
}

fn warmStartRun(run: *const Run, begin: usize, end: usize) void {
    run.step.world.warmStartRunHere(run.constraints[begin..end]);
}

/// The job that solves part of a run, for one kind of pass.
///
/// A job is a plain function, and `forRange` wants it at compile time, but
/// which pass it makes is a parameter. So this is a function that *makes*
/// the function: the struct inside is a fresh type for each `pass`, its
/// `go` sees `pass` as a constant, and `solveRun(.relax)` is as much a
/// compile-time value as a function written out by hand would be.
fn solveRun(comptime pass: contact.Pass) fn (*const Run, usize, usize) void {
    return struct {
        fn go(run: *const Run, begin: usize, end: usize) void {
            run.step.world.solveRunHere(run.constraints[begin..end], run.step.contact, pass);
        }
    }.go;
}

fn restituteRun(run: *const Run, begin: usize, end: usize) void {
    run.step.world.restituteRunHere(run.constraints[begin..end]);
}

fn warmStartRunHere(self: *World, constraints: []contact.Constraint) void {
    for (constraints) |*c| contact.warmStart(c, self.bodyAt(c.body_a), self.bodyAt(c.body_b));
}

fn solveRunHere(self: *World, constraints: []contact.Constraint, contact_step: contact.Step, comptime pass: contact.Pass) void {
    for (constraints) |*c| contact.solve(c, self.bodyAt(c.body_a), self.bodyAt(c.body_b), contact_step, pass);
}

fn restituteRunHere(self: *World, constraints: []contact.Constraint) void {
    const threshold = self.settings.restitution_threshold * self.settings.units_per_metre;
    for (constraints) |*c| contact.restitute(c, self.bodyAt(c.body_a), self.bodyAt(c.body_b), threshold);
}

fn warmStartJointRun(run: *const JointRun, begin: usize, end: usize) void {
    run.step.world.warmStartJointsHere(run.refs[begin..end]);
}

fn solveJointRun(run: *const JointRun, begin: usize, end: usize) void {
    run.step.world.solveJointsHere(run.refs[begin..end], run.joint_step);
}

fn warmStartJointsHere(self: *World, refs: []const joint_mod.Ref) void {
    for (refs) |ref| joint_mod.warmStart(self.jointAt(ref.joint), self.bodyAt(ref.body_a), self.bodyAt(ref.body_b));
}

fn solveJointsHere(self: *World, refs: []const joint_mod.Ref, joint_step: joint_mod.Step) void {
    for (refs) |ref| joint_mod.solve(self.jointAt(ref.joint), self.bodyAt(ref.body_a), self.bodyAt(ref.body_b), joint_step);
}

/// One substep's worth of movement.
fn integratePositions(ctx: *const Step, begin: usize, end: usize) void {
    const h = ctx.h;
    for (ctx.world.movers.items[begin..end]) |index| {
        const b = ctx.world.bodyAt(index);
        // Whatever `setTransform` changed has been looked at by now.
        b.teleported = false;
        if (!b.awake) continue;
        b.center = b.center.mulAdd(b.linear_velocity, h);
        b.angle += h * b.angular_velocity;
        b.syncTransform();
    }
}

/// The job that sweeps the fast bodies of one kind: bullets or the rest.
/// A function made per kind, as `solveRun` is.
fn sweepRun(comptime bullets: bool) fn (*const Step, usize, usize) void {
    return struct {
        fn go(ctx: *const Step, begin: usize, end: usize) void {
            const world = ctx.world;
            for (world.movers.items[begin..end]) |index| {
                const b = world.bodyAt(index);
                if (b.type != .dynamic or b.bullet != bullets) continue;
                b.stopped_short = false;
                if (!b.awake or !world.settings.enable_continuous) continue;
                world.sweepBody(index, b, ctx);
            }
        }
    }.go;
}

/// Sweep one body along the path it took this step, and if it went into
/// something in a way the step could not have seen, put it back where it
/// first touched. See `continuous`.
///
/// Each body reads only the level, or for a bullet the bodies that are not
/// bullets - all of them where they have finished moving - and writes only
/// itself, so every body can be swept at once.
fn sweepBody(self: *World, index: u32, b: *Body, ctx: *const Step) void {
    // Moved less than half its thinnest, it cannot have gone through
    // anything unseen: the worst it can be is sunk in, and pushed out.
    const moved = b.center.dist(b.center0) + @abs(b.angle - b.angle0) * b.extent;
    if (moved <= 0.5 * b.min_extent) return;

    var sweeper: Sweeper = .{
        .world = self,
        .ctx = ctx,
        .body = b,
        .path = .{ .local_center = b.local_center, .c0 = b.center0, .a0 = b.angle0, .c1 = b.center, .a1 = b.angle },
    };
    var cursor = b.first_shape;
    while (self.shapes.get(cursor)) |entry| : (cursor = entry.next) {
        if (entry.def.sensor) continue;
        sweeper.entry = entry;
        sweeper.entry_handle = cursor;
        // Every pose along the way fits in this: the centre's path, widened
        // by as far as the shape reaches from the centre.
        const reach = entry.def.geometry.reach(b.local_center);
        const box: Aabb = .{
            .min = b.center0.min(b.center).sub(.splat(reach)),
            .max = b.center0.max(b.center).add(.splat(reach)),
        };
        self.static_tree.query(box, &sweeper, Sweeper.visitLevel);
        if (!b.bullet) continue;
        for (self.movers.items) |other_index| {
            if (other_index == index) continue;
            const other = self.bodyAt(other_index);
            if (other.type == .dynamic and other.bullet) continue;
            if (!boxNears(other, box)) continue;
            var other_cursor = other.first_shape;
            while (self.shapes.get(other_cursor)) |other_entry| : (other_cursor = other_entry.next) {
                sweeper.consider(other_entry, other_cursor);
            }
        }
    }

    if (sweeper.fraction < 1) {
        const t = sweeper.fraction;
        b.center = b.center0.lerp(b.center, t);
        b.angle = b.angle0 + t * (b.angle - b.angle0);
        b.syncTransform();
        b.stopped_short = true;
        // Stopped at its first touch, it keeps its velocity, so the next
        // step bounces it as it would anything it hit. Stopped after it was
        // already pressed in - by a pile behind it, or by its own spin
        // turning a corner in where the step's contacts were not looking -
        // the speed that way is taken off: going on, it would go through.
        if (sweeper.into) |into| {
            const in_speed = b.linear_velocity.dot(into);
            if (in_speed > 0) b.linear_velocity = b.linear_velocity.sub(into.scale(in_speed));
        }
    }
}

/// One shape of a fast body, asked about everything near its path.
const Sweeper = struct {
    world: *World,
    ctx: *const Step,
    body: *const Body,
    path: continuous.Sweep,
    entry: *const ShapeEntry = undefined,
    entry_handle: ShapeId = undefined,
    /// How far along the earliest first touch that counts is, so far.
    fraction: f32 = 1,
    /// When that touch is one the shape was already pressed into, the way
    /// into what it touched. See `sweepBody`.
    into: ?Vec2 = null,

    fn visitLevel(self: *Sweeper, level_shape: u32) bool {
        const slot = &self.world.shapes.slots.items[level_shape];
        self.consider(&slot.value.?, .{ .index = level_shape, .generation = slot.generation });
        return true;
    }

    fn consider(self: *Sweeper, other: *const ShapeEntry, other_handle: ShapeId) void {
        if (other.def.sensor) return;
        const world = self.world;
        const ctx = self.ctx;
        const other_body = world.bodyAt(other.body_index);
        if (!world.mayTouch(self.entry, other, self.body, other_body)) return;
        const one_way = self.entry.def.one_way != null or other.def.one_way != null;
        if (one_way and self.letThrough(other_handle)) return;
        const g = &self.entry.def.geometry;
        const og = &other.def.geometry;
        const xo = other_body.transform;
        switch (continuous.timeOfImpact(g, self.path, og, xo, self.fraction, ctx.toi_target, ctx.toi_tolerance)) {
            .miss => {},
            // Touching it already: only the core is swept, which is not, and
            // which nothing the shape is sliding along can reach. A shape
            // pressed in so deep that its core touches too - a thin rod
            // shoved into a wall by a pile behind it - is left its centre,
            // which at least may not cross into what it is pressed against.
            .touching => {
                const core = continuous.core(g);
                switch (continuous.timeOfImpact(&core, self.path, og, xo, self.fraction, ctx.toi_target, ctx.toi_tolerance)) {
                    .hit => |t| self.pressedIn(&core, og, xo, t),
                    .miss => {},
                    .touching => {
                        const centre: shape_mod.Geometry = .{ .circle = .{ .center = g.centroid(), .radius = 0 } };
                        switch (continuous.timeOfImpact(&centre, self.path, og, xo, self.fraction, 0, ctx.toi_tolerance)) {
                            .hit => |t| self.pressedIn(&centre, og, xo, t),
                            .miss, .touching => {},
                        }
                    },
                }
            },
            .hit => |t| if (self.counts(g, og, xo, t) and (!one_way or self.holdsAt(t, other, other_body))) {
                self.fraction = t;
                self.into = null;
            },
        }
    }

    /// Whether the step has already let this pair through a one-way shape:
    /// then it is not swept at all, or a body let in at a platform's side and
    /// going on through would be stopped by its own centre.
    fn letThrough(self: *const Sweeper, other_handle: ShapeId) bool {
        const mine = self.entry_handle;
        const key: contact.PairKey = if (mine.index < other_handle.index)
            .{ .a = mine.toInt(), .b = other_handle.toInt() }
        else
            .{ .a = other_handle.toInt(), .b = mine.toInt() };
        return self.world.one_way_off.contains(key);
    }

    /// Whether a first touch at `t` with a one-way shape in the pair is one
    /// it holds: the question `holdsOneWay` asks at the step, of which side
    /// the touch is on. A fast body going up through a floor, or in at its
    /// side, is not put back.
    fn holdsAt(self: *const Sweeper, t: f32, other: *const ShapeEntry, other_body: *const Body) bool {
        const eps = 1e-5;
        const here = self.path.at(t);
        // From the fast body's shape towards the other.
        const normal = continuous.separation(&self.entry.def.geometry, here, &other.def.geometry, other_body.transform).normal;
        if (other.def.one_way) |way| {
            if (normal.neg().dot(other_body.transform.q.rotate(way.direction)) > -eps) return false;
        }
        if (self.entry.def.one_way) |way| {
            if (normal.dot(here.q.rotate(way.direction)) > -eps) return false;
        }
        return true;
    }

    /// The earliest touch is a shape that was already pressed into what it
    /// touched, found by sweeping `inner` - its core or its centre. Which
    /// way is in is kept, for the speed that way to be taken off.
    fn pressedIn(self: *Sweeper, inner: *const shape_mod.Geometry, og: *const shape_mod.Geometry, xo: Transform, t: f32) void {
        self.fraction = t;
        self.into = continuous.separation(inner, self.path.at(t), og, xo).normal;
    }

    /// Whether a first touch at `t` is one the step would have got wrong:
    /// the shape's core reaches what it touched, or the rest of the path
    /// sinks it in deeper than a graze. Brushing onto the next tile of a
    /// floor does neither, and is left to the next step.
    fn counts(self: *const Sweeper, g: *const shape_mod.Geometry, og: *const shape_mod.Geometry, xo: Transform, t: f32) bool {
        const ctx = self.ctx;
        const core = continuous.core(g);
        if (continuous.timeOfImpact(&core, self.path, og, xo, 1, ctx.toi_target, ctx.toi_tolerance) != .miss) return true;
        return continuous.sinksPast(g, self.path, og, xo, t, ctx.graze);
    }
};

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
    std.mem.swap(ContactMap, &self.one_way_off, &self.one_way_before);
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
    for (self.movers.items) |index| {
        const b = self.bodyAt(index);
        if (!b.awake) continue;
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
    for (self.movers.items) |index| {
        const b = self.bodyAt(index);
        if (b.type != .dynamic) continue;
        const root = find(parent, index);
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

    for (self.movers.items) |index| {
        const b = self.bodyAt(index);
        if (b.type != .dynamic) continue;
        if (rest[find(parent, index)] >= self.settings.time_to_sleep) {
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
/// The shapes that move are asked one by one - there are few beside a
/// level - and the ray is shortened to the nearest of them; then the
/// level's tree is walked with what is left, and only the branches that
/// ray crosses are looked at. Logarithmic in the level.
pub fn castRay(self: *World, origin: Vec2, translation: Vec2, filter: Filter) ?RayHit {
    var cast: RayCast = .{ .world = self, .origin = origin, .translation = translation, .filter = filter };
    for (self.sweep.order.items) |index| {
        const b = self.bodyAt(self.shapes.slots.items[index].value.?.body_index);
        if (!rayNears(b, origin, translation, cast.fraction)) continue;
        _ = cast.consider(index, cast.fraction);
    }
    self.static_tree.rayCast(origin, translation, cast.fraction, &cast, RayCast.consider);
    return cast.best;
}

// Whether a query can reach anything on a body at all: the circle about
// its centre of mass that its shapes all fit in, asked before the shapes
// are. A few multiplies that turn most of the moving shapes away unasked -
// which is what a query spends its time on once the level is in a tree.

fn rayNears(b: *const Body, origin: Vec2, translation: Vec2, max_fraction: f32) bool {
    const m = origin.sub(b.center);
    const dd = translation.lenSq();
    // The point of the ray nearest the centre, kept on the ray.
    const t = if (dd > 0) std.math.clamp(-m.dot(translation) / dd, 0, max_fraction) else 0;
    return m.mulAdd(translation, t).lenSq() <= b.extent * b.extent;
}

fn pointNears(b: *const Body, point: Vec2) bool {
    return point.distSq(b.center) <= b.extent * b.extent;
}

fn boxNears(b: *const Body, box: Aabb) bool {
    // The point of the box nearest the centre.
    const nearest = b.center.max(box.min).min(box.max);
    return nearest.distSq(b.center) <= b.extent * b.extent;
}

const RayCast = struct {
    world: *World,
    origin: Vec2,
    translation: Vec2,
    filter: Filter,
    /// How far along the nearest hit so far is: where the ray now stops.
    fraction: f32 = 1,
    best: ?RayHit = null,

    /// Ask one shape. Answers where the ray hits it if that is nearer than
    /// `max_fraction`, or a negative number - which is what the tree wants
    /// to hear from a visitor.
    fn consider(self: *RayCast, index: u32, max_fraction: f32) f32 {
        const world = self.world;
        const entry = &world.shapes.slots.items[index].value.?;
        if (!self.filter.shouldCollide(entry.def.filter)) return -1;
        const xf = world.bodyAt(entry.body_index).transform;
        const hit = rayAgainst(&entry.def.geometry, xf, self.origin, self.translation, max_fraction) orelse return -1;
        self.fraction = hit.fraction;
        self.best = .{
            .shape = world.handleOf(index),
            .body = entry.body,
            .point = self.origin.mulAdd(self.translation, hit.fraction),
            .normal = hit.normal,
            .fraction = hit.fraction,
        };
        return hit.fraction;
    }
};

/// The handle of the shape in a live slot.
fn handleOf(self: *World, index: u32) ShapeId {
    return .{ .index = index, .generation = self.shapes.slots.items[index].generation };
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

/// The first shape under a point, or null: a moving one if any is, else
/// the level's. Which of several is first is not promised, so a game that
/// wants the topmost should ask `overlapAabb` and choose.
pub fn overlapPoint(self: *World, point: Vec2) ?ShapeId {
    for (self.sweep.order.items) |index| {
        const b = self.bodyAt(self.shapes.slots.items[index].value.?.body_index);
        if (!pointNears(b, point)) continue;
        if (self.containsPoint(index, point)) return self.handleOf(index);
    }
    const Probe = struct {
        world: *World,
        point: Vec2,
        found: ?ShapeId = null,

        fn visit(p: *@This(), index: u32) bool {
            if (!p.world.containsPoint(index, p.point)) return true;
            p.found = p.world.handleOf(index);
            return false;
        }
    };
    var probe: Probe = .{ .world = self, .point = point };
    self.static_tree.query(.{ .min = point, .max = point }, &probe, Probe.visit);
    return probe.found;
}

fn containsPoint(self: *World, index: u32, point: Vec2) bool {
    const entry = &self.shapes.slots.items[index].value.?;
    const local = self.bodyAt(entry.body_index).transform.unapply(point);
    return switch (entry.def.geometry) {
        .circle => |c| local.distSq(c.center) <= c.radius * c.radius,
        .polygon => |*p| p.containsLocal(local),
    };
}

/// Call `visit(context, shape)` for every shape whose box overlaps `box`,
/// until it returns false: the moving shapes, then the level's.
pub fn overlapAabb(
    self: *World,
    box: Aabb,
    context: anytype,
    comptime visit: fn (@TypeOf(context), ShapeId) bool,
) void {
    for (self.sweep.order.items) |index| {
        const entry = &self.shapes.slots.items[index].value.?;
        const b = self.bodyAt(entry.body_index);
        if (!boxNears(b, box)) continue;
        if (!entry.def.geometry.aabb(b.transform).overlaps(box)) continue;
        if (!visit(context, self.handleOf(index))) return;
    }
    // A struct declared inside a function may use the function's
    // `comptime` parameters - here `visit` - which is how the tree's
    // visitor, which knows nothing of handles, hands each leaf on to the
    // caller's as one.
    const Probe = struct {
        world: *World,
        context: @TypeOf(context),

        fn each(p: *@This(), index: u32) bool {
            return visit(p.context, p.world.handleOf(index));
        }
    };
    var probe: Probe = .{ .world = self, .context = context };
    self.static_tree.query(box, &probe, Probe.each);
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "the level knows which edges of its pieces another piece covers" {
    var world: World = .init(testing.allocator, .{});
    defer world.deinit();

    // The edges of a box: 0 its top (y is down), 1 its right, 2 its
    // bottom, 3 its left.
    const Look = struct {
        fn hidden(w: *World, s: ShapeId) collide.Hidden {
            w.coverEdges(s.index);
            return w.shape(s).?.hidden;
        }
        fn tile(w: *World, at: Vec2, def: Shape) !ShapeId {
            return w.addShape(try w.createBody(.{ .type = .static, .position = at }), def);
        }
    };
    const box: Shape = .box(0.25, 0.25);

    // A block of three by three half-metre tiles: the middle one covered
    // all round, a corner one on its two inner sides, an edge one on three.
    var grid: [3][3]ShapeId = undefined;
    for (0..3) |c| {
        for (0..3) |r| grid[c][r] = try Look.tile(&world, .init(@as(f32, @floatFromInt(c)) * 0.5, @as(f32, @floatFromInt(r)) * 0.5), box);
    }
    try testing.expectEqual(@as(collide.Hidden, 0b1111), Look.hidden(&world, grid[1][1]));
    try testing.expectEqual(@as(collide.Hidden, 0b0110), Look.hidden(&world, grid[0][0]));
    try testing.expectEqual(@as(collide.Hidden, 0b1110), Look.hidden(&world, grid[1][0]));

    // A tile a metre wide on two half a metre wide: each covers part of its
    // bottom, and together all of it.
    const wide = try Look.tile(&world, .init(5.5, 0), .box(0.5, 0.25));
    const under_left = try Look.tile(&world, .init(5.25, 0.5), box);
    _ = try Look.tile(&world, .init(5.75, 0.5), box);
    try testing.expectEqual(@as(collide.Hidden, 0b0100), Look.hidden(&world, wide));
    try testing.expectEqual(@as(collide.Hidden, 0b0011), Look.hidden(&world, under_left));

    // Three millimetres apart, as a level placed by hand comes out, is
    // joined; a centimetre apart is a gap, with a face on each side of it.
    const close = try Look.tile(&world, .init(10, 0), box);
    _ = try Look.tile(&world, .init(10.503, 0), box);
    try testing.expectEqual(@as(collide.Hidden, 0b0010), Look.hidden(&world, close));
    const apart = try Look.tile(&world, .init(20, 0), box);
    _ = try Look.tile(&world, .init(20.51, 0), box);
    try testing.expectEqual(@as(collide.Hidden, 0), Look.hidden(&world, apart));

    // Something may pass through a piece with another filter, or a sensor,
    // and stop at this one: neither covers it.
    const beside_other = try Look.tile(&world, .init(30, 0), box);
    _ = try Look.tile(&world, .init(30.5, 0), .{ .geometry = box.geometry, .filter = .{ .category = 2 } });
    try testing.expectEqual(@as(collide.Hidden, 0), Look.hidden(&world, beside_other));
    const beside_sensor = try Look.tile(&world, .init(40, 0), box);
    _ = try Look.tile(&world, .init(40.5, 0), .{ .geometry = box.geometry, .sensor = true });
    try testing.expectEqual(@as(collide.Hidden, 0), Look.hidden(&world, beside_sensor));

    // Taking a tile away shows what it covered: the middle tile's right side
    // is to be worked out again, and is open.
    world.destroyBody(world.shape(grid[2][1]).?.body);
    try testing.expect(!world.shape(grid[1][1]).?.hidden_known);
    try testing.expectEqual(@as(collide.Hidden, 0b1101), Look.hidden(&world, grid[1][1]));

    // Moving one by hand shows what it covered where it was.
    world.bodyAt(world.shape(grid[0][1]).?.body_index).setTransform(.init(-3, 0), 0);
    try world.refitMovedLevel();
    world.moved_level.clearRetainingCapacity();
    try testing.expect(!world.shape(grid[1][1]).?.hidden_known);
    try testing.expectEqual(@as(collide.Hidden, 0b0101), Look.hidden(&world, grid[1][1]));
}

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
