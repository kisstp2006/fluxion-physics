// SPDX-License-Identifier: BSD-2-Clause

//! Fluxion Physics - rigid bodies in a plane, on every core or on none.
//!
//!   `World`       the bodies, the shapes and joints, and one step of time
//!   `Body`        a rigid body: where it is, how it moves, what it weighs
//!   `shape`       circles and convex polygons, materials and filters
//!   `joint`       hinges, sliders, rods, ropes, springs, welds, wheels
//!   `collide`     where two shapes touch, and how deep
//!   `broadphase`  which pairs are close enough to ask
//!   `contact`     a contact as the solver sees it
//!   `solver`      how a step is spread across the cores
//!   `geometry`    rotations, transforms, and 2D boxes
//!
//! ```zig
//! const physics = @import("fluxion_physics");
//!
//! var world: physics.World = .init(gpa, .{ .units_per_metre = 100 });
//! defer world.deinit();
//!
//! const ground = try world.createBody(.{ .type = .static, .position = .init(0, 500) });
//! _ = try world.addShape(ground, .box(1000, 10));
//! const ball = try world.createBody(.{ .position = .init(0, 0) });
//! _ = try world.addShape(ball, .circle(20));
//!
//! // Hung from a pin two hundred pixels up, on a rod.
//! const pin = try world.createBody(.{ .type = .static, .position = .init(0, -200) });
//! _ = try world.createJoint(.{ .distance = .{ .body_a = pin, .body_b = ball, .anchor_a = .init(0, -200), .anchor_b = .zero } });
//!
//! try world.step(1.0 / 60.0, &jobs);
//! ```
//!
//! **Two-dimensional, and only that, for now.** The types are 2D through
//! and through - a rotation is a cosine and a sine, a polygon is a fan of
//! points - rather than 3D types with a zero in them, because a 2D solver
//! that pretends to be 3D pays for the pretence in every multiply.
//!
//! **The step runs on [Fluxion Jobs](https://github.com/kisstp2006/fluxion-jobs)**,
//! and the contact solver is coloured so contacts that share no body are
//! solved at once. With no workers - the browser - the same code runs on
//! one thread, and produces the same numbers to the bit. See `solver`.
//!
//! **It knows nothing about entities.** A body is named by a handle and a
//! game keeps the handle wherever it keeps things - in a component, in an
//! engine's own table. That is what lets it be built and tested on its
//! own, and used by something that is not the engine.

const std = @import("std");

pub const World = @import("World.zig");
pub const Body = @import("body.zig");
pub const shape = @import("shape.zig");
pub const joint = @import("joint.zig");
pub const collide = @import("collide.zig");
pub const broadphase = @import("broadphase.zig");
pub const contact = @import("contact.zig");
pub const solver = @import("solver.zig");
pub const geometry = @import("geometry.zig");

/// What names a body. See `World.createBody`.
pub const BodyId = Body.Id;
/// What names a shape on a body. See `World.addShape`.
pub const ShapeId = Body.ShapeId;
/// Static, kinematic or dynamic. See `Body`.
pub const BodyType = Body.Type;
/// What `World.createBody` takes.
pub const BodyDef = Body.Def;

/// What `World.addShape` takes. See `shape`.
pub const Shape = shape.Shape;
pub const Circle = shape.Circle;
pub const Polygon = shape.Polygon;
pub const Material = shape.Material;
pub const Filter = shape.Filter;

/// What names a joint. See `World.createJoint`.
pub const JointId = joint.Id;
/// What `World.createJoint` takes. See `joint`.
pub const JointDef = joint.Def;
pub const Joint = joint.Joint;
pub const Spring = joint.Spring;
pub const Limit = joint.Limit;
pub const AngularMotor = joint.AngularMotor;
pub const LinearMotor = joint.LinearMotor;

pub const Settings = World.Settings;
pub const ContactEvent = World.ContactEvent;
pub const RayHit = World.RayHit;
pub const Manifold = collide.Manifold;

pub const Vec2 = geometry.Vec2;
pub const Rot = geometry.Rot;
pub const Transform = geometry.Transform;
pub const Aabb = geometry.Aabb;

/// The scheduler a step runs on. Re-exported so a program that has no
/// other use for it need not name the package.
pub const Jobs = @import("fluxion_jobs").Jobs;

/// The arithmetic underneath, for a caller that wants more of it than
/// `Vec2`.
pub const math = @import("fluxion_math");

test {
    _ = World;
    _ = Body;
    _ = shape;
    _ = joint;
    _ = collide;
    _ = broadphase;
    _ = contact;
    _ = solver;
    _ = geometry;
    _ = @import("physics_test.zig");
    _ = @import("joint_test.zig");
    _ = @import("sleep_test.zig");
}
