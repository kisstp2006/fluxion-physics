// SPDX-License-Identifier: BSD-2-Clause

//! The arithmetic a rigid body is placed with: a rotation, a transform made
//! of one and a translation, and a box around something.
//!
//! **A rotation is a cosine and a sine, not an angle.** Turning a point by an
//! angle costs a `cos` and a `sin` every time; turning it by a rotation that
//! already holds both costs four multiplies. A polygon with eight vertices is
//! placed in the world eight times a step, and a body's angle changes once -
//! so the angle is integrated, the pair is rebuilt from it once, and every
//! vertex uses the pair. `Rot` is that pair.
//!
//! **The axes are the engine's.** `+x` is right, `+y` is *down*, and a
//! positive angle turns `+x` towards `+y`, which is clockwise on a screen.
//! That is [Fluxion Engine](https://github.com/kisstp2006/fluxion-engine)'s
//! `Transform2D` and the interface layer's layout, so a body's position goes
//! into a sprite's transform with no flip anywhere. It also means gravity
//! pulls towards `+y` and a polygon with positive signed area appears to wind
//! clockwise - the arithmetic is the mathematician's, only the picture is
//! upside down, and nothing here cares which way up the picture is.
//!
//! **`Aabb` here is two-dimensional.** `fluxion-math` has one too, and it is
//! the 3D one a frustum is tested against; this package's is the box a shape
//! sweeps through the broad phase.

const std = @import("std");
const testing = std.testing;
const math = @import("fluxion_math");

/// A point, a displacement, a velocity. Fluxion Math's, re-exported so a
/// caller need not name that package for the one type everything here takes.
pub const Vec2 = math.Vec2;

/// A turn, kept as its cosine and sine.
pub const Rot = extern struct {
    c: f32 = 1,
    s: f32 = 0,

    pub const identity: Rot = .{};

    pub fn fromAngle(radians: f32) Rot {
        return .{ .c = @cos(radians), .s = @sin(radians) };
    }

    /// The angle back, in `[-pi, pi]`. For a log line or a sprite's
    /// `rotation`; nothing in a step calls this.
    pub fn angle(self: Rot) f32 {
        return std.math.atan2(self.s, self.c);
    }

    /// `v` turned by this rotation.
    pub inline fn rotate(self: Rot, v: Vec2) Vec2 {
        return .{ .x = self.c * v.x - self.s * v.y, .y = self.s * v.x + self.c * v.y };
    }

    /// `v` turned back. The transpose, because a rotation's inverse is one.
    pub inline fn invRotate(self: Rot, v: Vec2) Vec2 {
        return .{ .x = self.c * v.x + self.s * v.y, .y = -self.s * v.x + self.c * v.y };
    }

    /// `q` after `r`: the turn that is `r` and then `q`.
    pub fn mul(q: Rot, r: Rot) Rot {
        return .{ .c = q.c * r.c - q.s * r.s, .s = q.s * r.c + q.c * r.s };
    }

    /// `q` undone and then `r`: what turns something in `q`'s frame into
    /// `r`'s.
    pub fn invMul(q: Rot, r: Rot) Rot {
        return .{ .c = q.c * r.c + q.s * r.s, .s = q.c * r.s - q.s * r.c };
    }
};

/// Where a body's origin is and which way it faces.
pub const Transform = extern struct {
    p: Vec2 = .zero,
    q: Rot = .identity,

    pub const identity: Transform = .{};

    pub fn init(p: Vec2, radians: f32) Transform {
        return .{ .p = p, .q = .fromAngle(radians) };
    }

    /// A local point in the world.
    pub inline fn apply(self: Transform, v: Vec2) Vec2 {
        return self.q.rotate(v).add(self.p);
    }

    /// A world point in this frame.
    pub inline fn unapply(self: Transform, v: Vec2) Vec2 {
        return self.q.invRotate(v.sub(self.p));
    }

    /// `a` undone and then `b`: the transform that takes `b`'s frame into
    /// `a`'s. What two polygons are compared in - one is brought into the
    /// other's frame once, rather than both into the world's every vertex.
    pub fn invMul(a: Transform, b: Transform) Transform {
        return .{
            .q = a.q.invMul(b.q),
            .p = a.q.invRotate(b.p.sub(a.p)),
        };
    }
};

/// An axis-aligned box: the least a shape can be enclosed in without
/// looking at it, and what the broad phase compares.
pub const Aabb = extern struct {
    min: Vec2,
    max: Vec2,

    pub fn fromCenter(middle: Vec2, half: Vec2) Aabb {
        return .{ .min = middle.sub(half), .max = middle.add(half) };
    }

    pub fn fromPoints(points: []const Vec2) Aabb {
        std.debug.assert(points.len != 0);
        var box: Aabb = .{ .min = points[0], .max = points[0] };
        for (points[1..]) |p| {
            box.min = box.min.min(p);
            box.max = box.max.max(p);
        }
        return box;
    }

    /// The smallest box holding both.
    pub fn join(a: Aabb, b: Aabb) Aabb {
        return .{ .min = a.min.min(b.min), .max = a.max.max(b.max) };
    }

    /// Wider by `margin` on every side.
    pub fn grow(self: Aabb, margin: f32) Aabb {
        return .{ .min = self.min.sub(.splat(margin)), .max = self.max.add(.splat(margin)) };
    }

    pub fn center(self: Aabb) Vec2 {
        return self.min.add(self.max).scale(0.5);
    }

    /// Half the width and half the height.
    pub fn extent(self: Aabb) Vec2 {
        return self.max.sub(self.min).scale(0.5);
    }

    /// Touching counts. Two boxes sharing an edge overlap, because the
    /// shapes inside them might be resting on each other and the narrow
    /// phase is the one that should say so.
    pub inline fn overlaps(a: Aabb, b: Aabb) bool {
        return a.min.x <= b.max.x and b.min.x <= a.max.x and
            a.min.y <= b.max.y and b.min.y <= a.max.y;
    }

    pub inline fn contains(self: Aabb, p: Vec2) bool {
        return self.min.x <= p.x and p.x <= self.max.x and
            self.min.y <= p.y and p.y <= self.max.y;
    }
};

/// The cross product two vectors in a plane have: a number, the z of the 3D
/// one. Positive when `b` is anticlockwise of `a` in the mathematician's
/// axes - which on this screen looks clockwise, and does not matter.
pub inline fn cross(a: Vec2, b: Vec2) f32 {
    return a.x * b.y - a.y * b.x;
}

/// A vector crossed with a scalar out of the plane: `v` turned a quarter
/// turn clockwise in the mathematician's axes. `crossVS(v, 1)` in Box2D.
pub inline fn crossVS(v: Vec2, s: f32) Vec2 {
    return .{ .x = s * v.y, .y = -s * v.x };
}

/// A scalar crossed with a vector: the other quarter turn. What an angular
/// velocity does to a point at `v` from the centre: `crossSV(w, r)` is the
/// velocity the rotation gives that point.
pub inline fn crossSV(s: f32, v: Vec2) Vec2 {
    return .{ .x = -s * v.y, .y = s * v.x };
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "a rotation turns +x towards +y and undoes itself" {
    const quarter: Rot = .fromAngle(std.math.pi / 2.0);
    const turned = quarter.rotate(.unit_x);
    try testing.expect(turned.approxEql(.unit_y));
    try testing.expect(quarter.invRotate(turned).approxEql(.unit_x));
    try testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0), quarter.angle(), 1e-6);

    // Composing two eighths is a quarter, and undoing one of them is the
    // other.
    const eighth: Rot = .fromAngle(std.math.pi / 4.0);
    const both = eighth.mul(eighth);
    try testing.expectApproxEqAbs(quarter.c, both.c, 1e-6);
    try testing.expectApproxEqAbs(quarter.s, both.s, 1e-6);
    const rest = eighth.invMul(quarter);
    try testing.expectApproxEqAbs(eighth.c, rest.c, 1e-6);
    try testing.expectApproxEqAbs(eighth.s, rest.s, 1e-6);
}

test "a transform takes a local point out and back" {
    const xf: Transform = .init(.init(10, 20), std.math.pi / 2.0);
    const world = xf.apply(.init(1, 0));
    try testing.expect(world.approxEql(.init(10, 21)));
    try testing.expect(xf.unapply(world).approxEql(.init(1, 0)));

    // `a.invMul(b)` applied to a point in b's frame lands where `b` would
    // put it, seen from `a`.
    const a: Transform = .init(.init(5, 5), 0.3);
    const b: Transform = .init(.init(-2, 7), -1.1);
    const local_b: Vec2 = .init(0.5, -0.25);
    const direct = a.unapply(b.apply(local_b));
    const composed = a.invMul(b).apply(local_b);
    try testing.expect(direct.approxEql(composed));
}

test "boxes overlap when they touch, and not when they do not" {
    const a: Aabb = .{ .min = .init(0, 0), .max = .init(1, 1) };
    const b: Aabb = .{ .min = .init(1, 0), .max = .init(2, 1) };
    const c: Aabb = .{ .min = .init(1.01, 0), .max = .init(2, 1) };
    try testing.expect(a.overlaps(b));
    try testing.expect(!a.overlaps(c));
    try testing.expect(a.contains(.init(0.5, 0.5)));
    try testing.expect(!a.contains(.init(1.5, 0.5)));

    const both = a.join(c);
    try testing.expect(both.min.eql(.init(0, 0)));
    try testing.expect(both.max.eql(.init(2, 1)));
    try testing.expect(both.center().approxEql(.init(1, 0.5)));
}

test "the two cross products are the two quarter turns" {
    const v: Vec2 = .init(1, 0);
    try testing.expect(crossVS(v, 1).eql(.init(0, -1)));
    try testing.expect(crossSV(1, v).eql(.init(0, 1)));
    try testing.expectEqual(@as(f32, 1), cross(.unit_x, .unit_y));
}
