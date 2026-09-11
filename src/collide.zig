// SPDX-License-Identifier: BSD-2-Clause

//! The narrow phase: given two shapes that might touch, where and how deep.
//!
//! ```zig
//! const m = collide.polygons(&crate, crate_xf, &ramp, ramp_xf, 0);
//! for (m.points[0..m.count]) |p| { ... p.point, p.separation ... }
//! ```
//!
//! What comes out is a **manifold**: one normal, and one or two points along
//! it with how far each is inside. One normal for the pair, rather than one
//! per point, because two convex shapes touch along at most one edge, and
//! an edge has one direction. Two points, because a box resting on the
//! ground touches it along an edge and one point would let it rock.
//!
//! **Every point carries an id.** The solver keeps the impulse it found for
//! each point and starts the next step from it - *warm starting* - which is
//! most of why a stack of crates stands still instead of trembling. An id
//! names which corner of which face made the point, so the impulse follows
//! the corner and not the array index; when a box slides and its contacts
//! swap order, the impulses swap with them.
//!
//! **The arithmetic is Box2D's**, from the version everyone learned it from:
//! the separating axis test to find the reference face, the incident edge
//! clipped against it, and the circle cases by regions of a polygon. It is
//! well understood and its failure modes are known, which for a solver is
//! worth more than novelty.
//!
//! **`margin` reaches a little past touching.** With a margin of zero a
//! manifold is only ever made of points that are touching or sunk in. With
//! more, points up to that far apart are kept too, with a positive
//! separation - *speculative* points, which the solver lets close by the
//! next substep and no further. The world asks for a margin only for a
//! body the continuous pass has just stopped short of something, and which
//! is still moving at it; see `World` for why not for every pair.
//!
//! Nothing here allocates, locks or reads anything but its arguments, which
//! is what lets every pair the broad phase found be tested at the same time
//! on every core.

const std = @import("std");
const testing = std.testing;

const geometry = @import("geometry.zig");
const Vec2 = geometry.Vec2;
const Transform = geometry.Transform;
const cross = geometry.cross;
const shape = @import("shape.zig");
const Circle = shape.Circle;
const Polygon = shape.Polygon;

/// One place two shapes touch.
pub const Point = struct {
    /// In the world, midway between the two surfaces.
    point: Vec2,
    /// Negative when inside each other, which is the usual case for a
    /// contact that is holding something up: the solver pushes until it is
    /// nearly zero and no further.
    separation: f32,
    /// Which features made this point. See the module comment.
    id: u16,
};

/// Where two shapes touch, if they do.
pub const Manifold = struct {
    /// Unit length, from shape A towards shape B. Zero when `count` is.
    normal: Vec2 = .zero,
    points: [2]Point = undefined,
    count: u32 = 0,

    /// Not touching.
    pub const none: Manifold = .{};

    pub fn pointSlice(self: *const Manifold) []const Point {
        return self.points[0..self.count];
    }
};

/// A tolerance the tests below use where a division would otherwise be by
/// something that is zero to the bit only by luck.
const eps: f32 = std.math.floatEps(f32);

// -------------------------------------------------------------------------
// Circle against circle
// -------------------------------------------------------------------------

pub fn circles(a: Circle, xa: Transform, b: Circle, xb: Transform, margin: f32) Manifold {
    const pa = xa.apply(a.center);
    const pb = xb.apply(b.center);
    const d = pb.sub(pa);
    const dist_sq = d.lenSq();
    const r = a.radius + b.radius;
    if (dist_sq > (r + margin) * (r + margin)) return .none;

    const dist = @sqrt(dist_sq);
    // Two circles on exactly the same point have no direction between them.
    // Pushing them apart along +y is as good as any, and better than NaN.
    const normal: Vec2 = if (dist > eps) d.scale(1 / dist) else .unit_y;
    const surface_a = pa.mulAdd(normal, a.radius);
    const surface_b = pb.mulAdd(normal, -b.radius);
    var m: Manifold = .{ .normal = normal, .count = 1 };
    m.points[0] = .{
        .point = surface_a.lerp(surface_b, 0.5),
        .separation = dist - r,
        .id = 0,
    };
    return m;
}

// -------------------------------------------------------------------------
// Polygon against circle
// -------------------------------------------------------------------------

/// The polygon is shape A, so the normal points from it towards the circle.
pub fn polygonCircle(poly: *const Polygon, xa: Transform, circle: Circle, xb: Transform, margin: f32) Manifold {
    // The circle's centre in the polygon's frame, where the polygon's
    // normals mean something.
    const c_world = xb.apply(circle.center);
    const c = xa.unapply(c_world);
    const radius = circle.radius;
    // How far out the centre may be and still make a point.
    const reach = radius + margin;

    // The face the centre is furthest outside of. If it is outside any face
    // by more than that, nothing touches.
    var best: usize = 0;
    var separation: f32 = -std.math.floatMax(f32);
    for (poly.vertexSlice(), poly.normalSlice(), 0..) |v, n, i| {
        const s = n.dot(c.sub(v));
        if (s > reach) return .none;
        if (s > separation) {
            separation = s;
            best = i;
        }
    }

    const v1 = poly.vertices[best];
    const v2 = poly.vertices[(best + 1) % poly.count];
    const n_local = poly.normals[best];

    // The centre is inside the polygon. The nearest face is the way out.
    if (separation < eps) {
        const normal = xa.q.rotate(n_local);
        return facePoint(c_world, normal, radius, separation, @intCast(best));
    }

    // Which part of the face is nearest: past one end, past the other, or
    // the face itself.
    const along1 = c.sub(v1).dot(v2.sub(v1));
    const along2 = c.sub(v2).dot(v1.sub(v2));
    if (along1 <= 0) return vertexPoint(xa, c_world, v1, radius, reach, 0x100 | @as(u16, @intCast(best)));
    if (along2 <= 0) return vertexPoint(xa, c_world, v2, radius, reach, 0x100 | @as(u16, @intCast((best + 1) % poly.count)));

    const face_center = v1.add(v2).scale(0.5);
    const s = c.sub(face_center).dot(n_local);
    if (s > reach) return .none;
    return facePoint(c_world, xa.q.rotate(n_local), radius, s, @intCast(best));
}

/// The circle's centre is `distance` outside a face with world `normal`:
/// the contact is midway between the face and the circle's surface.
fn facePoint(c_world: Vec2, normal: Vec2, radius: f32, distance: f32, feature_id: u16) Manifold {
    var m: Manifold = .{ .normal = normal, .count = 1 };
    m.points[0] = .{
        .point = c_world.mulAdd(normal, -0.5 * (radius + distance)),
        .separation = distance - radius,
        .id = feature_id,
    };
    return m;
}

/// The circle's centre is nearest a corner: the normal runs from the corner
/// to the centre. Nothing if the centre is further than `reach` from it.
fn vertexPoint(xa: Transform, c_world: Vec2, v_local: Vec2, radius: f32, reach: f32, feature_id: u16) Manifold {
    const v = xa.apply(v_local);
    const d = c_world.sub(v);
    const dist_sq = d.lenSq();
    if (dist_sq > reach * reach) return .none;
    const dist = @sqrt(dist_sq);
    const normal: Vec2 = if (dist > eps) d.scale(1 / dist) else .unit_y;
    var m: Manifold = .{ .normal = normal, .count = 1 };
    m.points[0] = .{
        .point = v.lerp(c_world.mulAdd(normal, -radius), 0.5),
        .separation = dist - radius,
        .id = feature_id,
    };
    return m;
}

// -------------------------------------------------------------------------
// Polygon against polygon
// -------------------------------------------------------------------------

/// How the features that made a clipped point are packed into its id.
///
/// Three bits each for two vertex indices (eight vertices at most), a bit
/// for which kind of feature each was, and a bit for which polygon was the
/// reference. Two points from one pair never pack the same, and the same
/// two corners pack the same next step, which is all an id has to do.
const Feature = packed struct(u16) {
    index_a: u3,
    index_b: u3,
    type_a: enum(u1) { vertex, face },
    type_b: enum(u1) { vertex, face },
    flipped: bool,
    _: u7 = 0,

    fn id(self: Feature) u16 {
        return @bitCast(self);
    }
};

const ClipVertex = struct {
    v: Vec2,
    feature: Feature,
};

/// The face of `poly1` that `poly2` is furthest outside of, and by how
/// much. Positive means separated along that face.
///
/// `poly1` is brought into `poly2`'s frame once, so each of its normals is
/// turned once rather than each of `poly2`'s vertices being turned for each
/// normal.
fn findMaxSeparation(poly1: *const Polygon, xf1: Transform, poly2: *const Polygon, xf2: Transform) struct { edge: usize, separation: f32 } {
    const xf = xf2.invMul(xf1);
    var best: usize = 0;
    var max_separation: f32 = -std.math.floatMax(f32);
    for (poly1.vertexSlice(), poly1.normalSlice(), 0..) |v1_local, n1_local, i| {
        const n = xf.q.rotate(n1_local);
        const v1 = xf.apply(v1_local);
        var si: f32 = std.math.floatMax(f32);
        for (poly2.vertexSlice()) |v2| {
            si = @min(si, n.dot(v2.sub(v1)));
        }
        if (si > max_separation) {
            max_separation = si;
            best = i;
        }
    }
    return .{ .edge = best, .separation = max_separation };
}

/// The edge of `poly2` most nearly facing `edge1` of `poly1`: the one to be
/// clipped against the reference face. In world coordinates.
fn findIncidentEdge(poly1: *const Polygon, xf1: Transform, edge1: usize, poly2: *const Polygon, xf2: Transform, flipped: bool) [2]ClipVertex {
    // The reference normal, in poly2's frame.
    const normal1 = xf2.q.invRotate(xf1.q.rotate(poly1.normals[edge1]));

    var index: usize = 0;
    var min_dot: f32 = std.math.floatMax(f32);
    for (poly2.normalSlice(), 0..) |n2, i| {
        const d = normal1.dot(n2);
        if (d < min_dot) {
            min_dot = d;
            index = i;
        }
    }
    const first = index;
    const second = (index + 1) % poly2.count;

    return .{
        .{
            .v = xf2.apply(poly2.vertices[first]),
            .feature = .{ .index_a = @intCast(edge1), .index_b = @intCast(first), .type_a = .face, .type_b = .vertex, .flipped = flipped },
        },
        .{
            .v = xf2.apply(poly2.vertices[second]),
            .feature = .{ .index_a = @intCast(edge1), .index_b = @intCast(second), .type_a = .face, .type_b = .vertex, .flipped = flipped },
        },
    };
}

/// Sutherland-Hodgman for one segment against one line: what is left of
/// the segment on the inside of the plane `normal . x <= offset`. A point
/// made where the segment crosses the plane is named after the reference
/// vertex that made the side plane.
fn clipSegment(in: [2]ClipVertex, normal: Vec2, offset: f32, vertex_index_a: usize) struct { out: [2]ClipVertex, count: u32 } {
    var out: [2]ClipVertex = undefined;
    var count: u32 = 0;

    const d0 = normal.dot(in[0].v) - offset;
    const d1 = normal.dot(in[1].v) - offset;

    if (d0 <= 0) {
        out[count] = in[0];
        count += 1;
    }
    if (d1 <= 0) {
        out[count] = in[1];
        count += 1;
    }

    if (d0 * d1 < 0) {
        const t = d0 / (d0 - d1);
        out[count] = .{
            .v = in[0].v.lerp(in[1].v, t),
            .feature = .{
                .index_a = @intCast(vertex_index_a),
                .index_b = in[0].feature.index_b,
                .type_a = .vertex,
                .type_b = .face,
                .flipped = in[0].feature.flipped,
            },
        };
        count += 1;
    }
    return .{ .out = out, .count = count };
}

/// Two convex polygons. The normal points from A towards B.
pub fn polygons(a: *const Polygon, xa: Transform, b: *const Polygon, xb: Transform, margin: f32) Manifold {
    const from_a = findMaxSeparation(a, xa, b, xb);
    if (from_a.separation > margin) return .none;
    const from_b = findMaxSeparation(b, xb, a, xa);
    if (from_b.separation > margin) return .none;

    // Which polygon supplies the reference face. B's is chosen only when it
    // is clearly better, so a pair whose two candidates are nearly equal
    // does not flip between them every step and shake.
    const tolerance: f32 = 0.1 * 0.005;
    const flip = from_b.separation > 0.98 * from_a.separation + tolerance;

    const poly1 = if (flip) b else a;
    const poly2 = if (flip) a else b;
    const xf1 = if (flip) xb else xa;
    const xf2 = if (flip) xa else xb;
    const edge1 = if (flip) from_b.edge else from_a.edge;

    const incident = findIncidentEdge(poly1, xf1, edge1, poly2, xf2, flip);

    const iv1 = edge1;
    const iv2 = (edge1 + 1) % poly1.count;
    const v11_local = poly1.vertices[iv1];
    const v12_local = poly1.vertices[iv2];

    const local_tangent = v12_local.sub(v11_local).norm();
    const tangent = xf1.q.rotate(local_tangent);
    const normal = geometry.crossVS(tangent, 1);

    const v11 = xf1.apply(v11_local);
    const v12 = xf1.apply(v12_local);

    const front_offset = normal.dot(v11);
    const side_offset1 = -tangent.dot(v11);
    const side_offset2 = tangent.dot(v12);

    // Clip the incident edge to the two side planes of the reference face.
    const first = clipSegment(incident, tangent.neg(), side_offset1, iv1);
    if (first.count < 2) return .none;
    const second = clipSegment(first.out, tangent, side_offset2, iv2);
    if (second.count < 2) return .none;

    var m: Manifold = .{ .normal = if (flip) normal.neg() else normal };
    for (second.out) |cp| {
        const separation = normal.dot(cp.v) - front_offset;
        if (separation <= margin) {
            m.points[m.count] = .{
                // The clipped point is on the incident face; move it halfway
                // to the reference face so the contact sits between the two
                // surfaces.
                .point = cp.v.mulAdd(normal, -0.5 * separation),
                .separation = separation,
                .id = cp.feature.id(),
            };
            m.count += 1;
        }
    }
    return m;
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "two circles touch along the line between their centres" {
    const a: Circle = .{ .radius = 1 };
    const b: Circle = .{ .radius = 0.5 };
    const m = circles(a, .init(.zero, 0), b, .init(.init(1.2, 0), 0), 0);
    try testing.expectEqual(@as(u32, 1), m.count);
    try testing.expect(m.normal.approxEql(.unit_x));
    try testing.expectApproxEqAbs(@as(f32, -0.3), m.points[0].separation, 1e-6);
    // Midway between x = 1 and x = 0.7.
    try testing.expectApproxEqAbs(@as(f32, 0.85), m.points[0].point.x, 1e-6);

    try testing.expectEqual(@as(u32, 0), circles(a, .identity, b, .init(.init(2, 0), 0), 0).count);
}

test "a circle on a box's face, corner, and inside it" {
    const box: Polygon = .box(1, 1);
    const ball: Circle = .{ .radius = 0.5 };

    // Sitting on the face at +y, slightly inside.
    const on_face = polygonCircle(&box, .identity, ball, .init(.init(0.3, 1.4), 0), 0);
    try testing.expectEqual(@as(u32, 1), on_face.count);
    try testing.expect(on_face.normal.approxEql(.unit_y));
    try testing.expectApproxEqAbs(@as(f32, -0.1), on_face.points[0].separation, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.3), on_face.points[0].point.x, 1e-6);

    // Off the corner, diagonally.
    const at_corner = polygonCircle(&box, .identity, ball, .init(.init(1.3, 1.3), 0), 0);
    try testing.expectEqual(@as(u32, 1), at_corner.count);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / @sqrt(2.0)), at_corner.normal.x, 1e-5);
    try testing.expect(at_corner.points[0].separation < 0);
    try testing.expect(at_corner.points[0].id != on_face.points[0].id);

    // Past the corner, not touching.
    try testing.expectEqual(@as(u32, 0), polygonCircle(&box, .identity, ball, .init(.init(1.4, 1.4), 0), 0).count);

    // Deep inside: pushed out of the nearest face, which is +x here.
    const inside = polygonCircle(&box, .identity, ball, .init(.init(0.8, 0.1), 0), 0);
    try testing.expectEqual(@as(u32, 1), inside.count);
    try testing.expect(inside.normal.approxEql(.unit_x));
    try testing.expectApproxEqAbs(@as(f32, -0.7), inside.points[0].separation, 1e-6);
}

test "a box resting on a wider box makes two points and a normal towards it" {
    const ground: Polygon = .box(5, 0.5);
    const crate: Polygon = .box(0.5, 0.5);
    // Ground top at y = -0.5; crate bottom at y = -0.45 is 0.05 inside.
    const m = polygons(&ground, .identity, &crate, .init(.init(1, -0.95), 0), 0);
    try testing.expectEqual(@as(u32, 2), m.count);
    try testing.expect(m.normal.approxEql(.init(0, -1)));
    for (m.pointSlice()) |p| {
        try testing.expectApproxEqAbs(@as(f32, -0.05), p.separation, 1e-5);
        try testing.expectApproxEqAbs(@as(f32, -0.475), p.point.y, 1e-5);
    }
    try testing.expect(m.points[0].id != m.points[1].id);
    try testing.expect(@abs(m.points[0].point.x - m.points[1].point.x) > 0.9);

    // The same pair the other way round has the opposite normal.
    const back = polygons(&crate, .init(.init(1, -0.95), 0), &ground, .identity, 0);
    try testing.expectEqual(@as(u32, 2), back.count);
    try testing.expect(back.normal.approxEql(.init(0, 1)));

    // Lifted clear, nothing.
    try testing.expectEqual(@as(u32, 0), polygons(&ground, .identity, &crate, .init(.init(1, -1.2), 0), 0).count);
}

test "a margin keeps points that are near, with how far apart they are" {
    const ground: Polygon = .box(5, 0.5);
    const crate: Polygon = .box(0.5, 0.5);
    const ball: Circle = .{ .radius = 0.5 };
    // Each a hundredth clear of the ground, whose top is at y = -0.5.
    const crate_xf: Transform = .init(.init(1, -1.01), 0);
    const ball_xf: Transform = .init(.init(1, -1.01), 0);

    try testing.expectEqual(@as(u32, 0), polygons(&ground, .identity, &crate, crate_xf, 0).count);
    const near = polygons(&ground, .identity, &crate, crate_xf, 0.02);
    try testing.expectEqual(@as(u32, 2), near.count);
    for (near.pointSlice()) |p| try testing.expectApproxEqAbs(@as(f32, 0.01), p.separation, 1e-5);
    // And not past the margin.
    try testing.expectEqual(@as(u32, 0), polygons(&ground, .identity, &crate, crate_xf, 0.005).count);

    try testing.expectEqual(@as(u32, 0), polygonCircle(&ground, .identity, ball, ball_xf, 0).count);
    const near_ball = polygonCircle(&ground, .identity, ball, ball_xf, 0.02);
    try testing.expectEqual(@as(u32, 1), near_ball.count);
    try testing.expectApproxEqAbs(@as(f32, 0.01), near_ball.points[0].separation, 1e-5);

    const other: Transform = .init(.init(1.01, 0), 0);
    try testing.expectEqual(@as(u32, 0), circles(ball, .identity, ball, other, 0).count);
    try testing.expectApproxEqAbs(@as(f32, 0.01), circles(ball, .identity, ball, other, 0.02).points[0].separation, 1e-5);
}

test "a turned box on a face touches at one corner" {
    const ground: Polygon = .box(5, 0.5);
    const crate: Polygon = .box(0.5, 0.5);
    const r = @sqrt(2.0) * 0.5;
    // Balanced on a corner, that corner 0.02 into the ground.
    const m = polygons(&ground, .identity, &crate, .init(.init(0, -0.5 - r + 0.02), std.math.pi / 4.0), 0);
    try testing.expectEqual(@as(u32, 1), m.count);
    try testing.expectApproxEqAbs(@as(f32, -0.02), m.points[0].separation, 1e-4);
    try testing.expect(m.normal.approxEql(.init(0, -1)));
}

test "ids follow the corners when the incident edge is clipped" {
    // A crate hanging over the edge of the ground: one point is a crate
    // corner, the other is made by the ground's side plane.
    const ground: Polygon = .box(1, 0.5);
    const crate: Polygon = .box(0.5, 0.5);
    const m = polygons(&ground, .identity, &crate, .init(.init(1.2, -0.98), 0), 0);
    try testing.expectEqual(@as(u32, 2), m.count);
    const fa: Feature = @bitCast(m.points[0].id);
    const fb: Feature = @bitCast(m.points[1].id);
    try testing.expect(fa.type_a != fb.type_a);
}
