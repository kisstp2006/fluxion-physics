// SPDX-License-Identifier: BSD-2-Clause

//! The narrow phase: given two shapes that might touch, where and how deep.
//!
//! ```zig
//! const m = collide.polygons(&crate, crate_xf, &ramp, ramp_xf, 0, .none);
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
//! worth more than novelty. A capsule pairs as its core - a polygon of two
//! corners - with its radius round it, as Box2D v3 pairs rounded polygons:
//! the separations less the radii, the points midway between the rounded
//! surfaces, and two cores apart at two corners touching corner to corner.
//!
//! **`hidden` marks the edges of a polygon that another piece of the level
//! covers**, one bit an edge (`Hidden`). A floor of separate tiles is a row
//! of boxes whose sides touch, and to the narrow phase, which sees one tile
//! at a time, the side of the next tile along is a wall: a box resting a
//! slop deep in one tile meets it before it meets that tile's top, and is
//! pushed back - stopped dead, sliding slowly across a seam. No contact is
//! made against a hidden edge. Between two polygons, the shallowest way out
//! that is not through one is used instead; a circle beyond one is left to
//! the tile that covers it; and a corner with a hidden edge on one side is
//! not a corner at all, but the other edge running straight on. Jolt and
//! Bullet do the same for the triangles of a mesh; Box2D asks for the level
//! as a chain of segments instead. The world works out which edges are
//! hidden - see `World.coverEdges`.
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
const Capsule = shape.Capsule;

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

/// Which edges of a polygon are hidden, bit `i` for edge `i` - the one from
/// vertex `i` to the next. See the module comment. Eight bits, because a
/// polygon has eight edges at most.
pub const Hidden = u8;

/// Nothing hidden: every shape that is not part of the level.
pub const none_hidden: Hidden = 0;

inline fn isHidden(hidden: Hidden, edge: usize) bool {
    return hidden & (@as(Hidden, 1) << @intCast(edge)) != 0;
}

/// What a pairing is told about the seams of the level: which edges of
/// each shape are hidden, and how deep a contact may be and still be one
/// made at a seam.
///
/// Only a shallow contact is a seam's: a box sliding over one has sunk the
/// slop a resting box sinks, no more. Deeper, the shape has been pushed
/// into the level, and is pushed out the way it would have been had there
/// been no seam - whichever way is shortest, covered or not - because a
/// shape sunk into the middle of a floor that every tile declined to push
/// would fall through it.
pub const Seams = struct {
    /// Shape A's hidden edges, and shape B's.
    a: Hidden = none_hidden,
    b: Hidden = none_hidden,
    /// The deepest a contact may be and still be a seam's. World units.
    depth: f32 = 0,

    /// No seams: two shapes that are not both part of a level.
    pub const none: Seams = .{};

    /// The same, for the pairing written the other way round.
    pub fn swapped(self: Seams) Seams {
        return .{ .a = self.b, .b = self.a, .depth = self.depth };
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
/// `seams.a`: the polygon's covered edges; see `Seams`.
pub fn polygonCircle(poly: *const Polygon, xa: Transform, circle: Circle, xb: Transform, margin: f32, seams: Seams) Manifold {
    const hidden = seams.a;
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

    // The centre is inside the polygon. The nearest face is the way out -
    // and when that is a hidden one and the circle is only just in, the
    // nearest that is not, if that is as shallow; if it is not, the piece
    // covering the hidden face holds the circle, not this one.
    if (separation < eps) {
        var face = best;
        if (isHidden(hidden, best) and radius - separation <= seams.depth) {
            var open = -std.math.floatMax(f32);
            var found = false;
            for (poly.vertexSlice(), poly.normalSlice(), 0..) |v, n, i| {
                if (isHidden(hidden, i)) continue;
                const s = n.dot(c.sub(v));
                if (s > open) {
                    open = s;
                    face = i;
                    found = true;
                }
            }
            if (!found or radius - open > seams.depth) return .none;
            separation = open;
        }
        return facePoint(c_world, xa.q.rotate(poly.normals[face]), radius, separation, @intCast(face));
    }

    // Which part of the face is nearest: past one end, past the other, or
    // the face itself.
    const v1 = poly.vertices[best];
    const v2 = poly.vertices[(best + 1) % poly.count];
    const along1 = c.sub(v1).dot(v2.sub(v1));
    const along2 = c.sub(v2).dot(v1.sub(v2));
    if (along1 <= 0) return cornerPoint(poly, xa, c_world, c, best, radius, reach, seams);
    if (along2 <= 0) return cornerPoint(poly, xa, c_world, c, (best + 1) % poly.count, radius, reach, seams);

    const n_local = poly.normals[best];
    const s = c.sub(v1.add(v2).scale(0.5)).dot(n_local);
    if (s > reach) return .none;
    // Just over a hidden face is the piece of level that hides it, and the
    // circle is resting on that one's face, not this one's side.
    if (isHidden(hidden, best) and radius - s <= seams.depth) return .none;
    return facePoint(c_world, xa.q.rotate(n_local), radius, s, @intCast(best));
}

/// The circle's centre is nearest corner `k`, between edge `k - 1` and edge
/// `k`. A corner with a hidden edge on one side is not a corner: the
/// surface runs straight on through it, so a shallow contact there is with
/// the edge that is not hidden, as though the circle were over its middle.
/// With both sides hidden it is inside the level, and the pieces around it
/// hold the circle.
fn cornerPoint(poly: *const Polygon, xa: Transform, c_world: Vec2, c: Vec2, k: usize, radius: f32, reach: f32, seams: Seams) Manifold {
    const corner = poly.vertices[k];
    const before = (k + poly.count - 1) % poly.count;
    const hide_before = isHidden(seams.a, before);
    const hide_after = isHidden(seams.a, k);
    const plain = (!hide_before and !hide_after) or radius - c.dist(corner) > seams.depth;
    if (plain) return vertexPoint(xa, c_world, corner, radius, reach, 0x100 | @as(u16, @intCast(k)));
    if (hide_before and hide_after) return .none;
    const face = if (hide_before) k else before;
    const n = poly.normals[face];
    const s = n.dot(c.sub(corner));
    if (s > reach) return .none;
    return facePoint(c_world, xa.q.rotate(n), radius, s, @intCast(face));
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

/// Like `findMaxSeparation`, but only among the faces that may make a
/// contact: not hidden themselves, and not facing a hidden edge of `poly2`
/// - which is the edge the contact would be made against. Null when there
/// are none.
fn findMaxVisibleSeparation(poly1: *const Polygon, xf1: Transform, hidden1: Hidden, poly2: *const Polygon, xf2: Transform, hidden2: Hidden) ?struct { edge: usize, separation: f32 } {
    const xf = xf2.invMul(xf1);
    var best: ?usize = null;
    var max_separation: f32 = -std.math.floatMax(f32);
    for (poly1.vertexSlice(), poly1.normalSlice(), 0..) |v1_local, n1_local, i| {
        if (isHidden(hidden1, i)) continue;
        const n = xf.q.rotate(n1_local);
        const v1 = xf.apply(v1_local);
        var si: f32 = std.math.floatMax(f32);
        for (poly2.vertexSlice()) |v2| {
            si = @min(si, n.dot(v2.sub(v1)));
        }
        if (si <= max_separation) continue;
        if (isHidden(hidden2, incidentIndex(poly1, xf1, i, poly2, xf2))) continue;
        max_separation = si;
        best = i;
    }
    const edge = best orelse return null;
    return .{ .edge = edge, .separation = max_separation };
}

/// Which edge of `poly2` most nearly faces `edge1` of `poly1`.
fn incidentIndex(poly1: *const Polygon, xf1: Transform, edge1: usize, poly2: *const Polygon, xf2: Transform) usize {
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
    return index;
}

/// The edge of `poly2` most nearly facing `edge1` of `poly1`: the one to be
/// clipped against the reference face. In world coordinates.
fn findIncidentEdge(poly1: *const Polygon, xf1: Transform, edge1: usize, poly2: *const Polygon, xf2: Transform, flipped: bool) [2]ClipVertex {
    const first = incidentIndex(poly1, xf1, edge1, poly2, xf2);
    const second = (first + 1) % poly2.count;

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

/// Two convex polygons. The normal points from A towards B. `seams`: each
/// one's covered edges; see `Seams`.
pub fn polygons(a: *const Polygon, xa: Transform, b: *const Polygon, xb: Transform, margin: f32, seams: Seams) Manifold {
    return roundedPolygons(a, 0, xa, b, 0, xb, margin, seams);
}

/// A polygon, as A, and a capsule. `seams.a`: the polygon's covered edges.
pub fn polygonCapsule(poly: *const Polygon, xa: Transform, capsule: Capsule, xb: Transform, margin: f32, seams: Seams) Manifold {
    const core = capsule.core();
    return roundedPolygons(poly, 0, xa, &core, capsule.radius, xb, margin, seams);
}

/// Two capsules.
pub fn capsules(a: Capsule, xa: Transform, b: Capsule, xb: Transform, margin: f32) Manifold {
    const core_a = a.core();
    const core_b = b.core();
    return roundedPolygons(&core_a, a.radius, xa, &core_b, b.radius, xb, margin, .none);
}

/// A capsule, as A, and a circle: the circle against the nearest point of
/// the capsule's segment, as two circles.
pub fn capsuleCircle(capsule: Capsule, xa: Transform, circle: Circle, xb: Transform, margin: f32) Manifold {
    const p1 = xa.apply(capsule.center1);
    const p2 = xa.apply(capsule.center2);
    const c = xb.apply(circle.center);
    const nearest = shape.nearestOnSegment(c, p1, p2);
    const d = c.sub(nearest);
    const dist_sq = d.lenSq();
    const r = capsule.radius + circle.radius;
    if (dist_sq > (r + margin) * (r + margin)) return .none;
    const dist = @sqrt(dist_sq);
    const normal: Vec2 = if (dist > eps) d.scale(1 / dist) else xa.q.rotate(capsule.core().normals[0]);
    var m: Manifold = .{ .normal = normal, .count = 1 };
    m.points[0] = .{
        .point = nearest.mulAdd(normal, capsule.radius).lerp(c.mulAdd(normal, -circle.radius), 0.5),
        .separation = dist - r,
        .id = 0,
    };
    return m;
}

/// Two convex polygons with a radius round each - zero for a polygon, a
/// capsule's for its core. The separations are the cores', less the radii;
/// the points are midway between the rounded surfaces.
fn roundedPolygons(a: *const Polygon, ra: f32, xa: Transform, b: *const Polygon, rb: f32, xb: Transform, margin: f32, seams: Seams) Manifold {
    const radius = ra + rb;
    const from_a = findMaxSeparation(a, xa, b, xb);
    if (from_a.separation > margin + radius) return .none;
    const from_b = findMaxSeparation(b, xb, a, xa);
    if (from_b.separation > margin + radius) return .none;

    // Which polygon supplies the reference face. B's is chosen only when it
    // is clearly better, so a pair whose two candidates are nearly equal
    // does not flip between them every step and shake.
    const tolerance: f32 = 0.1 * 0.005;
    var flip = from_b.separation > 0.98 * from_a.separation + tolerance;
    var edge1 = if (flip) from_b.edge else from_a.edge;

    // The shallowest way out may be through an edge of the level that the
    // next piece covers: the side of the next tile of a floor, met by a box
    // sunk a slop into this one. Then, if the box is only that shallowly in,
    // the shallowest way out that exists is used instead - out through the
    // top - provided it is as shallow; if it is not, the box is not on this
    // piece's surface at all, and the pieces around it hold it. Only when
    // something is hidden; and the usual answer whenever it is allowed.
    if ((seams.a | seams.b) != none_hidden) {
        const allowed = if (flip)
            !isHidden(seams.b, edge1) and !isHidden(seams.a, incidentIndex(b, xb, edge1, a, xa))
        else
            !isHidden(seams.a, edge1) and !isHidden(seams.b, incidentIndex(a, xa, edge1, b, xb));
        const shallow = @max(from_a.separation, from_b.separation) - radius >= -seams.depth;
        if (!allowed and shallow) {
            const open_a = findMaxVisibleSeparation(a, xa, seams.a, b, xb, seams.b);
            const open_b = findMaxVisibleSeparation(b, xb, seams.b, a, xa, seams.a);
            const deep = radius - seams.depth;
            const ok_a = open_a != null and open_a.?.separation >= deep;
            const ok_b = open_b != null and open_b.?.separation >= deep;
            if (!ok_a and !ok_b) return .none;
            flip = if (!ok_a) true else if (!ok_b) false else open_b.?.separation > 0.98 * open_a.?.separation + tolerance;
            edge1 = if (flip) open_b.?.edge else open_a.?.edge;
        }
    }

    const poly1 = if (flip) b else a;
    const poly2 = if (flip) a else b;
    const xf1 = if (flip) xb else xa;
    const xf2 = if (flip) xa else xb;
    const r1 = if (flip) rb else ra;
    const r2 = if (flip) ra else rb;

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

    // Rounded cores that are apart touch where their surfaces nearest each
    // other do, which past the ends of the two edges is corner to corner -
    // and there the reference face's normal is not the way they touch: a
    // capsule over the edge of a step touches the step's corner along the
    // line between the two, and rolls off it.
    if (radius > 0 and @max(from_a.separation, from_b.separation) > 0) {
        const nearest = segmentsNearest(v11, v12, incident[0].v, incident[1].v);
        const at_end1 = nearest.fraction1 == 0 or nearest.fraction1 == 1;
        const at_end2 = nearest.fraction2 == 0 or nearest.fraction2 == 1;
        // A corner with a covered edge beside it is no corner: the surface
        // runs straight on into the next piece of the level, and what rests
        // there rests on the face - as for a circle, in `cornerPoint`.
        const hidden1 = if (flip) seams.b else seams.a;
        const hidden2 = if (flip) seams.a else seams.b;
        const corner1: usize = if (nearest.fraction1 == 0) iv1 else iv2;
        const corner2: usize = if (nearest.fraction2 == 0) incident[0].feature.index_b else incident[1].feature.index_b;
        const covered = isHidden(hidden1, corner1) or isHidden(hidden1, (corner1 + poly1.count - 1) % poly1.count) or
            isHidden(hidden2, corner2) or isHidden(hidden2, (corner2 + poly2.count - 1) % poly2.count);
        const dist = @sqrt(nearest.distance_sq);
        const way: Vec2 = if (dist > eps) nearest.point2.sub(nearest.point1).scale(1 / dist) else normal;
        // Straight out of the face, the corners are the face's: two edges
        // side by side whose ends are level touch along their length, and
        // the face makes the two points that hold them.
        const along_face = way.dot(normal) > 1 - 1e-4;
        if (at_end1 and at_end2 and !covered and !along_face) {
            if (dist > margin + radius) return .none;
            const surface1 = nearest.point1.mulAdd(way, r1);
            const surface2 = nearest.point2.mulAdd(way, -r2);
            var m: Manifold = .{ .normal = if (flip) way.neg() else way, .count = 1 };
            m.points[0] = .{
                .point = surface1.lerp(surface2, 0.5),
                .separation = dist - radius,
                .id = (Feature{ .index_a = @intCast(corner1), .index_b = @intCast(corner2), .type_a = .vertex, .type_b = .vertex, .flipped = flip }).id(),
            };
            return m;
        }
    }

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
        const core_separation = normal.dot(cp.v) - front_offset;
        const separation = core_separation - radius;
        if (separation <= margin) {
            m.points[m.count] = .{
                // The clipped point is on the incident core; the contact
                // sits midway between the two rounded surfaces - the
                // reference face moved out by its radius, the incident
                // point in by its.
                .point = cp.v.mulAdd(normal, 0.5 * (r1 - r2 - core_separation)),
                .separation = separation,
                .id = cp.feature.id(),
            };
            m.count += 1;
        }
    }
    return m;
}

/// The nearest points of two segments, how far along each they are, and
/// how far apart, squared.
const Nearest = struct {
    point1: Vec2,
    point2: Vec2,
    fraction1: f32,
    fraction2: f32,
    distance_sq: f32,
};

/// The nearest points of the segments from `p1` to `q1` and from `p2` to
/// `q2`: Ericson's, from Real-Time Collision Detection, a fraction clamped
/// to its segment exactly at an end so that a caller can tell a corner.
fn segmentsNearest(p1: Vec2, q1: Vec2, p2: Vec2, q2: Vec2) Nearest {
    const d1 = q1.sub(p1);
    const d2 = q2.sub(p2);
    const r = p1.sub(p2);
    const dd1 = d1.lenSq();
    const dd2 = d2.lenSq();
    const rd1 = r.dot(d1);
    const rd2 = r.dot(d2);
    var f1: f32 = 0;
    var f2: f32 = 0;
    if (dd1 < eps and dd2 < eps) {
        // Two points.
    } else if (dd1 < eps) {
        f2 = std.math.clamp(rd2 / dd2, 0, 1);
    } else if (dd2 < eps) {
        f1 = std.math.clamp(-rd1 / dd1, 0, 1);
    } else {
        const d12 = d1.dot(d2);
        const denominator = dd1 * dd2 - d12 * d12;
        // Parallel segments have no one nearest pair: the start of the
        // first will do, and the second is then found for it.
        f1 = if (denominator != 0) std.math.clamp((d12 * rd2 - rd1 * dd2) / denominator, 0, 1) else 0;
        f2 = (d12 * f1 + rd2) / dd2;
        if (f2 < 0) {
            f2 = 0;
            f1 = std.math.clamp(-rd1 / dd1, 0, 1);
        } else if (f2 > 1) {
            f2 = 1;
            f1 = std.math.clamp((d12 - rd1) / dd1, 0, 1);
        }
    }
    const point1 = p1.mulAdd(d1, f1);
    const point2 = p2.mulAdd(d2, f2);
    return .{ .point1 = point1, .point2 = point2, .fraction1 = f1, .fraction2 = f2, .distance_sq = point1.distSq(point2) };
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
    const on_face = polygonCircle(&box, .identity, ball, .init(.init(0.3, 1.4), 0), 0, .none);
    try testing.expectEqual(@as(u32, 1), on_face.count);
    try testing.expect(on_face.normal.approxEql(.unit_y));
    try testing.expectApproxEqAbs(@as(f32, -0.1), on_face.points[0].separation, 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.3), on_face.points[0].point.x, 1e-6);

    // Off the corner, diagonally.
    const at_corner = polygonCircle(&box, .identity, ball, .init(.init(1.3, 1.3), 0), 0, .none);
    try testing.expectEqual(@as(u32, 1), at_corner.count);
    try testing.expectApproxEqAbs(@as(f32, 1.0 / @sqrt(2.0)), at_corner.normal.x, 1e-5);
    try testing.expect(at_corner.points[0].separation < 0);
    try testing.expect(at_corner.points[0].id != on_face.points[0].id);

    // Past the corner, not touching.
    try testing.expectEqual(@as(u32, 0), polygonCircle(&box, .identity, ball, .init(.init(1.4, 1.4), 0), 0, .none).count);

    // Deep inside: pushed out of the nearest face, which is +x here.
    const inside = polygonCircle(&box, .identity, ball, .init(.init(0.8, 0.1), 0), 0, .none);
    try testing.expectEqual(@as(u32, 1), inside.count);
    try testing.expect(inside.normal.approxEql(.unit_x));
    try testing.expectApproxEqAbs(@as(f32, -0.7), inside.points[0].separation, 1e-6);
}

test "a box resting on a wider box makes two points and a normal towards it" {
    const ground: Polygon = .box(5, 0.5);
    const crate: Polygon = .box(0.5, 0.5);
    // Ground top at y = -0.5; crate bottom at y = -0.45 is 0.05 inside.
    const m = polygons(&ground, .identity, &crate, .init(.init(1, -0.95), 0), 0, .none);
    try testing.expectEqual(@as(u32, 2), m.count);
    try testing.expect(m.normal.approxEql(.init(0, -1)));
    for (m.pointSlice()) |p| {
        try testing.expectApproxEqAbs(@as(f32, -0.05), p.separation, 1e-5);
        try testing.expectApproxEqAbs(@as(f32, -0.475), p.point.y, 1e-5);
    }
    try testing.expect(m.points[0].id != m.points[1].id);
    try testing.expect(@abs(m.points[0].point.x - m.points[1].point.x) > 0.9);

    // The same pair the other way round has the opposite normal.
    const back = polygons(&crate, .init(.init(1, -0.95), 0), &ground, .identity, 0, .none);
    try testing.expectEqual(@as(u32, 2), back.count);
    try testing.expect(back.normal.approxEql(.init(0, 1)));

    // Lifted clear, nothing.
    try testing.expectEqual(@as(u32, 0), polygons(&ground, .identity, &crate, .init(.init(1, -1.2), 0), 0, .none).count);
}

// A tile of a floor whose top is at y = 0, spanning x from 0.5 to 1: the
// second tile along, the first being where x is below 0.5. Its left edge -
// edge 3 of a box - is the side the first tile covers.
const next_tile: Polygon = .box(0.25, 0.25);
const next_tile_xf: Transform = .init(.init(0.75, 0.25), 0);
/// The tile with its side covered, as the world would tell it, for four
/// half-centimetre slops.
const covered_side: Seams = .{ .a = 1 << 3, .depth = 0.02 };

test "a box crossing a seam meets the next tile's top, not the side the first tile covers" {
    // Sunk half a centimetre into the first tile, as a resting box is, its
    // right face three millimetres over the seam.
    const crate: Polygon = .box(0.2, 0.2);
    const crate_xf: Transform = .init(.init(0.503 - 0.2, -0.2 + 0.005), 0);

    // Seen alone, the tile's side is the shallowest way out, and the crate
    // is pushed back across the seam. That is the snag.
    const alone = polygons(&next_tile, next_tile_xf, &crate, crate_xf, 0, .none);
    try testing.expect(alone.count > 0);
    try testing.expect(alone.normal.approxEql(.init(-1, 0)));

    // With that side hidden, the way out is up, by the half centimetre it is
    // sunk: the crate is standing on the next tile, as it is on the first.
    const seamless = polygons(&next_tile, next_tile_xf, &crate, crate_xf, 0, covered_side);
    try testing.expect(seamless.count > 0);
    try testing.expect(seamless.normal.approxEql(.init(0, -1)));
    for (seamless.pointSlice()) |p| try testing.expectApproxEqAbs(@as(f32, -0.005), p.separation, 1e-5);

    // The same with the two the other way round, the normal turned with them.
    const swapped = polygons(&crate, crate_xf, &next_tile, next_tile_xf, 0, covered_side.swapped());
    try testing.expect(swapped.normal.approxEql(.init(0, 1)));

    // Covered on its top as well - a tile under a step - a box just over
    // its corner has no shallow way out of it at all, and is left to the
    // tiles around it. The way out through its far side is most of a tile
    // deep, and taking it would throw the box.
    const under_step: Seams = .{ .a = (1 << 3) | (1 << 0), .depth = 0.02 };
    try testing.expectEqual(@as(u32, 0), polygons(&next_tile, next_tile_xf, &crate, crate_xf, 0, under_step).count);
}

test "a ball at a seam rests on the next tile's top; deep in a tile, it is pushed out as ever" {
    const ball: Circle = .{ .radius = 0.2 };
    // Three centimetres short of the seam, half a centimetre deep.
    const at_seam: Transform = .init(.init(0.47, -0.195), 0);

    // Alone, the tile's corner is the nearest thing, and it pushes the ball
    // back and up at once.
    const alone = polygonCircle(&next_tile, next_tile_xf, ball, at_seam, 0, .none);
    try testing.expectEqual(@as(u32, 1), alone.count);
    try testing.expect(alone.normal.x < -0.1);

    // With its side hidden the corner is no corner, and the ball stands on
    // the tile's top as it does on the first tile's.
    const seamless = polygonCircle(&next_tile, next_tile_xf, ball, at_seam, 0, covered_side);
    try testing.expectEqual(@as(u32, 1), seamless.count);
    try testing.expect(seamless.normal.approxEql(.init(0, -1)));
    try testing.expectApproxEqAbs(@as(f32, -0.005), seamless.points[0].separation, 1e-5);

    // Just over the seam and covered all round, the tile is inside the
    // level: the tiles with a way out hold the ball.
    const covered_all: Seams = .{ .a = 0b1111, .depth = 0.02 };
    try testing.expectEqual(@as(u32, 0), polygonCircle(&next_tile, next_tile_xf, ball, at_seam, 0, covered_all).count);

    // Sunk far into the tile - past anything a seam does - it is pushed out
    // the shortest way, hidden or not, as it would be with no seam at all:
    // out through the side, three centimetres, not the top, ten.
    const sunk: Transform = .init(.init(0.53, 0.1), 0);
    const out = polygonCircle(&next_tile, next_tile_xf, ball, sunk, 0, covered_side);
    try testing.expect(out.normal.approxEql(.init(-1, 0)));
    try testing.expectApproxEqAbs(@as(f32, -0.23), out.points[0].separation, 1e-5);
}

test "a margin keeps points that are near, with how far apart they are" {
    const ground: Polygon = .box(5, 0.5);
    const crate: Polygon = .box(0.5, 0.5);
    const ball: Circle = .{ .radius = 0.5 };
    // Each a hundredth clear of the ground, whose top is at y = -0.5.
    const crate_xf: Transform = .init(.init(1, -1.01), 0);
    const ball_xf: Transform = .init(.init(1, -1.01), 0);

    try testing.expectEqual(@as(u32, 0), polygons(&ground, .identity, &crate, crate_xf, 0, .none).count);
    const near = polygons(&ground, .identity, &crate, crate_xf, 0.02, .none);
    try testing.expectEqual(@as(u32, 2), near.count);
    for (near.pointSlice()) |p| try testing.expectApproxEqAbs(@as(f32, 0.01), p.separation, 1e-5);
    // And not past the margin.
    try testing.expectEqual(@as(u32, 0), polygons(&ground, .identity, &crate, crate_xf, 0.005, .none).count);

    try testing.expectEqual(@as(u32, 0), polygonCircle(&ground, .identity, ball, ball_xf, 0, .none).count);
    const near_ball = polygonCircle(&ground, .identity, ball, ball_xf, 0.02, .none);
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
    const m = polygons(&ground, .identity, &crate, .init(.init(0, -0.5 - r + 0.02), std.math.pi / 4.0), 0, .none);
    try testing.expectEqual(@as(u32, 1), m.count);
    try testing.expectApproxEqAbs(@as(f32, -0.02), m.points[0].separation, 1e-4);
    try testing.expect(m.normal.approxEql(.init(0, -1)));
}

test "ids follow the corners when the incident edge is clipped" {
    // A crate hanging over the edge of the ground: one point is a crate
    // corner, the other is made by the ground's side plane.
    const ground: Polygon = .box(1, 0.5);
    const crate: Polygon = .box(0.5, 0.5);
    const m = polygons(&ground, .identity, &crate, .init(.init(1.2, -0.98), 0), 0, .none);
    try testing.expectEqual(@as(u32, 2), m.count);
    const fa: Feature = @bitCast(m.points[0].id);
    const fb: Feature = @bitCast(m.points[1].id);
    try testing.expect(fa.type_a != fb.type_a);
}

test "a capsule stands on a box's face with two points, as a box would" {
    const floor: Polygon = .box(4, 0.5);
    const person: Capsule = .{ .center1 = .init(-0.5, 0), .center2 = .init(0.5, 0), .radius = 0.25 };
    // Lying on its side just into the top of the floor, y being down.
    const m = polygonCapsule(&floor, .identity, person, .init(.init(0, -0.74), 0), 0, .none);
    try testing.expectEqual(@as(u32, 2), m.count);
    try testing.expect(m.normal.approxEql(.init(0, -1)));
    try testing.expectApproxEqAbs(@as(f32, -0.01), m.points[0].separation, 1e-5);
    // Midway between the floor's top and the capsule's bottom.
    try testing.expectApproxEqAbs(@as(f32, -0.495), m.points[0].point.y, 1e-5);
}

test "a capsule over a step's corner touches it along the line between them" {
    const step: Polygon = .box(1, 1);
    // Upright, its bottom end out past the step's top right corner and just
    // over it, diagonally.
    const person: Capsule = .{ .center1 = .init(0, -1), .center2 = .init(0, 0), .radius = 0.5 };
    const at = Vec2.init(1.3, -1.3);
    const m = polygonCapsule(&step, .identity, person, .init(at, 0), 0.1, .none);
    try testing.expectEqual(@as(u32, 1), m.count);
    // Out of the corner towards the capsule's end, not straight up.
    try testing.expectApproxEqAbs(@as(f32, 1.0 / @sqrt(2.0)), m.normal.x, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -1.0 / @sqrt(2.0)), m.normal.y, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.3 * @sqrt(2.0) - 0.5), m.points[0].separation, 1e-4);
}

test "a capsule and a circle, and two capsules" {
    const person: Capsule = .{ .center1 = .init(0, -1), .center2 = .init(0, 1), .radius = 0.5 };
    const ball: Circle = .{ .radius = 0.5 };
    const side = capsuleCircle(person, .identity, ball, .init(.init(0.9, 0.3), 0), 0);
    try testing.expectEqual(@as(u32, 1), side.count);
    try testing.expect(side.normal.approxEql(.unit_x));
    try testing.expectApproxEqAbs(@as(f32, -0.1), side.points[0].separation, 1e-5);
    try testing.expectEqual(@as(u32, 0), capsuleCircle(person, .identity, ball, .init(.init(0, 2.1), 0), 0).count);

    // Two upright side by side touch along their sides, at two points.
    const pair = capsules(person, .identity, person, .init(.init(0.95, 0), 0), 0);
    try testing.expectEqual(@as(u32, 2), pair.count);
    try testing.expect(pair.normal.approxEql(.unit_x));
    try testing.expectApproxEqAbs(@as(f32, -0.05), pair.points[0].separation, 1e-5);
}

test "the nearest points of two segments, at their ends and between" {
    const cross_ = segmentsNearest(.init(-1, 0), .init(1, 0), .init(0, 1), .init(0, 3));
    try testing.expectApproxEqAbs(@as(f32, 0.5), cross_.fraction1, 1e-6);
    try testing.expectEqual(@as(f32, 0), cross_.fraction2);
    try testing.expectApproxEqAbs(@as(f32, 1), cross_.distance_sq, 1e-6);
    const corners = segmentsNearest(.init(0, 0), .init(1, 0), .init(2, 1), .init(3, 2));
    try testing.expectEqual(@as(f32, 1), corners.fraction1);
    try testing.expectEqual(@as(f32, 0), corners.fraction2);
}
