// SPDX-License-Identifier: BSD-2-Clause

//! Continuous collision: where along its path a fast shape first touched
//! something, and whether that touch is one a step would have got wrong.
//!
//! ```zig
//! const path: continuous.Sweep = .{ .local_center = .zero, .c0 = .init(0, 0), .a0 = 0, .c1 = .init(10, 0), .a1 = 0 };
//! switch (continuous.timeOfImpact(&bullet, path, &wall, wall_xf, 1, slop, 0.25 * slop)) {
//!     .hit => |t| ..., // first within `slop` of the wall `t` of the way along
//!     .touching, .miss => {},
//! }
//! ```
//!
//! **Why a step needs it.** A step finds what touches from where the bodies
//! are, and then moves them. A ball five centimetres across at thirty
//! metres a second covers half a metre a step, so a wall ten centimetres
//! thick can lie wholly between where it is and where it will be: touching
//! nothing before, nothing after, and through. So the world sweeps every
//! body that moved more than half its thinnest extent in a step along the
//! path it took, and if the sweep says it went into something, puts it back
//! where it first touched. Its velocity is kept; the next step looks a
//! little ahead of it (`Body.stopped_short`), and the solver stops it or
//! bounces it as it would anything it touches. What it loses is the rest of
//! one step's travel, which nobody sees. Box2D v3 does the same.
//!
//! **How the first touch is found: conservative advancement.** At any pose,
//! how far apart two convex shapes are, and along which direction, is known
//! (`separation`). Nothing on a shape can close that gap faster than its
//! centre moves along that direction plus its spin times its reach. So the
//! shape may safely be moved on by the gap over that speed, and again from
//! there, each move landing short of touching and never past it, until the
//! gap is within a tolerance of the target. A wall of any thinness cannot
//! be skipped, because no move is ever longer than the gap in front of it.
//! Brian Mirtich's method, from his thesis; Erin Catto's talk on continuous
//! collision at GDC 2013 starts from it.
//!
//! **And whether it matters.** A box sliding along a floor of tiles comes
//! within touching of the next tile's corner at every seam - the tile's side
//! is a wall, as far as the tile alone can tell - and stopping it there
//! would stop a fast box at every seam. So a first touch counts only if the
//! rest of the path would sink the shape deeper than a graze into what it
//! touched (`sinksPast`), or bring the shape's core - a small circle deep
//! inside it - to it (`core`). A box sliding onto the next tile does
//! neither; a box into a wall or a kerb, or a thin rod into a thinner
//! plate, does one or the other. A shape already touching something when
//! the step began is swept by its core alone, Box2D's fallback, which is
//! what lets a box slide fast along a floor without being stopped by it.
//!
//! Nothing here allocates or reads anything but its arguments, so every
//! fast body can be swept at once on every core.

const std = @import("std");
const testing = std.testing;

const geometry = @import("geometry.zig");
const Vec2 = geometry.Vec2;
const Rot = geometry.Rot;
const Transform = geometry.Transform;
const shape = @import("shape.zig");
const Geometry = shape.Geometry;
const Circle = shape.Circle;
const Polygon = shape.Polygon;

const eps: f32 = std.math.floatEps(f32);
const huge: f32 = std.math.floatMax(f32);

/// A body's path through one step: its centre of mass in a straight line
/// and its angle at an even rate. Not quite how the substeps moved it, and
/// near enough to find where it first touched.
pub const Sweep = struct {
    /// The centre of mass in the body's own frame.
    local_center: Vec2,
    /// The centre of mass and the angle where the step began, and where it
    /// ended.
    c0: Vec2,
    a0: f32,
    c1: Vec2,
    a1: f32,

    /// Where the body is `t` of the way along, from zero to one.
    pub fn at(self: Sweep, t: f32) Transform {
        const q: Rot = .fromAngle(self.a0 + t * (self.a1 - self.a0));
        const c = self.c0.lerp(self.c1, t);
        return .{ .p = c.sub(q.rotate(self.local_center)), .q = q };
    }
};

// -------------------------------------------------------------------------
// How far apart
// -------------------------------------------------------------------------

/// How far apart two shapes are, and in which direction.
pub const Separation = struct {
    /// The gap between them. Negative is how deep they overlap.
    distance: f32,
    /// Unit, from A towards B: the direction the gap is measured along, or
    /// when they overlap, the shallowest way out.
    normal: Vec2,
};

/// The gap between two shapes: exact while they are apart, and while they
/// overlap, the depth along the shallowest way out - the separating-axis
/// answer, the narrow phase's too.
pub fn separation(a: *const Geometry, xa: Transform, b: *const Geometry, xb: Transform) Separation {
    // A capsule is its core with its radius round it: the gap to its core,
    // less the radius.
    switch (a.*) {
        .capsule => |c| {
            const inner: Geometry = .{ .polygon = c.core() };
            var s = separation(&inner, xa, b, xb);
            s.distance -= c.radius;
            return s;
        },
        else => {},
    }
    switch (b.*) {
        .capsule => |c| {
            const inner: Geometry = .{ .polygon = c.core() };
            var s = separation(a, xa, &inner, xb);
            s.distance -= c.radius;
            return s;
        },
        else => {},
    }
    return switch (a.*) {
        .circle => |ca| switch (b.*) {
            .circle => |cb| circles(ca, xa, cb, xb),
            .polygon => |*pb| flipped(polygonCircle(pb, xb, ca, xa)),
            .capsule => unreachable,
        },
        .polygon => |*pa| switch (b.*) {
            .circle => |cb| polygonCircle(pa, xa, cb, xb),
            .polygon => |*pb| polygons(pa, xa, pb, xb),
            .capsule => unreachable,
        },
        .capsule => unreachable,
    };
}

fn flipped(s: Separation) Separation {
    return .{ .distance = s.distance, .normal = s.normal.neg() };
}

fn circles(a: Circle, xa: Transform, b: Circle, xb: Transform) Separation {
    const d = xb.apply(b.center).sub(xa.apply(a.center));
    const len = d.len();
    return .{
        .distance = len - a.radius - b.radius,
        // Two centres on one point have no direction between them; any
        // will do, and one is better than NaN.
        .normal = if (len > eps) d.scale(1 / len) else .unit_y,
    };
}

/// A polygon, as A, and a circle.
fn polygonCircle(poly: *const Polygon, xa: Transform, circle: Circle, xb: Transform) Separation {
    // In the polygon's frame, where its normals are the stored ones.
    const c = xa.unapply(xb.apply(circle.center));
    var outside: f32 = -huge;
    var face: usize = 0;
    for (poly.vertexSlice(), poly.normalSlice(), 0..) |v, n, i| {
        const s = n.dot(c.sub(v));
        if (s > outside) {
            outside = s;
            face = i;
        }
    }
    // The centre is inside: out through the nearest face.
    if (outside <= 0) return .{ .distance = outside - circle.radius, .normal = xa.q.rotate(poly.normals[face]) };
    // Outside, it is at least `outside` from the nearest point of the
    // outline, so the division is by something that is not zero.
    const d = c.sub(nearestOnLoop(poly.vertexSlice(), c));
    const len = d.len();
    return .{ .distance = len - circle.radius, .normal = xa.q.rotate(d.scale(1 / len)) };
}

/// Two polygons. Overlapping, the separating-axis depth. Apart, the exact
/// gap: the nearest two points of two convex polygons are always a corner
/// of one and a point on an edge of the other, and with eight corners at
/// most, every such pairing can simply be asked.
fn polygons(a: *const Polygon, xa: Transform, b: *const Polygon, xb: Transform) Separation {
    // Everything in A's frame, B's corners and normals brought over once.
    const xf = xa.invMul(xb);
    var corners_b: [Polygon.max_vertices]Vec2 = undefined;
    var normals_b: [Polygon.max_vertices]Vec2 = undefined;
    for (0..b.count) |j| {
        corners_b[j] = xf.apply(b.vertices[j]);
        normals_b[j] = xf.q.rotate(b.normals[j]);
    }
    const vb = corners_b[0..b.count];

    // The separating-axis test, from both sides. A's normals point from A
    // towards B; B's from B towards A, so they are turned round.
    var best: f32 = -huge;
    var axis: Vec2 = .unit_y;
    for (a.vertexSlice(), a.normalSlice()) |v, n| {
        var s = huge;
        for (vb) |w| s = @min(s, n.dot(w.sub(v)));
        if (s > best) {
            best = s;
            axis = n;
        }
    }
    for (vb, normals_b[0..b.count]) |w, n| {
        var s = huge;
        for (a.vertexSlice()) |v| s = @min(s, n.dot(v.sub(w)));
        if (s > best) {
            best = s;
            axis = n.neg();
        }
    }
    if (best <= 0) return .{ .distance = best, .normal = xa.q.rotate(axis) };

    // Apart. Every corner of each against the outline of the other.
    var nearest_sq = huge;
    var from: Vec2 = .zero;
    var to: Vec2 = .zero;
    for (vb) |w| {
        const q = nearestOnLoop(a.vertexSlice(), w);
        const d_sq = q.distSq(w);
        if (d_sq < nearest_sq) {
            nearest_sq = d_sq;
            from = q;
            to = w;
        }
    }
    for (a.vertexSlice()) |v| {
        const q = nearestOnLoop(vb, v);
        const d_sq = q.distSq(v);
        if (d_sq < nearest_sq) {
            nearest_sq = d_sq;
            from = v;
            to = q;
        }
    }
    // At least `best` apart, so again not a division by zero.
    const len = @sqrt(nearest_sq);
    return .{ .distance = len, .normal = xa.q.rotate(to.sub(from).scale(1 / len)) };
}

/// The point on the closed outline through `corners` nearest `p`. Every
/// edge is asked - eight at most - which is simpler than working out which
/// region `p` is in, and as exact.
fn nearestOnLoop(corners: []const Vec2, p: Vec2) Vec2 {
    var best = corners[0];
    var best_sq = huge;
    for (corners, 0..) |c, i| {
        const q = nearestOnSegment(p, c, corners[(i + 1) % corners.len]);
        const d_sq = q.distSq(p);
        if (d_sq < best_sq) {
            best_sq = d_sq;
            best = q;
        }
    }
    return best;
}

fn nearestOnSegment(p: Vec2, a: Vec2, b: Vec2) Vec2 {
    const e = b.sub(a);
    const len_sq = e.lenSq();
    if (len_sq <= 0) return a;
    return a.mulAdd(e, std.math.clamp(p.sub(a).dot(e) / len_sq, 0, 1));
}

// -------------------------------------------------------------------------
// When it first touched
// -------------------------------------------------------------------------

/// What `timeOfImpact` found.
pub const Impact = union(enum) {
    /// Never within the target, before the time asked about.
    miss,
    /// Within it already at the start.
    touching,
    /// First within it this far along, from zero to one.
    hit: f32,
};

/// Past this many moves the search stops where it has safely got to. A
/// shape spinning fast past a corner closes in slowly, and a pose a little
/// short of touching is still a safe one to be put back to. Box2D's number.
const max_iterations = 20;

/// When shape `a`, carried along `path`, first comes within `target` of
/// shape `b`, standing still at `xb` - if it does before `t_max`. Stops
/// within `tolerance` beyond the target, never inside it.
pub fn timeOfImpact(a: *const Geometry, path: Sweep, b: *const Geometry, xb: Transform, t_max: f32, target: f32, tolerance: f32) Impact {
    const travel = path.c1.sub(path.c0);
    // The furthest any point of `a` can be carried by its turning, over the
    // whole step.
    const turn = @abs(path.a1 - path.a0) * a.reach(path.local_center);
    var t: f32 = 0;
    for (0..max_iterations) |i| {
        const s = separation(a, path.at(t), b, xb);
        if (s.distance < target + tolerance) return if (i == 0) .touching else .{ .hit = t };
        // The fastest the gap along `normal` can close, per whole step: the
        // centre's approach along it, and the most any spin can add. Not
        // closing at all, it never will: the gap along this one direction
        // is a floor under the real one.
        const closing = travel.dot(s.normal) + turn;
        if (closing <= 0) return .miss;
        t += (s.distance - target) / closing;
        if (t >= t_max) return .miss;
    }
    return .{ .hit = t };
}

/// Whether shape `a`, carried along `path` from `from` to the end of the
/// step, sinks into `b` deeper than `depth` anywhere on the way.
///
/// The gap along the way is a convex function of time for a shape that
/// does not turn - it is the distance from a point moving along a line to
/// a convex set - and nearly so for one that turns a little. So its lowest
/// point is found by golden-section search, which needs nothing more:
/// sixteen rounds narrow the stretch to under a two-thousandth of itself.
pub fn sinksPast(a: *const Geometry, path: Sweep, b: *const Geometry, xb: Transform, from: f32, depth: f32) bool {
    const floor = -depth;
    const gap = struct {
        fn at(g: *const Geometry, p: Sweep, other: *const Geometry, x: Transform, t: f32) f32 {
            return separation(g, p.at(t), other, x).distance;
        }
    }.at;
    if (gap(a, path, b, xb, 1) < floor) return true;

    // The golden ratio's reciprocal: each round keeps this much of the
    // stretch, and one of its two inner points is the next round's.
    const keep: f32 = 0.618034;
    var lo = from;
    var hi: f32 = 1;
    var m1 = hi - keep * (hi - lo);
    var m2 = lo + keep * (hi - lo);
    var g1 = gap(a, path, b, xb, m1);
    var g2 = gap(a, path, b, xb, m2);
    for (0..16) |_| {
        if (@min(g1, g2) < floor) return true;
        if (g1 < g2) {
            hi = m2;
            m2 = m1;
            g2 = g1;
            m1 = hi - keep * (hi - lo);
            g1 = gap(a, path, b, xb, m1);
        } else {
            lo = m1;
            m1 = m2;
            g1 = g2;
            m2 = lo + keep * (hi - lo);
            g2 = gap(a, path, b, xb, m2);
        }
    }
    return @min(g1, g2) < floor;
}

/// A shape's core: a circle at its centroid whose radius is a quarter of
/// how thin the shape is. A shape that is only brushing something cannot
/// bring its core to it; one whose core reaches a wall is going through.
pub fn core(g: *const Geometry) Geometry {
    return .{ .circle = .{ .center = g.centroid(), .radius = 0.25 * g.minExtent() } };
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const test_slop: f32 = 0.005;
const test_tolerance: f32 = 0.25 * test_slop;

fn boxAt(half_width: f32, half_height: f32) Geometry {
    return .{ .polygon = .box(half_width, half_height) };
}

fn straight(from: Vec2, to: Vec2) Sweep {
    return .{ .local_center = .zero, .c0 = from, .a0 = 0, .c1 = to, .a1 = 0 };
}

test "the gap between shapes is exact while apart, and the depth while in" {
    const ball: Geometry = .{ .circle = .{ .radius = 0.5 } };
    const box = boxAt(1, 1);

    // Two balls, a quarter apart.
    const two = separation(&ball, .identity, &ball, .init(.init(1.25, 0), 0));
    try testing.expectApproxEqAbs(@as(f32, 0.25), two.distance, 1e-6);
    try testing.expect(two.normal.approxEql(.unit_x));

    // A ball off a box's face, and off its corner, where the nearest point
    // is the corner and not the face: sqrt(2) * 0.5 from it.
    const off_face = separation(&box, .identity, &ball, .init(.init(0, 1.75), 0));
    try testing.expectApproxEqAbs(@as(f32, 0.25), off_face.distance, 1e-6);
    try testing.expect(off_face.normal.approxEql(.unit_y));
    const off_corner = separation(&box, .identity, &ball, .init(.init(1.5, 1.5), 0));
    try testing.expectApproxEqAbs(@as(f32, @sqrt(0.5) - 0.5), off_corner.distance, 1e-6);
    // And the other way round, pointing the other way.
    const back = separation(&ball, .init(.init(1.5, 1.5), 0), &box, .identity);
    try testing.expectApproxEqAbs(off_corner.distance, back.distance, 1e-6);
    try testing.expect(back.normal.approxEql(off_corner.normal.neg()));

    // Two boxes corner to corner: the corners are the nearest points, which
    // the separating-axis test alone would put only 0.5 apart.
    const corner_to_corner = separation(&box, .identity, &box, .init(.init(2.5, 2.5), 0));
    try testing.expectApproxEqAbs(@as(f32, @sqrt(0.5)), corner_to_corner.distance, 1e-5);

    // Overlapping: how deep, along the shallowest way out.
    const sunk = separation(&box, .identity, &box, .init(.init(1.9, 0.5), 0));
    try testing.expectApproxEqAbs(@as(f32, -0.1), sunk.distance, 1e-5);
    try testing.expect(sunk.normal.approxEql(.unit_x));
}

test "a ball fired through a thin wall first touches it where the arithmetic says" {
    // A ball of radius 0.1 from x = 0 to x = 10 in one step, through a wall
    // ten centimetres thick at x = 5. Its near face is at 4.95, so the ball
    // touches at a centre of 4.85: 0.485 of the way. Neither end of the step
    // is anywhere near the wall.
    const ball: Geometry = .{ .circle = .{ .radius = 0.1 } };
    const wall = boxAt(0.05, 5);
    const wall_xf: Transform = .init(.init(5, 0), 0);
    const path = straight(.zero, .init(10, 0));
    const t = switch (timeOfImpact(&ball, path, &wall, wall_xf, 1, test_slop, test_tolerance)) {
        .hit => |t| t,
        else => return error.TestUnexpectedResult,
    };
    try testing.expect(t > (4.85 - test_slop - test_tolerance) / 10 and t < (4.85 - test_slop) / 10 + 1e-6);

    // Short of the target, never inside it.
    const gap = separation(&ball, path.at(t), &wall, wall_xf).distance;
    try testing.expect(gap >= test_slop - 1e-5 and gap < test_slop + test_tolerance);

    // A limit before the wall is a miss.
    try testing.expectEqual(Impact.miss, timeOfImpact(&ball, path, &wall, wall_xf, 0.4, test_slop, test_tolerance));
}

test "a spinning rod is caught by its tip, and nothing is caught moving away or already touching" {
    // Half a metre long, two centimetres thick, turning three radians while
    // it crosses to a wall; its tip reaches the wall before its middle does.
    const rod = boxAt(0.25, 0.01);
    const wall = boxAt(0.05, 5);
    const wall_xf: Transform = .init(.init(2, 0), 0);
    const path: Sweep = .{ .local_center = .zero, .c0 = .zero, .a0 = 0, .c1 = .init(3, 0), .a1 = 3 };
    const t = switch (timeOfImpact(&rod, path, &wall, wall_xf, 1, test_slop, test_tolerance)) {
        .hit => |t| t,
        else => return error.TestUnexpectedResult,
    };
    const gap = separation(&rod, path.at(t), &wall, wall_xf).distance;
    try testing.expect(gap >= test_slop - 1e-5 and gap < test_slop + test_tolerance);
    // Its middle stopped short of the wall by more than the rod is thick,
    // and by no more than it reaches.
    const reach = rod.reach(.zero);
    try testing.expect(path.at(t).p.x < 1.95 - test_slop - 0.01 + 1e-4);
    try testing.expect(path.at(t).p.x > 1.95 - reach - test_slop - test_tolerance);

    const ball: Geometry = .{ .circle = .{ .radius = 0.1 } };
    // Away from the wall, and alongside it.
    try testing.expectEqual(Impact.miss, timeOfImpact(&ball, straight(.init(1, 0), .init(-5, 0)), &wall, wall_xf, 1, test_slop, test_tolerance));
    try testing.expectEqual(Impact.miss, timeOfImpact(&ball, straight(.init(1, -6), .init(1, 6)), &wall, wall_xf, 1, test_slop, test_tolerance));
    // Resting on a floor and sliding along it: touching from the start.
    const floor = boxAt(10, 0.5);
    const along = straight(.init(0, -0.6), .init(5, -0.6));
    try testing.expectEqual(Impact.touching, timeOfImpact(&ball, along, &floor, .identity, 1, test_slop, test_tolerance));
}

test "sliding onto the next tile is a graze; into a kerb or through a plate is not" {
    // A box resting half a test_slop into a tile whose top is at y = 0, sliding
    // over the seam at x = 0.5 onto the next tile along. It touches that
    // tile's corner - and then only ever sits half a test_slop into it.
    const crate = boxAt(0.2, 0.2);
    const tile = boxAt(0.25, 0.25);
    const next_tile: Transform = .init(.init(0.75, 0.25), 0);
    const resting_y = -0.2 + 0.5 * test_slop;
    const slide = straight(.init(0, resting_y), .init(1.5, resting_y));
    const t = switch (timeOfImpact(&crate, slide, &tile, next_tile, 1, test_slop, test_tolerance)) {
        .hit => |t| t,
        else => return error.TestUnexpectedResult,
    };
    try testing.expect(!sinksPast(&crate, slide, &tile, next_tile, t, 4 * test_slop));
    // Nor does its core come near the tile.
    const crate_core = core(&crate);
    try testing.expectEqual(Impact.miss, timeOfImpact(&crate_core, slide, &tile, next_tile, 1, test_slop, test_tolerance));

    // A kerb ten centimetres high at the same place is hit, and would be
    // sunk into by as much as it stands up.
    const kerb = boxAt(0.25, 0.05);
    const kerb_xf: Transform = .init(.init(0.75, -0.05), 0);
    const at_kerb = switch (timeOfImpact(&crate, slide, &kerb, kerb_xf, 1, test_slop, test_tolerance)) {
        .hit => |k| k,
        else => return error.TestUnexpectedResult,
    };
    try testing.expect(sinksPast(&crate, slide, &kerb, kerb_xf, at_kerb, 4 * test_slop));

    // A rod two centimetres thick thrown flat through a plate a millimetre
    // thick is never more than a centimetre from out of it - half its own
    // thickness - so it never sinks past a graze. Its core goes clean
    // through. (End-first, it would: the way out is back along its length.)
    const rod = boxAt(0.25, 0.01);
    const plate = boxAt(1, 0.0005);
    const plate_xf: Transform = .init(.init(0, 1), 0);
    const flat = straight(.zero, .init(0, 2));
    const at_plate = switch (timeOfImpact(&rod, flat, &plate, plate_xf, 1, test_slop, test_tolerance)) {
        .hit => |p| p,
        else => return error.TestUnexpectedResult,
    };
    try testing.expect(!sinksPast(&rod, flat, &plate, plate_xf, at_plate, 4 * test_slop));
    const rod_core = core(&rod);
    // Comparing a tagged union with an enum literal compares its tag.
    try testing.expect(timeOfImpact(&rod_core, flat, &plate, plate_xf, 1, test_slop, test_tolerance) == .hit);
    const end_first = straight(.zero, .init(2, 0));
    const side_plate: Transform = .init(.init(1, 0), std.math.pi / 2.0);
    const at_side = switch (timeOfImpact(&rod, end_first, &plate, side_plate, 1, test_slop, test_tolerance)) {
        .hit => |p| p,
        else => return error.TestUnexpectedResult,
    };
    try testing.expect(sinksPast(&rod, end_first, &plate, side_plate, at_side, 4 * test_slop));
}
