// SPDX-License-Identifier: BSD-2-Clause

//! What a body is made of: circles and convex polygons, what they weigh, and
//! what they will and will not touch.
//!
//! ```zig
//! const ball: Shape = .circle(0.5);
//! const crate: Shape = .box(1, 1);
//! const wedge: Shape = .{ .geometry = .{ .polygon = try .fromPoints(&.{ a, b, c }) } };
//! ```
//!
//! **Two geometries, and the list is short on purpose.** A circle and a
//! convex polygon of up to eight vertices cover crates, balls, ramps, wheels
//! and characters, and any concave outline is several convex ones on one
//! body. Each geometry pairs with each in the narrow phase, so every one
//! added is a row and a column of pairings to write and to keep correct;
//! capsules and rounded polygons are the next two and they are not here yet.
//!
//! **A polygon is always convex and always wound the same way**, because
//! `fromPoints` builds the convex hull of whatever it is given and the
//! collision arithmetic assumes it. Hand it the corners in any order, or
//! more corners than the hull needs; hand it eight that make a star and the
//! star's outer points come back. What cannot be built is refused rather
//! than guessed at.
//!
//! **Mass comes from density and area**, so a big crate is heavier than a
//! small one without anyone typing a mass. A body adds up what its shapes
//! weigh - see `World.addShape`.

const std = @import("std");
const testing = std.testing;

const geometry = @import("geometry.zig");
const Vec2 = geometry.Vec2;
const Transform = geometry.Transform;
const Aabb = geometry.Aabb;
const cross = geometry.cross;

/// What a shape contributes to its body: how heavy, where its weight sits,
/// and how hard it is to spin about the body's origin.
pub const MassData = struct {
    mass: f32,
    /// Where the mass is, in the body's frame.
    center: Vec2,
    /// Rotational inertia about the body's *origin*, not about `center`.
    /// About the origin because that is what sums across shapes; the body
    /// moves it to its centre of mass afterwards.
    inertia: f32,
};

/// A disc, offset from the body's origin.
pub const Circle = extern struct {
    center: Vec2 = .zero,
    radius: f32,

    pub fn aabb(self: Circle, xf: Transform) Aabb {
        return .fromCenter(xf.apply(self.center), .splat(self.radius));
    }

    pub fn massData(self: Circle, density: f32) MassData {
        const r2 = self.radius * self.radius;
        const mass = density * std.math.pi * r2;
        return .{
            .mass = mass,
            .center = self.center,
            // A disc about its own centre is m r^2 / 2; about the origin, add
            // the parallel axis term.
            .inertia = mass * (0.5 * r2 + self.center.lenSq()),
        };
    }
};

/// A convex polygon, wound so that its signed area is positive, with an
/// outward unit normal per edge kept beside the vertices.
///
/// The normals are stored rather than recomputed because the narrow phase
/// reads each one several times a step and the polygon never changes.
pub const Polygon = extern struct {
    vertices: [max_vertices]Vec2,
    normals: [max_vertices]Vec2,
    count: u32,

    /// Enough for a hexagon and a chamfered box. Eight because the arrays are
    /// inline - a polygon owns nothing and is copied by value - and beyond
    /// eight a shape is a mesh of several.
    pub const max_vertices = 8;

    pub const Error = error{
        /// Fewer than three distinct points, or all of them on one line.
        Degenerate,
        /// The convex hull has more than `max_vertices` corners.
        TooManyVertices,
    };

    /// A rectangle centred on the origin.
    pub fn box(half_width: f32, half_height: f32) Polygon {
        return offsetBox(half_width, half_height, .zero, 0);
    }

    /// A rectangle somewhere else on the body, turned by `radians`.
    pub fn offsetBox(half_width: f32, half_height: f32, center: Vec2, radians: f32) Polygon {
        const xf: Transform = .init(center, radians);
        var p: Polygon = .{ .vertices = undefined, .normals = undefined, .count = 4 };
        p.vertices[0] = xf.apply(.init(-half_width, -half_height));
        p.vertices[1] = xf.apply(.init(half_width, -half_height));
        p.vertices[2] = xf.apply(.init(half_width, half_height));
        p.vertices[3] = xf.apply(.init(-half_width, half_height));
        p.normals[0] = xf.q.rotate(.init(0, -1));
        p.normals[1] = xf.q.rotate(.init(1, 0));
        p.normals[2] = xf.q.rotate(.init(0, 1));
        p.normals[3] = xf.q.rotate(.init(-1, 0));
        return p;
    }

    /// The convex hull of `points`, in any order and any number up to what
    /// the hull needs.
    ///
    /// Andrew's monotone chain: sort by x, walk the lower hull and then the
    /// upper, dropping every point that would make a right turn. Collinear
    /// points are dropped too, because an edge of zero length has no normal
    /// and the separating-axis test divides by one.
    pub fn fromPoints(points: []const Vec2) Error!Polygon {
        var sorted: [max_hull_input]Vec2 = undefined;
        if (points.len > max_hull_input) return error.TooManyVertices;
        if (points.len < 3) return error.Degenerate;
        @memcpy(sorted[0..points.len], points);
        const input = sorted[0..points.len];
        std.sort.pdq(Vec2, input, {}, lexLess);

        var hull: [2 * max_hull_input]Vec2 = undefined;
        var n: usize = 0;
        // Lower hull.
        for (input) |p| {
            while (n >= 2 and cross(hull[n - 1].sub(hull[n - 2]), p.sub(hull[n - 2])) <= 0) n -= 1;
            hull[n] = p;
            n += 1;
        }
        // Upper hull, walking back; the last point of the lower hull is the
        // first of the upper and must not be popped.
        const lower_len = n + 1;
        var i = input.len - 1;
        while (i > 0) : (i -= 1) {
            const p = input[i - 1];
            while (n >= lower_len and cross(hull[n - 1].sub(hull[n - 2]), p.sub(hull[n - 2])) <= 0) n -= 1;
            hull[n] = p;
            n += 1;
        }
        // The first point is now also the last.
        n -= 1;

        if (n < 3) return error.Degenerate;
        if (n > max_vertices) return error.TooManyVertices;

        var poly: Polygon = .{ .vertices = undefined, .normals = undefined, .count = @intCast(n) };
        @memcpy(poly.vertices[0..n], hull[0..n]);
        for (0..n) |k| {
            const edge = poly.vertices[(k + 1) % n].sub(poly.vertices[k]);
            const len_sq = edge.lenSq();
            if (len_sq < 1e-12) return error.Degenerate;
            // The outward normal of an edge on a positively wound polygon is
            // the edge turned a quarter turn clockwise.
            poly.normals[k] = geometry.crossVS(edge, 1).scale(1 / @sqrt(len_sq));
        }
        return poly;
    }

    /// More input than output, so a rounded-off outline can be handed over
    /// and reduced. Sixty-four is a generous polyline.
    const max_hull_input = 64;

    fn lexLess(_: void, a: Vec2, b: Vec2) bool {
        return a.x < b.x or (a.x == b.x and a.y < b.y);
    }

    pub fn vertexSlice(self: *const Polygon) []const Vec2 {
        return self.vertices[0..self.count];
    }

    pub fn normalSlice(self: *const Polygon) []const Vec2 {
        return self.normals[0..self.count];
    }

    pub fn aabb(self: *const Polygon, xf: Transform) Aabb {
        var lo = xf.apply(self.vertices[0]);
        var hi = lo;
        for (self.vertices[1..self.count]) |v| {
            const w = xf.apply(v);
            lo = lo.min(w);
            hi = hi.max(w);
        }
        return .{ .min = lo, .max = hi };
    }

    /// Whether a point in the body's frame is inside.
    pub fn containsLocal(self: *const Polygon, p: Vec2) bool {
        for (self.vertexSlice(), self.normalSlice()) |v, n| {
            if (n.dot(p.sub(v)) > 0) return false;
        }
        return true;
    }

    /// The area-weighted centre and the inertia, by fanning triangles out
    /// from a point inside.
    ///
    /// Box2D's arithmetic. The reference point is the mean of the vertices
    /// rather than the origin because the origin may be far away, and a
    /// triangle fan from far away sums large numbers that nearly cancel.
    pub fn massData(self: *const Polygon, density: f32) MassData {
        const n = self.count;
        var s: Vec2 = .zero;
        for (self.vertexSlice()) |v| s = s.add(v);
        s = s.scale(1 / @as(f32, @floatFromInt(n)));

        var area: f32 = 0;
        var center: Vec2 = .zero;
        var inertia: f32 = 0;
        const third: f32 = 1.0 / 3.0;

        for (0..n) |i| {
            const e1 = self.vertices[i].sub(s);
            const e2 = self.vertices[(i + 1) % n].sub(s);
            const d = cross(e1, e2);
            const tri_area = 0.5 * d;
            area += tri_area;
            center = center.add(e1.add(e2).scale(tri_area * third));

            const intx2 = e1.x * e1.x + e2.x * e1.x + e2.x * e2.x;
            const inty2 = e1.y * e1.y + e2.y * e1.y + e2.y * e2.y;
            inertia += (0.25 * third * d) * (intx2 + inty2);
        }

        const mass = density * area;
        std.debug.assert(area > std.math.floatEps(f32));
        center = center.scale(1 / area);
        const centroid = center.add(s);

        // `inertia` is about `s`; move it to the origin through the centroid.
        const about_origin = density * inertia + mass * (centroid.lenSq() - center.lenSq());
        return .{ .mass = mass, .center = centroid, .inertia = about_origin };
    }
};

/// The two kinds of geometry a shape can be.
pub const Geometry = union(enum) {
    circle: Circle,
    polygon: Polygon,

    pub fn aabb(self: *const Geometry, xf: Transform) Aabb {
        return switch (self.*) {
            .circle => |c| c.aabb(xf),
            .polygon => |*p| p.aabb(xf),
        };
    }

    pub fn massData(self: *const Geometry, density: f32) MassData {
        return switch (self.*) {
            .circle => |c| c.massData(density),
            .polygon => |*p| p.massData(density),
        };
    }

    /// The middle of its area, in the body's frame.
    pub fn centroid(self: *const Geometry) Vec2 {
        return switch (self.*) {
            .circle => |c| c.center,
            .polygon => |*p| p.massData(1).center,
        };
    }

    /// How far it reaches from `from`, a point in the body's frame: the
    /// radius of the circle about that point that it fits in.
    pub fn reach(self: *const Geometry, from: Vec2) f32 {
        return switch (self.*) {
            .circle => |c| c.center.dist(from) + c.radius,
            .polygon => |*p| blk: {
                var far: f32 = 0;
                for (p.vertexSlice()) |v| far = @max(far, v.dist(from));
                break :blk far;
            },
        };
    }

    /// How thin it is: the least distance from its centroid to its edge.
    /// A shape that moves more than about half of this in one step can pass
    /// through something before a step that only looks at where it ends up
    /// sees it. See `continuous`.
    pub fn minExtent(self: *const Geometry) f32 {
        return switch (self.*) {
            .circle => |c| c.radius,
            .polygon => |*p| blk: {
                const middle = p.massData(1).center;
                var least = std.math.floatMax(f32);
                for (p.vertexSlice(), p.normalSlice()) |v, n| least = @min(least, n.dot(v.sub(middle)));
                break :blk least;
            },
        };
    }
};

/// How a surface behaves when touched.
pub const Material = struct {
    /// Coulomb friction. Two surfaces' values are combined by geometric
    /// mean, so ice against anything is slippery.
    friction: f32 = 0.6,
    /// How much of the approach speed comes back. Zero is a beanbag, one a
    /// superball; the pair's larger value wins.
    restitution: f32 = 0,
    /// Mass per unit area. Zero is allowed and weighs nothing; a body all of
    /// whose shapes weigh nothing gets a mass of one so it can still move.
    density: f32 = 1,
};

/// Who touches whom. Sixteen categories and a group, which is Box2D's
/// scheme and enough for every game that has used it.
pub const Filter = struct {
    /// What this shape is. One bit, usually.
    category: u16 = 1,
    /// What this shape touches. Every bit, usually.
    mask: u16 = 0xFFFF,
    /// Shapes in the same positive group always touch; in the same negative
    /// group never do; zero is no group. Overrides the bits, which is what
    /// makes it worth having: the pieces of one ragdoll never touch each
    /// other, whatever their categories say.
    group: i16 = 0,

    pub fn shouldCollide(a: Filter, b: Filter) bool {
        if (a.group == b.group and a.group != 0) return a.group > 0;
        return (a.mask & b.category) != 0 and (a.category & b.mask) != 0;
    }

    /// Whether a sensor and another shape see each other: one side's mask
    /// having the other's category is enough, so a hitbox that asks for
    /// nothing is still seen by the hurtbox that asks for hitboxes. Groups
    /// as in `shouldCollide`. Which of the two cares is the caller's to say,
    /// by its own mask.
    pub fn shouldSense(a: Filter, b: Filter) bool {
        if (a.group == b.group and a.group != 0) return a.group > 0;
        return (a.mask & b.category) != 0 or (a.category & b.mask) != 0;
    }
};

test "a sensor's pair needs one side to ask for the other, a collision both" {
    const hitbox: Filter = .{ .category = 2, .mask = 0 };
    const hurtbox: Filter = .{ .category = 0, .mask = 2 };
    try std.testing.expect(hitbox.shouldSense(hurtbox) and hurtbox.shouldSense(hitbox));
    try std.testing.expect(!hitbox.shouldCollide(hurtbox));

    const blind: Filter = .{ .category = 4, .mask = 0 };
    try std.testing.expect(!hitbox.shouldSense(blind));

    // A negative group still keeps its own apart, whatever the bits ask.
    const a: Filter = .{ .category = 1, .mask = 1, .group = -3 };
    try std.testing.expect(!a.shouldSense(a));
}

/// A shape as a game describes it: geometry, surface, and who it touches.
/// What `World.addShape` takes and keeps.
pub const Shape = struct {
    geometry: Geometry,
    material: Material = .{},
    filter: Filter = .{},
    /// A sensor reports what overlaps it and pushes nothing. A trigger
    /// volume, a pickup radius, the bottom of a pit.
    sensor: bool = false,
    /// Yours. A shape never reads it.
    user_data: u64 = 0,

    pub fn circle(radius: f32) Shape {
        return .{ .geometry = .{ .circle = .{ .radius = radius } } };
    }

    pub fn box(half_width: f32, half_height: f32) Shape {
        return .{ .geometry = .{ .polygon = .box(half_width, half_height) } };
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

test "a box is wound positively with outward normals" {
    const b: Polygon = .box(2, 1);
    try testing.expectEqual(@as(u32, 4), b.count);
    var area: f32 = 0;
    for (0..4) |i| {
        area += cross(b.vertices[i], b.vertices[(i + 1) % 4]);
        // Each normal points away from the centre and along its edge's
        // outside.
        try testing.expect(b.normals[i].dot(b.vertices[i]) > 0);
    }
    try testing.expectApproxEqAbs(@as(f32, 16), area, 1e-5);
    try testing.expect(b.containsLocal(.init(1.9, 0.9)));
    try testing.expect(!b.containsLocal(.init(2.1, 0)));
}

test "a hull from shuffled points with one inside comes out the same as a box" {
    const points = [_]Vec2{
        .init(1, 1), .init(-1, -1), .init(0, 0.2), .init(1, -1), .init(-1, 1), .init(0, -1),
    };
    const p = try Polygon.fromPoints(&points);
    try testing.expectEqual(@as(u32, 4), p.count);
    const b: Polygon = .box(1, 1);
    const pm = p.massData(1);
    const bm = b.massData(1);
    try testing.expectApproxEqAbs(bm.mass, pm.mass, 1e-5);
    try testing.expectApproxEqAbs(bm.inertia, pm.inertia, 1e-5);
    try testing.expect(pm.center.approxEql(.zero));

    try testing.expectError(error.Degenerate, Polygon.fromPoints(&.{ .init(0, 0), .init(1, 1), .init(2, 2) }));
    try testing.expectError(error.Degenerate, Polygon.fromPoints(&.{ .init(0, 0), .init(1, 1) }));
}

test "mass and inertia match the textbook" {
    // A rectangle of width w, height h: I = m (w^2 + h^2) / 12 about its
    // centre.
    const b: Polygon = .box(1.5, 0.5);
    const m = b.massData(2);
    try testing.expectApproxEqAbs(@as(f32, 2 * 3 * 1), m.mass, 1e-5);
    try testing.expectApproxEqAbs(m.mass * (9 + 1) / 12.0, m.inertia, 1e-4);

    // Moved off the origin, the inertia gains m d^2.
    const off: Polygon = .offsetBox(1.5, 0.5, .init(3, 4), 0);
    const om = off.massData(2);
    try testing.expectApproxEqAbs(m.mass, om.mass, 1e-5);
    try testing.expect(om.center.approxEql(.init(3, 4)));
    try testing.expectApproxEqAbs(m.inertia + m.mass * 25, om.inertia, 1e-3);

    const c: Circle = .{ .radius = 2 };
    const cm = c.massData(1);
    try testing.expectApproxEqAbs(@as(f32, std.math.pi * 4), cm.mass, 1e-4);
    try testing.expectApproxEqAbs(cm.mass * 2, cm.inertia, 1e-4);
}

test "filters: bits, then groups override them" {
    const a: Filter = .{ .category = 0b01, .mask = 0b10 };
    const b: Filter = .{ .category = 0b10, .mask = 0b01 };
    const c: Filter = .{ .category = 0b10, .mask = 0b10 };
    try testing.expect(a.shouldCollide(b));
    try testing.expect(!a.shouldCollide(c));

    const same_negative: Filter = .{ .group = -3 };
    try testing.expect(!same_negative.shouldCollide(same_negative));
    const same_positive: Filter = .{ .group = 3, .category = 0, .mask = 0 };
    try testing.expect(same_positive.shouldCollide(same_positive));
}

test "a box around a turned polygon holds every corner" {
    const b: Polygon = .box(1, 1);
    const xf: Transform = .init(.init(5, 5), std.math.pi / 4.0);
    const box = b.aabb(xf);
    const r = @sqrt(2.0);
    try testing.expectApproxEqAbs(@as(f32, 5 - r), box.min.x, 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 5 + r), box.max.y, 1e-5);
    for (b.vertexSlice()) |v| try testing.expect(box.grow(1e-5).contains(xf.apply(v)));
}
