// SPDX-License-Identifier: BSD-2-Clause

//! A tree of boxes, for finding what overlaps a box or a ray without
//! looking at everything.
//!
//! ```zig
//! var tree: Tree = .empty;
//! defer tree.deinit(gpa);
//! const proxy = try tree.insert(gpa, tile_box, tile_index);
//! tree.query(player_box, &hits, Hits.add);
//! tree.remove(proxy);
//! ```
//!
//! **What it is for: the level.** A game's level is thousands of shapes that
//! never move - tiles, walls, the ground - and a few hundred that do. The
//! sweep in `broadphase` sorts everything along x and compares neighbours,
//! which is the right tool for things that move and the wrong one for a
//! tile map: every tile in a column has the same x, so every tile is a
//! neighbour of every other in its column, and a step spent most of its
//! time discovering that the level does not collide with itself. Static
//! shapes live here instead, and each moving shape asks the tree what of
//! the level is near it: a walk down a few branches, not along a column.
//!
//! **A bounding volume hierarchy**: every leaf is one shape's box, and every
//! branch holds the box around both its children. Asking what overlaps a
//! box walks down only the branches whose boxes do; a ray walks down only
//! the branches it crosses. Balanced, that is a logarithmic walk - a
//! thousand rays onto seven thousand tiles with three hundred bodies on
//! them went from about a hundred milliseconds to under two.
//!
//! **Built one leaf at a time, and kept balanced** - Erin Catto's dynamic
//! tree from Box2D. A new leaf goes down the tree towards the sibling whose
//! box would grow least by taking it in, measured by *perimeter*: in two
//! dimensions the chance that a random ray or box meets a region is
//! proportional to its perimeter, as it is to surface area in three. Then,
//! walking back up, any branch whose two children differ in height by more
//! than one is rotated, the way an AVL tree rotates, so no order of
//! insertion - a tile map added row by row is the worst - can grow it into
//! a list.
//!
//! Leaves stay where they are when others come and go, so a leaf's index -
//! its *proxy* - is a handle to it until it is removed.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const geometry = @import("geometry.zig");
const Vec2 = geometry.Vec2;
const Aabb = geometry.Aabb;

pub const Tree = @This();

/// No node: an empty tree's root, a leaf's missing children, the end of the
/// free list.
pub const null_node = std.math.maxInt(u32);

nodes: std.ArrayList(Node) = .empty,
root: u32 = null_node,
/// The first of the nodes that are not in use, linked through `parent`.
free: u32 = null_node,
/// How many leaves.
leaf_count: u32 = 0,

pub const empty: Tree = .{};

const Node = struct {
    box: Aabb,
    /// The parent, or while the node is free, the next free node.
    parent: u32,
    child1: u32 = null_node,
    child2: u32 = null_node,
    /// Zero for a leaf, one more than its taller child for a branch, and
    /// minus one while free.
    height: i32 = 0,
    /// What a leaf stands for: the caller's index. Unused by a branch.
    value: u32 = 0,

    fn isLeaf(self: *const Node) bool {
        return self.child1 == null_node;
    }
};

pub fn deinit(self: *Tree, gpa: Allocator) void {
    self.nodes.deinit(gpa);
    self.* = undefined;
}

// -------------------------------------------------------------------------
// Changing it
// -------------------------------------------------------------------------

/// Put a box in, standing for `value`, and hand back its proxy.
pub fn insert(self: *Tree, gpa: Allocator, box: Aabb, value: u32) Allocator.Error!u32 {
    const leaf = try self.allocate(gpa);
    self.nodes.items[leaf] = .{ .box = box, .parent = null_node, .value = value };
    try self.insertLeaf(gpa, leaf);
    self.leaf_count += 1;
    return leaf;
}

/// Take a leaf out. Its proxy means nothing afterwards.
pub fn remove(self: *Tree, proxy: u32) void {
    self.removeLeaf(proxy);
    self.release(proxy);
    self.leaf_count -= 1;
}

/// Give a leaf a new box: out and back in, keeping its proxy.
pub fn move(self: *Tree, gpa: Allocator, proxy: u32, box: Aabb) Allocator.Error!void {
    self.removeLeaf(proxy);
    self.nodes.items[proxy].box = box;
    try self.insertLeaf(gpa, proxy);
}

/// The box a leaf has.
pub fn boxOf(self: *const Tree, proxy: u32) Aabb {
    return self.nodes.items[proxy].box;
}

/// How deep the tree is: zero for one leaf. For tests and the log.
pub fn height(self: *const Tree) i32 {
    if (self.root == null_node) return 0;
    return self.nodes.items[self.root].height;
}

fn allocate(self: *Tree, gpa: Allocator) Allocator.Error!u32 {
    if (self.free != null_node) {
        const index = self.free;
        self.free = self.nodes.items[index].parent;
        return index;
    }
    const index: u32 = @intCast(self.nodes.items.len);
    try self.nodes.append(gpa, .{ .box = undefined, .parent = null_node });
    return index;
}

fn release(self: *Tree, index: u32) void {
    self.nodes.items[index] = .{ .box = undefined, .parent = self.free, .height = -1 };
    self.free = index;
}

/// Perimeter: what inserting a leaf tries to keep small. See the module
/// comment for why perimeter and not area.
fn cost(box: Aabb) f32 {
    const size = box.max.sub(box.min);
    return 2 * (size.x + size.y);
}

fn insertLeaf(self: *Tree, gpa: Allocator, leaf: u32) Allocator.Error!void {
    if (self.root == null_node) {
        self.root = leaf;
        self.nodes.items[leaf].parent = null_node;
        return;
    }

    // Down to the best sibling. At each branch there are three choices:
    // make a new parent for this branch and the leaf here, or go on down
    // into one child or the other - which enlarges this branch by the leaf
    // either way, the cost every choice below inherits.
    const box = self.nodes.items[leaf].box;
    var index = self.root;
    while (!self.nodes.items[index].isLeaf()) {
        const node = &self.nodes.items[index];
        const combined = cost(node.box.join(box));
        const here = 2 * combined;
        const inherited = 2 * (combined - cost(node.box));
        const cost1 = self.descentCost(node.child1, box) + inherited;
        const cost2 = self.descentCost(node.child2, box) + inherited;
        if (here < cost1 and here < cost2) break;
        index = if (cost1 < cost2) node.child1 else node.child2;
    }
    const sibling = index;

    // A new branch where the sibling was, holding the sibling and the leaf.
    // `allocate` may move the nodes, so no pointer into them lives across it.
    const old_parent = self.nodes.items[sibling].parent;
    const branch = try self.allocate(gpa);
    const nodes = self.nodes.items;
    nodes[branch] = .{
        .box = box.join(nodes[sibling].box),
        .parent = old_parent,
        .child1 = sibling,
        .child2 = leaf,
        .height = nodes[sibling].height + 1,
    };
    nodes[sibling].parent = branch;
    nodes[leaf].parent = branch;
    if (old_parent == null_node) {
        self.root = branch;
    } else if (nodes[old_parent].child1 == sibling) {
        nodes[old_parent].child1 = branch;
    } else {
        nodes[old_parent].child2 = branch;
    }

    self.refit(nodes[leaf].parent);
}

/// What going down into `child` would cost the leaf `box`: the whole of the
/// new box for a leaf, which would get a new parent, or only how much it
/// grows for a branch, which would be enlarged.
fn descentCost(self: *const Tree, child: u32, box: Aabb) f32 {
    const node = &self.nodes.items[child];
    const grown = cost(node.box.join(box));
    return if (node.isLeaf()) grown else grown - cost(node.box);
}

fn removeLeaf(self: *Tree, leaf: u32) void {
    if (leaf == self.root) {
        self.root = null_node;
        return;
    }
    const nodes = self.nodes.items;
    const parent = nodes[leaf].parent;
    const grandparent = nodes[parent].parent;
    const sibling = if (nodes[parent].child1 == leaf) nodes[parent].child2 else nodes[parent].child1;

    // The parent goes, and the sibling takes its place.
    if (grandparent == null_node) {
        self.root = sibling;
        nodes[sibling].parent = null_node;
        self.release(parent);
        return;
    }
    if (nodes[grandparent].child1 == parent) nodes[grandparent].child1 = sibling else nodes[grandparent].child2 = sibling;
    nodes[sibling].parent = grandparent;
    self.release(parent);
    self.refit(grandparent);
}

/// From `start` to the root: rebalance each branch, then give it the box
/// and height its children now call for.
fn refit(self: *Tree, start: u32) void {
    var index = start;
    while (index != null_node) {
        index = self.balance(index);
        const nodes = self.nodes.items;
        const a = nodes[index].child1;
        const b = nodes[index].child2;
        nodes[index].height = 1 + @max(nodes[a].height, nodes[b].height);
        nodes[index].box = nodes[a].box.join(nodes[b].box);
        index = nodes[index].parent;
    }
}

/// If `a`'s children differ in height by more than one, lift the taller
/// child into `a`'s place, and hand back whichever node is there now.
///
/// ```text
///         a                 c
///       /   \             /   \
///      b     c    ->     a     f        (c's taller child f stays with c;
///           / \         / \              the shorter, g, goes to a)
///          f   g       b   g
/// ```
fn balance(self: *Tree, a: u32) u32 {
    const nodes = self.nodes.items;
    if (nodes[a].isLeaf() or nodes[a].height < 2) return a;
    const b = nodes[a].child1;
    const c = nodes[a].child2;
    const lean = nodes[c].height - nodes[b].height;
    if (lean > 1) return self.rotate(a, c, .right);
    if (lean < -1) return self.rotate(a, b, .left);
    return a;
}

const Side = enum { left, right };

/// Lift `up`, a child of `a`, into `a`'s place. `side` is which child of
/// `a` it is: `.right` for `child2`, `.left` for `child1`.
fn rotate(self: *Tree, a: u32, up: u32, side: Side) u32 {
    const nodes = self.nodes.items;
    const other = if (side == .right) nodes[a].child1 else nodes[a].child2;
    const f = nodes[up].child1;
    const g = nodes[up].child2;

    // `up` takes `a`'s place under `a`'s parent, with `a` as its first child.
    nodes[up].child1 = a;
    nodes[up].parent = nodes[a].parent;
    nodes[a].parent = up;
    if (nodes[up].parent == null_node) {
        self.root = up;
    } else if (nodes[nodes[up].parent].child1 == a) {
        nodes[nodes[up].parent].child1 = up;
    } else {
        nodes[nodes[up].parent].child2 = up;
    }

    // Its taller child stays with it; the shorter goes down to `a`, into
    // the place `up` left.
    const keep, const give = if (nodes[f].height > nodes[g].height) .{ f, g } else .{ g, f };
    nodes[up].child2 = keep;
    if (side == .right) nodes[a].child2 = give else nodes[a].child1 = give;
    nodes[give].parent = a;

    nodes[a].box = nodes[other].box.join(nodes[give].box);
    nodes[a].height = 1 + @max(nodes[other].height, nodes[give].height);
    nodes[up].box = nodes[a].box.join(nodes[keep].box);
    nodes[up].height = 1 + @max(nodes[a].height, nodes[keep].height);
    return up;
}

// -------------------------------------------------------------------------
// Asking it
// -------------------------------------------------------------------------

/// Deep enough for any tree that balance allows: a balanced tree of
/// height 64 would hold more leaves than there are bytes.
const stack_depth = 128;

/// Call `visit(context, value)` for every leaf whose box overlaps `box`,
/// until it returns false.
///
/// `visit` is `comptime`, so it is compiled into the walk: no function
/// pointer is followed per leaf, and a visitor that is one line costs one
/// line.
pub fn query(self: *const Tree, box: Aabb, context: anytype, comptime visit: fn (@TypeOf(context), u32) bool) void {
    if (self.root == null_node) return;
    // A stack on the stack: a walk needs one slot per level, and a fixed
    // array of them is one fewer thing to allocate in the middle of a step.
    var stack: [stack_depth]u32 = undefined;
    var top: usize = 1;
    stack[0] = self.root;
    while (top > 0) {
        top -= 1;
        const node = &self.nodes.items[stack[top]];
        if (!node.box.overlaps(box)) continue;
        if (node.isLeaf()) {
            if (!visit(context, node.value)) return;
        } else if (top + 2 <= stack_depth) {
            stack[top] = node.child1;
            stack[top + 1] = node.child2;
            top += 2;
        }
    }
}

/// Walk the leaves a ray from `origin` along `translation` may hit, nearest
/// first as far as the tree can tell, calling `visit(context, value,
/// max_fraction)` for each. The visitor answers with the fraction of the
/// ray it hit that leaf at, which shortens the ray for the rest of the
/// walk; or with a negative number to say it missed; or with zero to stop.
///
/// A branch is skipped when its box misses the ray's own box, or when the
/// ray's line passes it by - the separating axis of a segment and a box,
/// which is the line's perpendicular (Gino van den Bergen's test).
pub fn rayCast(
    self: *const Tree,
    origin: Vec2,
    translation: Vec2,
    max_fraction: f32,
    context: anytype,
    comptime visit: fn (@TypeOf(context), u32, f32) f32,
) void {
    if (self.root == null_node) return;
    const length = translation.len();
    if (length == 0) return;
    const across = geometry.crossSV(1, translation.scale(1 / length));
    const across_abs = across.abs();

    var fraction = max_fraction;
    var reach = rayBox(origin, translation, fraction);
    var stack: [stack_depth]u32 = undefined;
    var top: usize = 1;
    stack[0] = self.root;
    while (top > 0) {
        top -= 1;
        const node = &self.nodes.items[stack[top]];
        if (!node.box.overlaps(reach)) continue;
        const separation = @abs(across.dot(origin.sub(node.box.center()))) - across_abs.dot(node.box.extent());
        if (separation > 0) continue;
        if (node.isLeaf()) {
            const hit = visit(context, node.value, fraction);
            if (hit == 0) return;
            if (hit > 0 and hit < fraction) {
                fraction = hit;
                reach = rayBox(origin, translation, fraction);
            }
        } else if (top + 2 <= stack_depth) {
            stack[top] = node.child1;
            stack[top + 1] = node.child2;
            top += 2;
        }
    }
}

/// The box around the part of a ray that is still being asked about.
fn rayBox(origin: Vec2, translation: Vec2, fraction: f32) Aabb {
    const end = origin.mulAdd(translation, fraction);
    return .{ .min = origin.min(end), .max = origin.max(end) };
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

/// Every branch holds its children, heights add up, the balance rule holds,
/// and parents and children agree. Panics otherwise, which is what a test
/// wants.
fn validate(self: *const Tree) !void {
    if (self.root == null_node) return;
    try testing.expectEqual(null_node, self.nodes.items[self.root].parent);
    var leaves: u32 = 0;
    try self.validateNode(self.root, &leaves);
    try testing.expectEqual(self.leaf_count, leaves);
}

fn validateNode(self: *const Tree, index: u32, leaves: *u32) !void {
    const node = &self.nodes.items[index];
    if (node.isLeaf()) {
        try testing.expectEqual(@as(i32, 0), node.height);
        leaves.* += 1;
        return;
    }
    const a = &self.nodes.items[node.child1];
    const b = &self.nodes.items[node.child2];
    try testing.expectEqual(index, a.parent);
    try testing.expectEqual(index, b.parent);
    try testing.expectEqual(1 + @max(a.height, b.height), node.height);
    try testing.expect(@abs(a.height - b.height) <= 1);
    const joined = a.box.join(b.box);
    try testing.expect(joined.min.eql(node.box.min) and joined.max.eql(node.box.max));
    try self.validateNode(node.child1, leaves);
    try self.validateNode(node.child2, leaves);
}

const Collect = struct {
    found: *std.ArrayList(u32),
    fn visit(self: *const Collect, value: u32) bool {
        self.found.append(testing.allocator, value) catch return false;
        return true;
    }
};

test "a tile map added row by row stays shallow, and finds exactly what a search of everything finds" {
    const gpa = testing.allocator;
    var tree: Tree = .empty;
    defer tree.deinit(gpa);

    // Sixty by forty tiles, added in the worst order for an unbalanced
    // tree: sorted.
    var boxes: std.ArrayList(Aabb) = .empty;
    defer boxes.deinit(gpa);
    for (0..40) |r| for (0..60) |c| {
        const at: Vec2 = .init(@floatFromInt(c), @floatFromInt(r));
        const box: Aabb = .fromCenter(at, .splat(0.5));
        _ = try tree.insert(gpa, box, @intCast(boxes.items.len));
        try boxes.append(gpa, box);
    };
    try tree.validate();
    // 2400 leaves: a perfect tree is 12 deep; balance allows about 1.44x.
    try testing.expect(tree.height() <= 18);

    var prng: std.Random.DefaultPrng = .init(3);
    const random = prng.random();
    var found: std.ArrayList(u32) = .empty;
    defer found.deinit(gpa);
    for (0..200) |_| {
        const at: Vec2 = .init(random.float(f32) * 64 - 2, random.float(f32) * 44 - 2);
        const probe: Aabb = .fromCenter(at, .init(random.float(f32) * 3, random.float(f32) * 3));
        found.clearRetainingCapacity();
        tree.query(probe, &Collect{ .found = &found }, Collect.visit);
        var expected: usize = 0;
        for (boxes.items) |b| {
            if (b.overlaps(probe)) expected += 1;
        }
        try testing.expectEqual(expected, found.items.len);
        for (found.items) |v| try testing.expect(boxes.items[v].overlaps(probe));
    }
}

test "leaves come and go, and the tree stays whole" {
    const gpa = testing.allocator;
    var tree: Tree = .empty;
    defer tree.deinit(gpa);

    var prng: std.Random.DefaultPrng = .init(5);
    const random = prng.random();
    var proxies: std.ArrayList(u32) = .empty;
    defer proxies.deinit(gpa);
    for (0..3000) |i| {
        // Mostly add, sometimes take away or move something already in.
        const roll = random.float(f32);
        if (roll < 0.25 and proxies.items.len > 0) {
            const k = random.uintLessThan(usize, proxies.items.len);
            tree.remove(proxies.swapRemove(k));
        } else if (roll < 0.4 and proxies.items.len > 0) {
            const k = random.uintLessThan(usize, proxies.items.len);
            const at: Vec2 = .init(random.float(f32) * 100, random.float(f32) * 100);
            try tree.move(gpa, proxies.items[k], .fromCenter(at, .splat(1)));
        } else {
            const at: Vec2 = .init(random.float(f32) * 100, random.float(f32) * 100);
            try proxies.append(gpa, try tree.insert(gpa, .fromCenter(at, .splat(1 + random.float(f32))), @intCast(i)));
        }
        if (i % 500 == 0) try tree.validate();
    }
    try tree.validate();
    try testing.expectEqual(@as(u32, @intCast(proxies.items.len)), tree.leaf_count);
    // Freed nodes are reused: the tree never holds more than a leaf and a
    // branch for each of the most leaves it ever had.
    try testing.expect(tree.nodes.items.len < 2 * 3000);
    while (proxies.pop()) |p| tree.remove(p);
    try testing.expectEqual(null_node, tree.root);
}

const Nearest = struct {
    boxes: []const Aabb,
    origin: Vec2,
    translation: Vec2,
    best: f32 = 1,
    best_value: ?u32 = null,
    asked: usize = 0,

    /// The box itself stands in for a shape: where the ray enters it.
    fn visit(self: *Nearest, value: u32, max_fraction: f32) f32 {
        self.asked += 1;
        const hit = slab(self.boxes[value], self.origin, self.translation) orelse return -1;
        if (hit > max_fraction) return -1;
        if (hit < self.best) {
            self.best = hit;
            self.best_value = value;
        }
        return hit;
    }

    /// Where a ray first enters a box, by slabs; null if it misses.
    fn slab(box: Aabb, origin: Vec2, translation: Vec2) ?f32 {
        var lo: f32 = 0;
        var hi: f32 = 1;
        inline for (.{ "x", "y" }) |axis| {
            const o = @field(origin, axis);
            const d = @field(translation, axis);
            const min = @field(box.min, axis);
            const max = @field(box.max, axis);
            if (d == 0) {
                if (o < min or o > max) return null;
            } else {
                var t1 = (min - o) / d;
                var t2 = (max - o) / d;
                if (t1 > t2) std.mem.swap(f32, &t1, &t2);
                lo = @max(lo, t1);
                hi = @min(hi, t2);
                if (lo > hi) return null;
            }
        }
        return lo;
    }
};

test "a ray finds the nearest box, asking about few" {
    const gpa = testing.allocator;
    var tree: Tree = .empty;
    defer tree.deinit(gpa);
    var boxes: std.ArrayList(Aabb) = .empty;
    defer boxes.deinit(gpa);
    for (0..50) |r| for (0..50) |c| {
        const box: Aabb = .fromCenter(.init(@floatFromInt(c * 2), @floatFromInt(r * 2)), .splat(0.5));
        _ = try tree.insert(gpa, box, @intCast(boxes.items.len));
        try boxes.append(gpa, box);
    };

    var prng: std.Random.DefaultPrng = .init(9);
    const random = prng.random();
    for (0..100) |_| {
        const origin: Vec2 = .init(random.float(f32) * 100 - 1, -5);
        const translation: Vec2 = .init(random.float(f32) * 20 - 10, 120);
        var nearest: Nearest = .{ .boxes = boxes.items, .origin = origin, .translation = translation };
        tree.rayCast(origin, translation, 1, &nearest, Nearest.visit);

        // What looking at every box finds.
        var best: f32 = 1;
        var best_value: ?u32 = null;
        for (boxes.items, 0..) |b, i| {
            const hit = Nearest.slab(b, origin, translation) orelse continue;
            if (hit < best) {
                best = hit;
                best_value = @intCast(i);
            }
        }
        try testing.expectEqual(best_value, nearest.best_value);
        if (best_value != null) try testing.expectApproxEqAbs(best, nearest.best, 1e-6);
        // And it did not have to ask about all 2500 to know.
        try testing.expect(nearest.asked < 200);
    }
}
