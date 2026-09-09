// SPDX-License-Identifier: BSD-2-Clause

//! The broad phase: which pairs of shapes are close enough to be worth
//! looking at.
//!
//! Testing every shape against every other is quadratic, and a level with a
//! thousand tiles would spend its whole frame discovering that tiles far
//! apart are far apart. This keeps the shapes sorted along x by the left
//! edge of their boxes and sweeps once: a shape is compared only with the
//! ones whose left edge falls before its right edge, which for anything but
//! a scene of overlapping giants is a handful.
//!
//! **The order is kept between steps**, and re-sorted with an insertion
//! sort. A body moves a little per step, so the list is nearly sorted
//! already and the sort is a single pass with a few swaps - linear, where a
//! sort from scratch is `n log n`. A shape that teleports across the level
//! costs a long walk once.
//!
//! **The pairs come out in one order, every time**, whatever the machine.
//! The sweep visits shapes in list order and the list order depends only on
//! the boxes and on the order shapes were added. That is what makes a step
//! deterministic: the solver's colouring walks the pairs in this order and
//! gives the same contact the same colour on every run.
//!
//! What this is not: a tree. A dynamic AABB tree answers a ray or a box
//! query in logarithmic time and a sweep answers it in linear time, so the
//! queries in `World` walk every shape. A tree is the next thing this file
//! grows if a game asks a thousand questions a frame; the pair-finding
//! would stay a sweep, which for pairs is the faster of the two.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;

const geometry = @import("geometry.zig");
const Aabb = geometry.Aabb;

/// Two shapes whose boxes overlap, by index into the world's shape table,
/// with `a < b` so a pair has one spelling.
pub const Pair = struct {
    a: u32,
    b: u32,
};

/// What the sweep asks of a shape besides its box: whether this pair is
/// worth handing on at all. Same body, both static, filtered out - the
/// world knows, the sweep does not.
pub fn Sweep(comptime Context: type) type {
    return struct {
        const Self = @This();

        /// Shape indices, sorted by `aabbs[i].min.x` after `update`.
        order: std.ArrayList(u32) = .empty,

        pub const empty: Self = .{};

        pub fn deinit(self: *Self, gpa: Allocator) void {
            self.order.deinit(gpa);
            self.* = undefined;
        }

        /// A shape that now exists. It goes to the end and the next sort
        /// walks it into place.
        pub fn add(self: *Self, gpa: Allocator, index: u32) Allocator.Error!void {
            try self.order.append(gpa, index);
        }

        /// A shape that no longer does. A linear search, because removal is
        /// rare beside stepping and a map from index to position would be
        /// paid for on every swap of the sort.
        pub fn remove(self: *Self, index: u32) void {
            for (self.order.items, 0..) |item, i| {
                if (item == index) {
                    _ = self.order.orderedRemove(i);
                    return;
                }
            }
        }

        /// Re-sort, sweep, and append every overlapping pair `context`
        /// accepts to `pairs`.
        pub fn update(
            self: *Self,
            gpa: Allocator,
            aabbs: []const Aabb,
            context: Context,
            comptime accept: fn (Context, u32, u32) bool,
            pairs: *std.ArrayList(Pair),
        ) Allocator.Error!void {
            const order = self.order.items;

            // Insertion sort by the left edge. Ties are broken by index so
            // the order is a function of the boxes alone and not of the
            // sort's history.
            var i: usize = 1;
            while (i < order.len) : (i += 1) {
                const key = order[i];
                const key_x = aabbs[key].min.x;
                var j = i;
                while (j > 0 and lessThan(aabbs[order[j - 1]].min.x, order[j - 1], key_x, key)) : (j -= 1) {
                    order[j] = order[j - 1];
                }
                order[j] = key;
            }

            for (order, 0..) |a, k| {
                const box_a = aabbs[a];
                for (order[k + 1 ..]) |b| {
                    const box_b = aabbs[b];
                    // Everything after this starts further right than this
                    // box ends, so nothing after it can overlap either.
                    if (box_b.min.x > box_a.max.x) break;
                    if (box_b.min.y > box_a.max.y or box_a.min.y > box_b.max.y) continue;
                    if (!accept(context, a, b)) continue;
                    try pairs.append(gpa, .{ .a = @min(a, b), .b = @max(a, b) });
                }
            }
        }

        /// "Does `(x1, i1)` sort after `(x2, i2)`", for the insertion sort.
        inline fn lessThan(x1: f32, index1: u32, x2: f32, index2: u32) bool {
            return x1 > x2 or (x1 == x2 and index1 > index2);
        }
    };
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

const AcceptAll = struct {
    fn accept(_: void, _: u32, _: u32) bool {
        return true;
    }
};

test "the sweep finds the overlapping pairs and only those" {
    const gpa = testing.allocator;
    var sweep: Sweep(void) = .empty;
    defer sweep.deinit(gpa);

    const boxes = [_]Aabb{
        .{ .min = .init(0, 0), .max = .init(2, 2) }, // 0
        .{ .min = .init(1, 1), .max = .init(3, 3) }, // 1: overlaps 0
        .{ .min = .init(5, 0), .max = .init(6, 1) }, // 2: alone
        .{ .min = .init(1, 5), .max = .init(3, 6) }, // 3: same x as 1, wrong y
        .{ .min = .init(-1, 1.5), .max = .init(9, 1.6) }, // 4: a long bar across 0 and 1, above 2
    };
    // Added out of order, to be sorted into it.
    for ([_]u32{ 3, 0, 4, 2, 1 }) |i| try sweep.add(gpa, i);

    var pairs: std.ArrayList(Pair) = .empty;
    defer pairs.deinit(gpa);
    try sweep.update(gpa, &boxes, {}, AcceptAll.accept, &pairs);

    try testing.expectEqualSlices(u32, &.{ 4, 0, 1, 3, 2 }, sweep.order.items);
    try testing.expectEqual(@as(usize, 3), pairs.items.len);
    var seen = [_]bool{false} ** 5;
    for (pairs.items) |p| {
        try testing.expect(p.a < p.b);
        try testing.expect(boxes[p.a].overlaps(boxes[p.b]));
        seen[p.a] = true;
        seen[p.b] = true;
    }
    try testing.expect(!seen[3]);

    // Running it again on the same boxes gives the same pairs in the same
    // order, and removing one takes its pairs with it.
    pairs.clearRetainingCapacity();
    try sweep.update(gpa, &boxes, {}, AcceptAll.accept, &pairs);
    try testing.expectEqual(@as(usize, 3), pairs.items.len);

    sweep.remove(4);
    pairs.clearRetainingCapacity();
    try sweep.update(gpa, &boxes, {}, AcceptAll.accept, &pairs);
    try testing.expectEqual(@as(usize, 1), pairs.items.len);
    try testing.expectEqual(Pair{ .a = 0, .b = 1 }, pairs.items[0]);
}

test "a shape that moved is sorted back into place" {
    const gpa = testing.allocator;
    var sweep: Sweep(void) = .empty;
    defer sweep.deinit(gpa);
    var boxes = [_]Aabb{
        .{ .min = .init(0, 0), .max = .init(1, 1) },
        .{ .min = .init(2, 0), .max = .init(3, 1) },
        .{ .min = .init(4, 0), .max = .init(5, 1) },
    };
    for (0..3) |i| try sweep.add(gpa, @intCast(i));
    var pairs: std.ArrayList(Pair) = .empty;
    defer pairs.deinit(gpa);
    try sweep.update(gpa, &boxes, {}, AcceptAll.accept, &pairs);
    try testing.expectEqual(@as(usize, 0), pairs.items.len);

    // Shape 0 jumps to the far right, onto shape 2.
    boxes[0] = .{ .min = .init(4.5, 0), .max = .init(5.5, 1) };
    try sweep.update(gpa, &boxes, {}, AcceptAll.accept, &pairs);
    try testing.expectEqualSlices(u32, &.{ 1, 2, 0 }, sweep.order.items);
    try testing.expectEqual(@as(usize, 1), pairs.items.len);
    try testing.expectEqual(Pair{ .a = 0, .b = 2 }, pairs.items[0]);
}
