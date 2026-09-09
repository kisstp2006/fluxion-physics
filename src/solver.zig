// SPDX-License-Identifier: BSD-2-Clause

//! How a step is spread across the cores, and why the answer does not
//! depend on how many there are.
//!
//! Most of a step is embarrassingly parallel: every body integrates on its
//! own, every shape's box is its own, every pair the broad phase found is
//! tested on its own. `forRange` cuts those into chunks and hands each to
//! [Fluxion Jobs](https://github.com/kisstp2006/fluxion-jobs).
//!
//! **The solver is the hard part**, because a contact writes to two bodies
//! and a body may be in a dozen contacts. Solving two contacts that share a
//! body at once is a race. The answer is *graph colouring*: contacts are
//! sorted into colours such that no two contacts of one colour share a body
//! that can move, and then each colour is solved with every contact in it
//! at once, colour after colour. Within a colour nothing touches the same
//! memory, so nothing locks; between colours there is a join. Box2D v3 does
//! this, and it is what made its solver scale.
//!
//! **A wall may be in every colour.** A static or kinematic body is never
//! written to by the solver - see `contact.store` - so two contacts against
//! the ground are not a conflict and go in one colour. Without that, a
//! floor with a hundred crates on it would put every crate in a different
//! colour and the colouring would be a serial solve with extra steps.
//!
//! **The result is the same on one core and on sixteen.** The colouring is
//! greedy in pair order, and pair order comes from the broad phase's sweep,
//! which is deterministic. Within a colour the contacts are independent, so
//! the order jobs run them in cannot change what they compute. The demo
//! proves it by running the same scene both ways and comparing every
//! position to the bit.
//!
//! **Past twenty-four colours the rest are solved serially.** A body in
//! more contacts than that - the bottom of a huge pile - forces a new
//! colour for each, and a colour with one contact in it is a join for
//! nothing. Those go in an overflow bucket the main thread walks alone.

const std = @import("std");
const testing = std.testing;
const Allocator = std.mem.Allocator;
const Jobs = @import("fluxion_jobs").Jobs;

const contact = @import("contact.zig");
const Constraint = contact.Constraint;

/// Call `function(context, begin, end)` for consecutive chunks of
/// `[0, count)` at most `grain` long, at the same time where there are
/// workers, and return when all have run.
///
/// Unlike `fluxion_jobs.parallel.forEach` this never fails: a scheduler
/// with no room left runs the chunk here instead, and one with no workers
/// - the browser - runs the whole range here without spawning anything,
/// which is what would happen anyway with the queue in between.
pub fn forRange(
    jobs: *Jobs,
    count: usize,
    grain: usize,
    context: anytype,
    comptime function: fn (@TypeOf(context), usize, usize) void,
) void {
    if (count == 0) return;
    const g = @max(grain, 1);
    if (count <= g or jobs.workerCount() == 0) {
        function(context, 0, count);
        return;
    }

    var handles: [64]Jobs.Handle = undefined;
    var begin: usize = 0;
    while (begin < count) {
        var spawned: usize = 0;
        while (spawned < handles.len and begin < count) {
            const stop = @min(begin + g, count);
            if (jobs.spawn(function, .{ context, begin, stop })) |handle| {
                handles[spawned] = handle;
                spawned += 1;
            } else |_| {
                function(context, begin, stop);
            }
            begin = stop;
        }
        for (handles[0..spawned]) |handle| jobs.wait(handle);
    }
}

/// How many colours before the overflow bucket. A `u32` of used-colour bits
/// per body, with room to spare.
pub const max_colours = 24;
pub const overflow = max_colours;

/// The contacts of a step, grouped by colour.
pub const Colouring = struct {
    /// Which colours each body slot is already in this step, as bits.
    used: std.ArrayList(u32) = .empty,
    /// The colour of each constraint, in input order.
    colour: std.ArrayList(u8) = .empty,
    /// The constraints, reordered so each colour is one run.
    sorted: std.ArrayList(Constraint) = .empty,
    /// Where each colour starts in `sorted`; one past the end at the end.
    offsets: [max_colours + 2]u32 = @splat(0),

    pub const empty: Colouring = .{};

    pub fn deinit(self: *Colouring, gpa: Allocator) void {
        self.used.deinit(gpa);
        self.colour.deinit(gpa);
        self.sorted.deinit(gpa);
        self.* = undefined;
    }

    /// Give every constraint a colour and sort them into runs.
    ///
    /// Greedy: each constraint takes the lowest colour neither of its
    /// movable bodies is in yet. Not optimal - that problem is hard - and
    /// not needed to be: a few more colours cost a few more joins.
    pub fn assign(self: *Colouring, gpa: Allocator, constraints: []const Constraint, body_slots: usize) Allocator.Error!void {
        try self.used.resize(gpa, body_slots);
        @memset(self.used.items, 0);
        try self.colour.resize(gpa, constraints.len);

        var counts: [max_colours + 1]u32 = @splat(0);
        for (constraints, self.colour.items) |*c, *colour| {
            var mask: u32 = 0;
            if (c.inv_mass_a != 0) mask |= self.used.items[c.body_a];
            if (c.inv_mass_b != 0) mask |= self.used.items[c.body_b];

            const free = @ctz(~mask);
            const chosen: u8 = if (free < max_colours) @intCast(free) else overflow;
            colour.* = chosen;
            counts[chosen] += 1;
            if (chosen != overflow) {
                const bit = @as(u32, 1) << @intCast(chosen);
                if (c.inv_mass_a != 0) self.used.items[c.body_a] |= bit;
                if (c.inv_mass_b != 0) self.used.items[c.body_b] |= bit;
            }
        }

        // Prefix sums give each colour its run; a second pass scatters.
        self.offsets[0] = 0;
        for (counts, 0..) |n, i| self.offsets[i + 1] = self.offsets[i] + n;

        try self.sorted.resize(gpa, constraints.len);
        var cursor = self.offsets;
        for (constraints, self.colour.items) |c, colour| {
            self.sorted.items[cursor[colour]] = c;
            cursor[colour] += 1;
        }
    }

    /// The constraints of one colour, to solve at once.
    pub fn run(self: *Colouring, colour: usize) []Constraint {
        return self.sorted.items[self.offsets[colour]..self.offsets[colour + 1]];
    }

    /// How many colours were needed, not counting the overflow.
    pub fn colourCount(self: *const Colouring) usize {
        var n: usize = 0;
        for (0..max_colours) |c| {
            if (self.offsets[c + 1] > self.offsets[c]) n = c + 1;
        }
        return n;
    }
};

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

fn fakeConstraint(a: u32, b: u32, a_moves: bool, b_moves: bool) Constraint {
    return .{
        .key = .{ .a = a, .b = b },
        .body_a = a,
        .body_b = b,
        .normal = .unit_y,
        .tangent = .unit_x,
        .friction = 0,
        .restitution = 0,
        .inv_mass_a = if (a_moves) 1 else 0,
        .inv_mass_b = if (b_moves) 1 else 0,
        .inv_inertia_a = 0,
        .inv_inertia_b = 0,
        .points = undefined,
        .count = 0,
    };
}

test "no colour holds two contacts that share a movable body" {
    const gpa = testing.allocator;
    var colouring: Colouring = .empty;
    defer colouring.deinit(gpa);

    // A chain: 0-1, 1-2, 2-3, and everything against the ground, body 9.
    const constraints = [_]Constraint{
        fakeConstraint(0, 1, true, true),
        fakeConstraint(1, 2, true, true),
        fakeConstraint(2, 3, true, true),
        fakeConstraint(0, 9, true, false),
        fakeConstraint(1, 9, true, false),
        fakeConstraint(2, 9, true, false),
        fakeConstraint(3, 9, true, false),
    };
    try colouring.assign(gpa, &constraints, 10);
    try testing.expectEqual(constraints.len, colouring.sorted.items.len);

    for (0..max_colours) |colour| {
        var seen: [10]bool = @splat(false);
        for (colouring.run(colour)) |c| {
            if (c.inv_mass_a != 0) {
                try testing.expect(!seen[c.body_a]);
                seen[c.body_a] = true;
            }
            if (c.inv_mass_b != 0) {
                try testing.expect(!seen[c.body_b]);
                seen[c.body_b] = true;
            }
        }
    }
    // Two colours for the chain, and the ground contacts fit beside them.
    try testing.expect(colouring.colourCount() <= 3);
    try testing.expectEqual(@as(usize, 0), colouring.run(overflow).len);
}

test "a body in more contacts than there are colours spills into the overflow" {
    const gpa = testing.allocator;
    var colouring: Colouring = .empty;
    defer colouring.deinit(gpa);

    var constraints: [max_colours + 3]Constraint = undefined;
    for (&constraints, 0..) |*c, i| c.* = fakeConstraint(0, @intCast(i + 1), true, true);
    try colouring.assign(gpa, &constraints, constraints.len + 1);
    try testing.expectEqual(@as(usize, max_colours), colouring.colourCount());
    try testing.expectEqual(@as(usize, 3), colouring.run(overflow).len);
}

fn fill(values: []u32, begin: usize, end: usize) void {
    for (values[begin..end], begin..) |*v, i| v.* = @intCast(i);
}

test "forRange covers the range exactly once, with workers and without" {
    const gpa = testing.allocator;
    const modes = [_]Jobs.Options{
        .{ .io = testing.io, .workers = .auto },
        .{ .io = null },
    };
    for (modes) |mode| {
        var jobs: Jobs = try .init(gpa, mode);
        defer jobs.deinit();

        const values = try gpa.alloc(u32, 10_000);
        defer gpa.free(values);
        @memset(values, 0xFFFF_FFFF);
        forRange(&jobs, values.len, 7, values, fill);
        for (values, 0..) |v, i| try testing.expectEqual(@as(u32, @intCast(i)), v);
    }
}
