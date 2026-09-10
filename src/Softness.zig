// SPDX-License-Identifier: BSD-2-Clause

//! A spring as the solver uses it: three numbers that turn a rigid
//! constraint's impulse into a spring's.
//!
//! **Why every constraint is a little bit of a spring.** A solver that
//! takes back drift by asking for a fixed fraction of it as a velocity -
//! Baumgarte's way - puts the correction into the bodies as real speed.
//! When the solver converges that is harmless. When it does not - a heavy
//! crate on a light bridge, a wrecking ball on a chain - each correction
//! overshoots a little, the next is fed from the overshoot, and the
//! structure gains energy until it flies apart. A spring, stepped
//! implicitly, cannot add energy however stiff it is set: its damping is
//! built into how it is solved. So contacts and joints take back drift the
//! way a very stiff, heavily damped spring would, and a stiff enough spring
//! is rigid to the eye.
//!
//! This is Erin Catto's "soft step" ("Solver2D", 2024, and Box2D v3). The
//! arithmetic is the implicit Euler step of a damped spring of `hertz` and
//! `damping_ratio`, rearranged so that the spring's stiffness and damping
//! never appear on their own and cannot be too big for the step:
//!
//! ```text
//! impulse = -mass_scale * m * (cdot + bias_rate * C) - impulse_scale * total
//! ```
//!
//! for a constraint of effective mass `m`, speed `cdot`, error `C` and
//! accumulated impulse `total`. Given in hertz and not in newtons per
//! metre, because a frequency means the same for a crate as for a car.

const std = @import("std");
const testing = std.testing;

const Softness = @This();

/// How much of the error is asked back, as a velocity, per unit of it.
bias_rate: f32 = 0,
/// How much of a rigid constraint's impulse is applied.
mass_scale: f32 = 0,
/// How much of what was already applied is taken back each pass: the give
/// that makes it a spring and not a rod, and the damping.
impulse_scale: f32 = 0,

/// Rigid, and asking nothing back: what a relaxing pass solves with.
pub const rigid: Softness = .{ .bias_rate = 0, .mass_scale = 1, .impulse_scale = 0 };

/// A spring of `hertz` and `damping_ratio`, stepped `h` seconds at a time.
/// Box2D v3's `b2MakeSoft`. Zero hertz is all zeros, and pushes nothing.
pub fn of(hertz: f32, damping_ratio: f32, h: f32) Softness {
    if (hertz <= 0) return .{};
    const omega = 2 * std.math.pi * hertz;
    const a1 = 2 * damping_ratio + h * omega;
    const a2 = h * omega * a1;
    const a3 = 1 / (1 + a2);
    return .{ .bias_rate = omega / a1, .mass_scale = a2 * a3, .impulse_scale = a3 };
}

/// Rigid, taking back `fraction` of the drift per step as a velocity:
/// Baumgarte's way, with none of a spring's give. For tests that check a
/// pass against numbers worked out by hand.
pub fn baumgarte(fraction: f32, inv_h: f32) Softness {
    return .{ .bias_rate = fraction * inv_h, .mass_scale = 1, .impulse_scale = 0 };
}

/// The impulse a constraint with effective mass `mass`, speed `cdot` and
/// error `c` asks for this pass, given `total` so far.
pub inline fn impulse(s: Softness, mass: f32, cdot: f32, c: f32, total: f32) f32 {
    return -s.mass_scale * mass * (cdot + s.bias_rate * c) - s.impulse_scale * total;
}

test "a spring of zero hertz is nothing, and a stiff one is nearly a rod" {
    const none: Softness = .of(0, 0, 1.0 / 60.0);
    try testing.expectEqual(@as(f32, 0), none.mass_scale);
    try testing.expectEqual(@as(f32, 0), none.impulse(1, 5, 5, 5));

    const stiff: Softness = .of(1000, 1, 1.0 / 60.0);
    try testing.expect(stiff.mass_scale > 0.99);
    try testing.expect(stiff.impulse_scale < 0.01);

    // Softer springs give way more: less of the rigid impulse, more taken back.
    const soft: Softness = .of(2, 0.5, 1.0 / 60.0);
    try testing.expect(soft.mass_scale < stiff.mass_scale);
    try testing.expect(soft.impulse_scale > stiff.impulse_scale);

    // Rigid asks for exactly what stops the constraint, and keeps it.
    try testing.expectEqual(@as(f32, -6), rigid.impulse(2, 3, 100, 100));
}
