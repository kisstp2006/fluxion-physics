# Fluxion Physics

Rigid bodies in a plane. For Zig 0.16, on every core the machine has or on
none at all, which is the browser.

| Module | What it is |
| --- | --- |
| `World` | The bodies, the shapes on them, and one step of time. |
| `Body` | A rigid body: where it is, how it moves, what it weighs. |
| `shape` | Circles and convex polygons, materials, and who touches whom. |
| `collide` | Where two shapes touch, and how deep. |
| `broadphase` | Which pairs are close enough to be worth asking. |
| `contact` | A contact as the solver sees it. |
| `solver` | How a step is spread across the cores, and why the answer does not depend on how many. |
| `geometry` | Rotations, transforms, and 2D boxes. |

```zig
const physics = @import("fluxion_physics");

var world: physics.World = .init(gpa, .{ .units_per_metre = 100 });
defer world.deinit();

const ground = try world.createBody(.{ .type = .static, .position = .init(400, 580) });
_ = try world.addShape(ground, .box(400, 20));

const crate = try world.createBody(.{ .position = .init(400, 100), .angle = 0.3 });
_ = try world.addShape(crate, .{ .geometry = .{ .polygon = .box(24, 24) }, .material = .{ .friction = 0.8 } });

var jobs: physics.Jobs = try .init(gpa, .{ .io = io });
defer jobs.deinit();

try world.step(1.0 / 60.0, &jobs);
const where = world.body(crate).?.position();
```

**A body is a handle, and a shape is a handle on a body.** Both are
[Fluxion Id](https://github.com/kisstp2006/fluxion-id) generational handles:
eight bytes to copy into a component, and a handle to something destroyed
answers null for ever, however many times its slot is reused. `world.body(h)`
hands out a pointer to push on; hold the handle, not the pointer.

**Three kinds of body.** Static never moves and holds everything up.
Kinematic moves the way it is told and is pushed by nothing. Dynamic is what
physics is for. Only dynamic bodies collide with each other, and a kinematic
platform carries what stands on it.

**Two geometries, on purpose.** A circle and a convex polygon of up to eight
corners. `Polygon.fromPoints` takes corners in any order and hands back the
convex hull, so a shape is never concave by accident and never wound the
wrong way. Several shapes on one body make a compound.

**Mass comes from density and area**, so a big crate is heavier than a small
one without anyone typing a mass, and a body's centre of mass is where its
shapes put it. A dynamic body with no mass gets one, because it was meant to
move.

**The axes are the engine's.** `+x` right, `+y` down, a positive angle
turning `+x` towards `+y`. A body's position goes into a sprite's transform
with no flip anywhere, and gravity pulls towards `+y`.

**Units are yours.** Set `units_per_metre` and every tolerance the solver
keeps in metres - how deep a shape may sink, how slow a hit stops bouncing,
how fast a penetration is pushed apart - is scaled. A hundred, for a game
that thinks in pixels.

## A step, on every core

A step is nine phases, and six run on every core the scheduler has:

| Phase | What | Where |
| --- | --- | --- |
| integrate velocities | gravity, forces, damping | every core |
| update boxes | each shape's box in the world | every core |
| broad phase | which boxes overlap: a sort and a sweep | one thread |
| narrow phase | where each pair touches | every core |
| prepare | contacts into constraints, warm-started | one thread |
| colour | contacts into runs that share no body | one thread |
| solve | warm start, then the velocity passes | every core, per colour |
| integrate positions | move, turn, rebuild transforms | every core |
| remember | impulses for next step, begin and end events | one thread |

The solver is the interesting one. A contact writes to two bodies, and a
body may be in a dozen contacts, so two contacts that share a body cannot be
solved at once. **Graph colouring** sorts the contacts into colours such that
no two of one colour share a body that can move; each colour is then solved
with every contact in it at once, colour after colour, with a join between.
A wall is never written to, so a hundred crates on one floor are one colour
and not a hundred. Box2D v3 does this, and it is what made its solver scale.

**The result is the same on one core and on sixteen.** The pairs come out of
the broad phase in one order, the colouring is greedy in that order, and
within a colour nothing shares memory, so the order jobs run in cannot change
what they compute. The demo and the tests run every scene both ways and
compare every position to the bit.

**It runs on [Fluxion Jobs](https://github.com/kisstp2006/fluxion-jobs)**,
and what that costs is worth knowing. That scheduler is one queue and one
lock, and a job that wakes a sleeping worker costs tens of microseconds; a
body integrates in tens of nanoseconds. So the grains are large - a colour
with fewer than a few hundred contacts is solved on the calling thread - and
a scene of a few hundred bodies never spawns a solver job at all, which is
right for it. On the machine this was written on, six hundred bodies in a
heap take about the same time both ways; the parallel structure is there and
correct, and pays for itself when the scheduler underneath stops waking a
thread per job. The demo prints both timings so the crossover can be seen.

## In a browser

```bash
zig build web          # zig-out/web/index.html and the .wasm beside it
```

A bowl of balls and crates on a canvas, stepped by the page's animation
frame. No threads: the scheduler sees the target has none and has zero
workers, and every phase runs on the thread that asked. Nothing in the
library is compiled out and nothing is stubbed. **It runs anywhere because
it does nothing**: no files, no clock, no threads of its own - an allocator
and arithmetic.

## What comes back from a step

- **Positions**, on the bodies: `body.position()`, `body.angle`, and
  `body.transform` for placing a sprite.
- **Events**: `world.beginEvents()` and `world.endEvents()` list every pair
  of shapes that started or stopped touching during the last step, including
  because one was destroyed. A **sensor** shape appears in them and pushes
  nothing: a trigger volume, a pickup radius.
- **Queries**, between steps: `castRay` for the first thing along a line,
  `overlapPoint` for what is under the mouse, `overlapAabb` for everything in
  a box. They walk every shape - see "What is not here".

## Filtering

Sixteen category bits, a mask, and a group, which is Box2D's scheme:

```zig
const player: physics.Filter = .{ .category = 0b01, .mask = 0b10 };  // touches only category 2
const ragdoll_part: physics.Filter = .{ .group = -7 };               // never touches its own group
```

## What is not here

Each of these is a known piece of work, listed with what it would take.

- **Joints.** Revolute, prismatic, distance, and the rest. The solver's
  colouring already treats a constraint as two bodies and an impulse, so a
  joint is a new `prepare` and `solve` beside `contact`'s and a table to keep
  them in. The first thing this package should grow.
- **Sleeping.** A body at rest for a while is skipped until something touches
  it. Needs islands - connected sets of bodies - which the colouring does not
  compute. Without it a level of a thousand resting crates costs a thousand
  crates a step.
- **Continuous collision.** A fast bullet against a thin wall goes through it
  between steps, and a body squeezed hard by a pile can be pushed through
  thin static geometry. Until then: make static geometry thick, keep dynamic
  bodies bigger than what they move per step, and cap speeds.
- **A tree for queries.** The broad phase is a sweep and the queries walk
  every shape. A dynamic AABB tree makes a ray logarithmic; it is one file,
  and the sweep would stay for pairs.
- **Capsules and rounded polygons.** Each is a row and a column of the narrow
  phase's pairings.
- **A position solver.** Penetration is corrected through a velocity bias
  (Baumgarte), so a resting stack sits a few millimetres into itself. Box2D's
  separate position pass is the alternative and is stiffer for tall stacks.
- **Entities.** This package knows nothing about an ECS. The engine keeps a
  `BodyId` in a component and copies transforms out after its `.fixed` stage;
  that integration lives in the engine, not here.

## Install

```bash
zig fetch --save git+https://github.com/kisstp2006/fluxion-physics
```

```zig
const fluxion = b.dependency("fluxion_physics", .{ .target = target, .optimize = optimize });
exe_mod.addImport("fluxion_physics", fluxion.module("fluxion_physics"));
```

Three dependencies come with it, fetched the same way and needing nothing
from you: [Fluxion Math](https://github.com/kisstp2006/fluxion-math) for
`Vec2`, [Fluxion Jobs](https://github.com/kisstp2006/fluxion-jobs) for the
step, and [Fluxion Id](https://github.com/kisstp2006/fluxion-id) for the
handles. All three are pinned to the commits the rest of the ecosystem pins,
so a program that already has them gets one copy.

## Where it sits

The third tier of the Fluxion licence ladder: `BSD-2-Clause`, a subsystem
built on three tier-two libraries. Ship it in a game and its notice goes in
the about box.

## The tests

Every scene runs twice, once with a scheduler and every core, once with
neither, and passes the same way. A ball dropped on a floor lands and stays;
a stack of six crates stands; a ball with restitution comes back up; eighty
mixed bodies poured into a bowl end up in the same places to the bit; masks
keep a ghost from the floor; a sensor reports and pushes nothing; a kinematic
platform carries a crate; rays find the nearest thing and honour filters. The
browser build is part of `zig build test`, so a change that breaks the
no-thread path fails here and not in a browser later.

## Build

```bash
zig build test        # run the test suite, and build the wasm
zig build example     # a heap in a bowl, with every core and with none
zig build web         # the browser example into zig-out/web
zig build docs        # generate API docs into zig-out/docs
```

## Licence

`BSD-2-Clause`. See [LICENSE](LICENSE).
