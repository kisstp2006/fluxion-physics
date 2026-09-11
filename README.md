# Fluxion Physics

Rigid bodies in a plane. For Zig 0.16, on every core the machine has or on
none at all, which is the browser.

| Module | What it is |
| --- | --- |
| `World` | The bodies, the shapes and joints on them, and one step of time. |
| `Body` | A rigid body: where it is, how it moves, what it weighs, whether it sleeps. |
| `shape` | Circles and convex polygons, materials, and who touches whom. |
| `joint` | Hinges, sliders, rods, ropes, springs, welds, wheels, and a pointer to drag with. |
| `collide` | Where two shapes touch, and how deep. |
| `broadphase` | Which pairs of moving shapes are close enough to be worth asking. |
| `Tree` | The level's shapes in a tree of boxes: what is near, without looking at all of it. |
| `continuous` | Where a fast body first touched something, and whether that touch mattered. |
| `contact` | A contact as the solver sees it. |
| `Softness` | A spring as the solver steps it, and why every constraint is a little of one. |
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

// A door on a hinge, a quarter turn each way.
const door = try world.createBody(.{ .position = .init(200, 300) });
_ = try world.addShape(door, .box(40, 4));
_ = try world.createJoint(.{ .revolute = .{
    .body_a = ground,
    .body_b = door,
    .anchor = .init(160, 300),
    .limit = .{ .lower = -std.math.pi / 2.0, .upper = std.math.pi / 2.0 },
} });

var jobs: physics.Jobs = try .init(gpa, .{ .io = io });
defer jobs.deinit();

try world.step(1.0 / 60.0, &jobs);
const where = world.body(crate).?.position();
```

**A body is a handle, and so is a shape and a joint.** All three are
[Fluxion Id](https://github.com/kisstp2006/fluxion-id) generational handles:
eight bytes to copy into a component, and a handle to something destroyed
answers null for ever, however many times its slot is reused.
`world.body(h)` hands out a pointer to push on; hold the handle, not the
pointer.

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

## Joints

| Joint | What it holds | What it grows |
| --- | --- | --- |
| `distance` | two points a length apart | a rod, a spring, or a rope |
| `revolute` | two points together | a hinge: limits, a motor, a spring |
| `prismatic` | one body sliding along a line on the other | a piston, a lift: limits, a motor, a spring |
| `weld` | two bodies in one pose | rigid, or bending on a spring |
| `mouse` | a point of one body towards a target | dragging with the pointer |
| `wheel` | a wheel on a sprung axle | a car: suspension, limits, a motor |

**Anchors are given in the world, once**: the pose the bodies are in when
the joint is made is its rest pose. A motor's speed, a limit, a mouse
joint's target are fields, written between steps - a throttle writes the
motor every frame. Two bodies a joint holds do not collide with each other
unless it says `collide_connected`. `world.jointReaction(h)` is the force
it held with, which is what a breakable joint is made of.

**Springs are given in hertz**, not newtons per metre, because a frequency
means the same for a crate as for a car. A distance joint with a slack
spring and a `max_length` is a rope.

## Sleeping

An *island* is a set of dynamic bodies joined by touching and by joints.
When every body in one has been still for half a second, the whole island
sleeps: a step no longer moves it, tests its pairs or solves its contacts,
which keep their impulses for when it wakes. It wakes as a whole, when
anything could have changed its mind: a velocity, force or impulse from
outside, `setTransform`, a shape added or taken away, a contact that ends
(destroy the bottom crate of a sleeping stack and the rest comes down), an
awake body touching it, a kinematic body moving against it, a joint's motor
given a speed or a mouse joint's target moved. Islands are found again
every step, so there is none to keep in step with what touches what.

A settled heap of four hundred crates costs nothing at all once it sleeps:
no pairs, no constraints, a pass over the bodies.

## The level

Everything static - the ground, the walls, a tile map - is kept in a
**tree of boxes** (`Tree`), and everything that moves in a list sorted
along x. Each moving shape asks the tree what is near it, and a query walks
only the branches its ray or box crosses, so a step costs what is near the
things that move, not what the level is made of. A tile map is fine built
either way: a body per tile, the way an entity system would, or one body of
many shapes. The passes over bodies walk only the bodies that can move, so
seven thousand tiles that never do are not visited a dozen times a step.

Measured on a level of 7200 half-metre tiles, 120 columns sixty deep, with
300 crates and balls on it:

| | Before: a sort and a sweep | Now |
| --- | --- | --- |
| a step, every tile its own body | 2.37 ms | 0.34 ms |
| a step, every tile a shape on one body | 2.04 ms | 0.29 ms |
| a thousand rays onto it | 85-106 ms | 1.9 ms |

The sweep alone was slowest at exactly this: every tile in a column
overlaps every other along x. It still finds the pairs among the moving
shapes, which it is faster at than a tree would be.

A static body moved with `setTransform` takes its shapes to their new place
in the tree at the next step, and whatever slept on it wakes. That costs a
walk over the bodies, because a body cannot tell the world it was moved;
moving platforms want a kinematic body anyway.

## Fast bodies

A step finds what touches from where the bodies are, and then moves them.
A ball ten centimetres across at thirty metres a second covers half a metre
a step, so a wall ten centimetres thick can lie wholly between where it is
and where it will be: touching nothing before, nothing after, and through.
So every body that moved more than half its thinnest extent in a step is
**swept** along the path it took, and if the sweep says it went into the
level, it is put back where it first touched - with its velocity, so the
next step stops or bounces it as it would anything it hit. A body flagged
`bullet` is swept against the other moving bodies too, for a shot that must
hit a crate rather than pass through it. The first touch is found by
conservative advancement (Mirtich), which can never step past a wall
however thin, because no step is longer than the gap in front of it. See
`continuous`.

| Scene | Without the sweep | With it |
| --- | --- | --- |
| a ball 10 cm across into a wall 10 cm thick, at 10, 30, 100 and 300 m/s | through, every time | stopped, every time |
| the same at ten degrees to the wall | through at 100 and 300 m/s | stopped |
| a rod 50 cm long spinning at 20 rad/s, at 10 to 100 m/s | through | stopped |
| a ball dropped from 100 m onto a floor 10 cm thick | fell through it | lands on it |
| a box sliding over tiles at 60 m/s into a thin wall | through | stopped |
| a box 40 cm tall at 10 to 60 m/s into a kerb 10 cm high | slid through it, all 10 cm of it | never more than 1.3 cm into it: it trips over it, as a real one would |
| a shot at 200 m/s at a free-standing plank | through | through, unless it is a `bullet`: then it hits and pushes the plank |

**What the sweep leaves alone** matters as much. A box sliding along a
floor of tiles comes within touching of the next tile's corner at every
seam, and as far as that tile alone can tell, its side is a wall. So a first
touch counts only if the rest of the path would sink the shape deeper than
a graze into what it touched, or bring its core - a small circle deep inside
it - to it. A box at 10, 30 or 60 m/s across a tiled floor keeps all its
speed. Towers, heaps, pendulums and the loaded joints above come out the
same to the bit with the sweep on and off, and a heap of four hundred costs
the same: nothing in it is fast.

Box2D v3 also gives every pair a speculative margin, a manifold that starts
a little before two shapes touch. Here only a body the sweep has just
stopped gets one, for a step, because on a floor of tiles a margin reaches
the next tile's corner before the box is on it, and stops a sliding box dead
at every seam. `Settings.enable_continuous` turns the sweep off.

## A step, on every core

What touches is found once per step. Then the step is cut into
**substeps** - four by default - and each is a whole small step of its own:

| Phase | What | Where |
| --- | --- | --- |
| wake | sleepers a joint drives or somebody pushed | one thread, then every core |
| update boxes | each moving shape's box in the world | every core |
| broad phase | which boxes overlap: moving shapes by a sort and a sweep, the level by its tree | one thread |
| narrow phase | where each pair touches | every core |
| prepare | contacts into constraints, warm-started | one thread |
| colour | contacts and joints into runs that share no body | one thread |
| each substep | forces, one solving pass, move, re-measure the joints, one relaxing pass | every core, per colour |
| restitution | bounces | every core, per colour |
| sweep | fast bodies along their paths, put back where they first touched | every core |
| remember | impulses for next step, begin and end events | one thread |
| sleep | which islands rest, which wake | one thread |

**Why substeps.** A pass of the solver works on a picture of the world taken
when it starts. Eight passes over one picture of a light chain holding a
heavy weight solve, very precisely, the chain as it was - and the weight has
moved on. Four substeps take four pictures, each of the world as it is. This
is Erin Catto's "soft step", Box2D v3's solver, and it is the difference
between these, measured on this package, eight passes per step before and
four substeps now:

| Load | Worst joint gap, before | Now |
| --- | --- | --- |
| ball 26 times a link, on a ten-link chain, knocked | 829 px, came apart | 0.8 px |
| thirty-link chain let fall from level | 1972 px | 3.4 px |
| crates ten times a plank dropped on a light bridge | 60 px | 4.7 px |
| a two-metre pendulum's stretch at the bottom of its swing | 13 mm | 0.7 mm |

**Every constraint is a little bit of a spring** - see `Softness`. Drift is
taken back the way a very stiff, heavily damped spring would take it back,
and a relaxing pass then takes the correction back out of the velocity. A
spring stepped that way cannot add energy however hard it is pulled, which
is what the plainer way - a fixed fraction of the drift fed back as speed -
does to a heavy weight on a light chain until it flies apart. Two crates made
half inside each other slide apart and stop, rather than leaving at speed.

**The two corners of a box on a box are solved together**, which is what
lets a tower stand: solved one after the other, the second corner has the
last word in every pass and a stack of ten leaned until it fell. Twenty-five
now stand straight and fall asleep in under two seconds.

**The solver is coloured.** A contact writes to two bodies, and a body may
be in a dozen contacts, so two contacts that share a body cannot be solved
at once. **Graph colouring** sorts the contacts - and, separately, the
joints - into colours such that no two of one colour share a body that can
move; each colour is then solved with every constraint in it at once, colour
after colour, with a join between. A wall is never written to, so a hundred
crates on one floor are one colour and not a hundred.

**The result is the same on one core and on sixteen.** The pairs come out of
the broad phase in one order, the colouring is greedy in that order, and
within a colour nothing shares memory, so the order jobs run in cannot change
what they compute. The demo and the tests run scenes both ways and compare
every position to the bit.

**It runs on [Fluxion Jobs](https://github.com/kisstp2006/fluxion-jobs)**,
and what that costs is worth knowing. That scheduler is one queue and one
lock, and a job that wakes a sleeping worker costs tens of microseconds; a
body integrates in tens of nanoseconds. So the grains are large - a colour
with fewer than a few hundred constraints is solved on the calling thread -
and a scene of a few hundred bodies never spawns a solver job at all, which
is right for it. On the machine this was written on, six hundred bodies in a
heap take about the same time both ways; the parallel structure is there and
correct, and pays for itself when the scheduler underneath stops waking a
thread per job. The demo prints both timings so the crossover can be seen.

## In a browser

```bash
zig build web          # zig-out/web/index.html and the .wasm beside it
```

Two scenes on a canvas, stepped by the page's animation frame: a heap poured
into a bowl, and a yard of joints - a rope bridge with crates on it, a chain
with a ball, a weight on a spring, and a car the arrow keys drive. The
pointer drags anything through a mouse joint, and bodies that have fallen
asleep are drawn grey. No threads: the scheduler sees the target has none
and has zero workers, and every phase runs on the thread that asked. Nothing
in the library is compiled out and nothing is stubbed. **It runs anywhere
because it does nothing**: no files, no clock, no threads of its own - an
allocator and arithmetic.

## What comes back from a step

- **Positions**, on the bodies: `body.position()`, `body.angle`, and
  `body.transform` for placing a sprite.
- **Events**: `world.beginEvents()` and `world.endEvents()` list every pair
  of shapes that started or stopped touching during the last step, including
  because one was destroyed. A **sensor** shape appears in them and pushes
  nothing: a trigger volume, a pickup radius. A sleeping contact is still
  touching, and ends nothing.
- **Queries**, between steps: `castRay` for the first thing along a line,
  `overlapPoint` for what is under the mouse, `overlapAabb` for everything in
  a box. The level answers from its tree; the shapes that move are asked one
  by one, after a circle round each body has turned most of them away.

## Filtering

Sixteen category bits, a mask, and a group, which is Box2D's scheme:

```zig
const player: physics.Filter = .{ .category = 0b01, .mask = 0b10 };  // touches only category 2
const ragdoll_part: physics.Filter = .{ .group = -7 };               // never touches its own group
```

## What is not here

Each of these is a known piece of work, listed with what it would take.

- **Smooth seams between tiles.** A box sliding slowly over a floor of
  separate tiles catches on the seams: resting a slop into one tile, it
  meets the next tile's side before its top, and the narrow phase, which
  sees one tile at a time, pushes it back. Measured, frictionless: at 0.5
  and 1 m/s the box is stopped dead at a seam, at 3 m/s it loses two thirds
  of its speed, from 10 m/s it skips over them; a ball at 1 m/s loses four
  fifths. On one long slab, nothing. Box2D's answer is chain shapes, a
  one-sided outline built by hand; Jolt and Bullet find the internal edges -
  sides covered by a neighbour - and leave them out of the narrow phase,
  which here would be done when a static shape goes into the tree. Until
  then: merge runs of tiles into longer boxes.
- **Fast bodies against fast bodies, and a pile.** The sweep puts a fast body
  back against the level, and a bullet against the other moving bodies where
  they ended the step; two bullets do not see each other, and a kinematic
  body is not swept. A thin rod already pressed deep into a wall by a pile
  hammering it from behind can still be shoved through: of three hundred
  thin rods fired into a heap against a wall at 30-150 m/s, two.
- **Capsules and rounded polygons.** Each is a row and a column of the narrow
  phase's pairings. A capsule is what a character controller wants.
- **Rolling resistance.** A ball on a slope rolls for ever, so a heap with
  balls in it never quite comes to rest and never sleeps. Box2D v3 adds a
  small torque against rolling for this.
- **Entities.** This package knows nothing about an ECS. The engine keeps a
  `BodyId` in a component and copies transforms out after its `.fixed`
  stage; that integration lives in the engine, not here.

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

Scenes run with a scheduler and every core, and with neither, and pass the
same way - and where it matters, compare every position to the bit. A ball
dropped on a floor lands and stays; a tower of twenty-five stands straight
and sleeps; crates made inside each other slide apart and stop; a ball with
restitution comes back up; masks keep a ghost from the floor; a sensor
reports and pushes nothing; a kinematic platform carries a crate; rays find
the nearest thing. Every joint has a scene - a pendulum that keeps its
length and its swing, a thousand-link chain, a rope that is slack until it
is taut, a spring that bounces at its frequency, a hinge at its limits, a
slider, a weld through a fall, a mouse joint, a car that drives - and five
scenes put joints under loads many times heavier than they are. Sleeping has
a scene for every way in and out of it. Fast bodies have a scene for each
row of the table above, each run with the sweep off as well and required to
fail then; a box crosses a tiled floor at 60 m/s without losing speed; six
hundred shots come out the same with workers and without. The level's tree
answers every ray, point and box exactly as a walk over every shape does,
and a step over seven thousand tiles pairs a resting crate with the tiles
under it and nothing else. The browser build is part of
`zig build test`, so a change that breaks the no-thread path fails here and
not in a browser later.

## Build

```bash
zig build test        # run the test suite, and build the wasm
zig build example     # a heap in a bowl, with every core and with none
zig build web         # the browser example into zig-out/web
zig build docs        # generate API docs into zig-out/docs
```

## Licence

`BSD-2-Clause`. See [LICENSE](LICENSE).
