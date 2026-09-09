// SPDX-License-Identifier: BSD-2-Clause

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = module(b, target, optimize);

    // zig build test
    const tests = b.addTest(.{ .name = "fluxion-physics-tests", .root_module = mod });
    const test_step = b.step("test", "Run the library test suite");
    test_step.dependOn(&b.addRunArtifact(tests).step);

    // zig build example
    //
    // Headless: it opens no window and needs no GPU, because a physics step
    // is arithmetic and the picture is somebody else's job. What it prints is
    // the same scene stepped with every core and with none, and whether the
    // two agree to the bit.
    const example_mod = b.createModule(.{
        .root_source_file = b.path("examples/demo.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{.{ .name = "fluxion_physics", .module = mod }},
    });
    const example = b.addExecutable(.{ .name = "fluxion-physics-demo", .root_module = example_mod });
    b.installArtifact(example);

    const run_example = b.addRunArtifact(example);
    run_example.step.dependOn(b.getInstallStep());
    if (b.args) |args| run_example.addArgs(args);
    b.step("example", "Build and run the demo program").dependOn(&run_example.step);

    test_step.dependOn(&b.addRunArtifact(b.addTest(.{
        .name = "fluxion-physics-demo-tests",
        .root_module = example_mod,
    })).step);

    // zig build web -> zig-out/web/{index.html, fluxion-physics-web.wasm}
    //
    // The same library in a browser: no threads, so the scheduler has no
    // workers and the page's animation frame does the stepping. Built for
    // wasm32-freestanding whatever -Dtarget says, because that is the only
    // target a browser loads.
    const wasm_target = b.resolveTargetQuery(.{ .cpu_arch = .wasm32, .os_tag = .freestanding });
    const wasm_mod = module(b, wasm_target, .ReleaseSmall);
    const web_mod = b.createModule(.{
        .root_source_file = b.path("examples/web.zig"),
        .target = wasm_target,
        .optimize = .ReleaseSmall,
        .imports = &.{.{ .name = "fluxion_physics", .module = wasm_mod }},
    });
    const web = b.addExecutable(.{ .name = "fluxion-physics-web", .root_module = web_mod });
    web.entry = .disabled;
    web.rdynamic = true;

    const web_step = b.step("web", "Build the browser example into zig-out/web");
    web_step.dependOn(&b.addInstallArtifact(web, .{
        .dest_dir = .{ .override = .{ .custom = "web" } },
    }).step);
    web_step.dependOn(&b.addInstallFile(b.path("examples/web/index.html"), "web/index.html").step);

    // The browser build is part of the test step: a change that breaks the
    // no-thread, no-operating-system path should fail here, not in a browser
    // later.
    test_step.dependOn(web_step);

    // zig build docs -> zig-out/docs
    const docs_lib = b.addLibrary(.{ .name = "fluxion-physics", .root_module = mod });
    b.step("docs", "Generate API documentation into zig-out/docs").dependOn(&b.addInstallDirectory(.{
        .source_dir = docs_lib.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    }).step);
}

/// The importable module, for one target. Consumers do:
///   const physics = @import("fluxion_physics");
fn module(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Module {
    const math = b.dependency("fluxion_math", .{ .target = target, .optimize = optimize });
    const jobs = b.dependency("fluxion_jobs", .{ .target = target, .optimize = optimize });
    const ident = b.dependency("fluxion_id", .{ .target = target, .optimize = optimize });

    // The wasm module gets a different name so the two can live in one build
    // graph; a consumer only ever sees the one for its own target.
    return b.addModule(b.fmt("fluxion_physics{s}", .{
        if (target.result.cpu.arch == .wasm32) "_wasm" else "",
    }), .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "fluxion_math", .module = math.module("fluxion_math") },
            .{ .name = "fluxion_jobs", .module = jobs.module("fluxion_jobs") },
            .{ .name = "fluxion_id", .module = ident.module("fluxion_id") },
        },
    });
}
