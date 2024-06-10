const std = @import("std");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const mach = @import("mach");

const vec4 = mach.math.vec4;
const mat4 = mach.math.mat4x4;
const Vec4 = mach.math.Vec4;
const Mat4x4 = mach.math.Mat4x4;

const width = 800;
const height = 600;

/// Default GLFW error handling callback
fn errorCallback(error_code: glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}\n", .{ error_code, description });
}

const BaseDispatch = vk.BaseWrapper(&.{.{
    .base_commands = .{
        .createInstance = true,
        .enumerateInstanceExtensionProperties = true,
        .getInstanceProcAddr = true,
    },
}});

const HelloTriangleApplication = struct {
    window: *const glfw.Window,

    pub fn run(self: *HelloTriangleApplication) !void {
        self.initWindow();
        defer self.cleanup();
        self.initVulkan();
        self.mainLoop();
    }

    fn initWindow(self: *HelloTriangleApplication) void {
        if (!glfw.init(.{})) {
            std.log.err("failed to initialize GLFW: {?s}", .{glfw.getErrorString()});
            std.process.exit(1);
        }

        const window = glfw.Window.create(width, height, "Vulkan", null, null, .{
            .client_api = .no_api,
            .resizable = false,
        }) orelse {
            std.log.err("failed to create GLFW window: {?s}", .{glfw.getErrorString()});
            std.process.exit(1);
        };

        self.window = &window;
    }

    fn initVulkan(_: *HelloTriangleApplication) void {}

    fn mainLoop(self: *HelloTriangleApplication) void {
        while (!self.window.shouldClose()) {
            glfw.pollEvents();
        }
    }

    fn cleanup(self: *HelloTriangleApplication) void {
        self.window.destroy();
        glfw.terminate();
    }
};

pub fn main() !void {
    var app: HelloTriangleApplication = undefined;

    try app.run();
}

pub fn _main() !void {
    glfw.setErrorCallback(errorCallback);

    if (!glfw.init(.{})) {
        std.log.err("failed to initialize GLFW: {?s}", .{glfw.getErrorString()});
        std.process.exit(1);
    }

    defer glfw.terminate();

    const extent = vk.Extent2D{ .width = 800, .height = 600 };

    const window = glfw.Window.create(extent.width, extent.height, "Vulkan window", null, null, .{
        .client_api = .no_api,
    }) orelse {
        std.log.err("failed to create GLFW window: {?s}", .{glfw.getErrorString()});
        std.process.exit(1);
    };

    const base_dispatch = try BaseDispatch.load(@as(
        vk.PfnGetInstanceProcAddr,
        @ptrCast(&glfw.getInstanceProcAddress),
    ));

    var extension_count: u32 = 0;
    _ = try base_dispatch.enumerateInstanceExtensionProperties(null, &extension_count, null);

    const matrix: Mat4x4 = undefined;
    const vec = Vec4.splat(0);

    _ = matrix.mulVec(&vec);

    defer window.destroy();

    while (!window.shouldClose()) {
        glfw.pollEvents();
    }
}
