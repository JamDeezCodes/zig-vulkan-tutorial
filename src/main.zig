const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const mach = @import("mach");

const assert = std.debug.assert;

const vec4 = mach.math.vec4;
const mat4 = mach.math.mat4x4;
const Vec4 = mach.math.Vec4;
const Mat4x4 = mach.math.Mat4x4;

const width = 800;
const height = 600;

var allocator: std.mem.Allocator = std.heap.page_allocator;
var base: BaseDispatch = undefined;
var instance: InstanceDispatch = undefined;

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

const InstanceDispatch = vk.InstanceWrapper(&.{
    .{
        .instance_commands = .{
            .destroyInstance = true,
        },
    },
});

fn printAvailableExtensions(available_extensions: []vk.ExtensionProperties) void {
    std.debug.print("available extensions:\n", .{});

    for (available_extensions) |extension| {
        const len = std.mem.indexOfScalar(u8, &extension.extension_name, 0).?;
        const extension_name = extension.extension_name[0..len];
        std.debug.print("\t{s}\n", .{extension_name});
    }
}

fn assertRequiredExtensionsAreSupported(
    glfw_extensions: [][*:0]const u8,
    available_extensions: []vk.ExtensionProperties,
) void {
    for (glfw_extensions) |glfw_extension| {
        var extension_is_available = false;

        for (available_extensions) |extension| {
            const len = std.mem.indexOfScalar(u8, &extension.extension_name, 0).?;
            const extension_name = extension.extension_name[0..len];

            if (std.mem.eql(u8, extension_name, std.mem.span(glfw_extension))) {
                extension_is_available = true;

                break;
            }
        }

        assert(extension_is_available);
    }
}

const HelloTriangleApplication = struct {
    window: *const glfw.Window,
    instance: vk.Instance,

    pub fn run(self: *HelloTriangleApplication) !void {
        self.initWindow();
        try self.initVulkan();
        defer self.cleanup();
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

    fn initVulkan(self: *HelloTriangleApplication) !void {
        try createInstance(self);
    }

    fn createInstance(self: *HelloTriangleApplication) !void {
        base = try BaseDispatch.load(@as(
            vk.PfnGetInstanceProcAddr,
            @ptrCast(&glfw.getInstanceProcAddress),
        ));

        const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return blk: {
            const err = glfw.mustGetError();

            std.log.err(
                "failed to get required vulkan instance extensions: error={s}",
                .{err.description},
            );

            break :blk error.code;
        };

        var instance_extensions = try std.ArrayList([*:0]const u8)
            .initCapacity(allocator, glfw_extensions.len + 1);

        defer instance_extensions.deinit();
        try instance_extensions.appendSlice(glfw_extensions);

        if (builtin.os.tag == .macos) {
            try instance_extensions.append(@ptrCast(
                vk.extensions.khr_portability_enumeration.name,
            ));
        }

        var extension_count: u32 = 0;
        _ = try base.enumerateInstanceExtensionProperties(null, &extension_count, null);

        const available_extensions = try allocator.alloc(vk.ExtensionProperties, extension_count);
        defer allocator.free(available_extensions);

        _ = try base.enumerateInstanceExtensionProperties(
            null,
            &extension_count,
            available_extensions.ptr,
        );

        printAvailableExtensions(available_extensions);
        assertRequiredExtensionsAreSupported(glfw_extensions, available_extensions);

        const app_info = vk.ApplicationInfo{
            .p_application_name = "Hello Triangle",
            .application_version = vk.makeApiVersion(0, 0, 0, 0),
            .p_engine_name = "No Engine",
            .engine_version = vk.makeApiVersion(0, 0, 0, 0),
            .api_version = vk.makeApiVersion(0, 1, 3, 0),
        };

        var create_info = vk.InstanceCreateInfo{
            .flags = if (builtin.os.tag == .macos) .{
                .enumerate_portability_bit_khr = true,
            } else .{},
            .p_application_info = &app_info,
            .enabled_layer_count = 0,
            .enabled_extension_count = @intCast(instance_extensions.items.len),
            .pp_enabled_extension_names = @ptrCast(instance_extensions.items),
        };

        self.instance = try base.createInstance(&create_info, null);
        errdefer instance.destroyInstance(self.instance, null);

        instance = try InstanceDispatch.load(self.instance, base.dispatch.vkGetInstanceProcAddr);
    }

    fn mainLoop(self: *HelloTriangleApplication) void {
        while (!self.window.shouldClose()) {
            glfw.pollEvents();
        }
    }

    fn cleanup(self: *HelloTriangleApplication) void {
        instance.destroyInstance(self.instance, null);
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
