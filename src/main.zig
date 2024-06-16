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

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

const enable_validation_layers =
    if (builtin.mode == std.builtin.Mode.Debug) true else false;

var allocator: std.mem.Allocator = std.heap.page_allocator;
var vkb: BaseDispatch = undefined;
var vki: InstanceDispatch = undefined;

// NOTE: Default GLFW error handling callback
fn errorCallback(error_code: glfw.ErrorCode, description: [:0]const u8) void {
    std.log.err("glfw: {}: {s}\n", .{ error_code, description });
}

const BaseDispatch = vk.BaseWrapper(&.{.{
    .base_commands = .{
        .createInstance = true,
        .enumerateInstanceExtensionProperties = true,
        .enumerateInstanceLayerProperties = true,
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

fn checkValidationLayerSupport() !bool {
    var layer_count: u32 = 0;

    _ = try vkb.enumerateInstanceLayerProperties(&layer_count, null);
    const available_layers = try allocator.alloc(vk.LayerProperties, layer_count);
    defer allocator.free(available_layers);
    _ = try vkb.enumerateInstanceLayerProperties(&layer_count, available_layers.ptr);

    for (validation_layers) |layer| {
        var layer_found = false;

        for (available_layers) |prop| {
            const len = std.mem.indexOfScalar(u8, &prop.layer_name, 0).?;
            const layer_name = prop.layer_name[0..len];

            if (std.mem.eql(u8, std.mem.span(layer), layer_name)) {
                layer_found = true;
                break;
            }
        }

        if (!layer_found) {
            return false;
        }
    }

    return true;
}

fn getRequiredExtensions() !std.ArrayList([*:0]const u8) {
    const glfw_extensions = glfw.getRequiredInstanceExtensions() orelse return blk: {
        const err = glfw.mustGetError();

        std.log.err(
            "failed to get required vulkan instance extensions: error={s}",
            .{err.description},
        );

        break :blk error.code;
    };

    var instance_extensions = try std.ArrayList([*:0]const u8)
        .initCapacity(allocator, glfw_extensions.len);

    try instance_extensions.appendSlice(glfw_extensions);

    if (enable_validation_layers) {
        try instance_extensions.append(vk.extensions.ext_debug_utils.name);
    }

    if (builtin.os.tag == .macos) {
        try instance_extensions.append(@ptrCast(
            vk.extensions.khr_portability_enumeration.name,
        ));
    }

    var extension_count: u32 = 0;
    _ = try vkb.enumerateInstanceExtensionProperties(null, &extension_count, null);

    const available_extensions = try allocator.alloc(vk.ExtensionProperties, extension_count);
    defer allocator.free(available_extensions);

    _ = try vkb.enumerateInstanceExtensionProperties(
        null,
        &extension_count,
        available_extensions.ptr,
    );

    printAvailableExtensions(available_extensions);
    assertRequiredExtensionsAreSupported(glfw_extensions, available_extensions);

    return instance_extensions;
}

fn createDebugUtilsMessengerEXT(
    instance: vk.Instance,
    p_create_info: *const vk.DebugUtilsMessengerCreateInfoEXT,
    p_allocator: ?*const vk.AllocationCallbacks,
    p_debug_messenger: *vk.DebugUtilsMessengerEXT,
) !vk.Result {
    var result: vk.Result = undefined;

    const maybe_func = @as(
        ?vk.PfnCreateDebugUtilsMessengerEXT,
        @ptrCast(vkb.getInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT")),
    );

    if (maybe_func) |func| {
        result = func(instance, p_create_info, p_allocator, p_debug_messenger);
    } else {
        result = .error_extension_not_present;
    }

    return result;
}

fn destroyDebugUtilsMessengerEXT(
    instance: vk.Instance,
    debug_messenger: vk.DebugUtilsMessengerEXT,
    p_allocator: ?*const vk.AllocationCallbacks,
) void {
    const maybe_func = @as(
        ?vk.PfnDestroyDebugUtilsMessengerEXT,
        @ptrCast(vkb.getInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT")),
    );

    if (maybe_func) |func| {
        func(instance, debug_messenger, p_allocator);
    }
}

const HelloTriangleApplication = struct {
    window: *const glfw.Window,
    instance: vk.Instance,
    debug_messenger: vk.DebugUtilsMessengerEXT,

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
        setupDebugMessenger(self);
    }

    fn vkDebugUtilsMessengerCreateInfo() vk.DebugUtilsMessengerCreateInfoEXT {
        const result: vk.DebugUtilsMessengerCreateInfoEXT = .{
            .message_severity = vk.DebugUtilsMessageSeverityFlagsEXT{
                .verbose_bit_ext = true,
                .warning_bit_ext = true,
                .error_bit_ext = true,
            },
            .message_type = vk.DebugUtilsMessageTypeFlagsEXT{
                .general_bit_ext = true,
                .validation_bit_ext = true,
                .performance_bit_ext = true,
            },
            .pfn_user_callback = debugCallback,
        };

        return result;
    }

    fn setupDebugMessenger(self: *HelloTriangleApplication) void {
        if (enable_validation_layers) {
            const create_info = vkDebugUtilsMessengerCreateInfo();

            if (try createDebugUtilsMessengerEXT(
                self.instance,
                &create_info,
                null,
                &self.debug_messenger,
            ) != .success) {
                @panic("failed to set up debug messenger!");
            }
        }
    }

    fn createInstance(self: *HelloTriangleApplication) !void {
        vkb = try BaseDispatch.load(@as(
            vk.PfnGetInstanceProcAddr,
            @ptrCast(&glfw.getInstanceProcAddress),
        ));

        const instance_extensions = try getRequiredExtensions();
        defer instance_extensions.deinit();

        // TODO: Improve the clarity of this conditional
        if (enable_validation_layers and !(try checkValidationLayerSupport())) {
            @panic("validation layers requested, but not available!");
        }

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
            .enabled_extension_count = @intCast(instance_extensions.items.len),
            .pp_enabled_extension_names = @ptrCast(instance_extensions.items),
        };

        var debug_create_info = vkDebugUtilsMessengerCreateInfo();
        if (enable_validation_layers) {
            create_info.enabled_layer_count = validation_layers.len;
            create_info.pp_enabled_layer_names = &validation_layers;
            create_info.p_next = &debug_create_info;
        } else {
            create_info.enabled_layer_count = 0;
            create_info.pp_enabled_layer_names = null;
            create_info.p_next = null;
        }

        self.instance = try vkb.createInstance(&create_info, null);
        errdefer vki.destroyInstance(self.instance, null);

        vki = try InstanceDispatch.load(self.instance, vkb.dispatch.vkGetInstanceProcAddr);
    }

    fn mainLoop(self: *HelloTriangleApplication) void {
        while (!self.window.shouldClose()) {
            glfw.pollEvents();
        }
    }

    fn cleanup(self: *HelloTriangleApplication) void {
        if (enable_validation_layers) {
            destroyDebugUtilsMessengerEXT(
                self.instance,
                self.debug_messenger,
                null,
            );
        }

        vki.destroyInstance(self.instance, null);
        // TODO: Why does produce an illegal instruction at address error?
        self.window.destroy();
        glfw.terminate();
    }

    fn debugCallback(
        message_severity: vk.DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk.DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: ?*const vk.DebugUtilsMessengerCallbackDataEXT,
        p_user_data: ?*anyopaque,
    ) callconv(vk.vulkan_call_conv) vk.Bool32 {
        std.debug.print("validation layer: {?s}\n", .{p_callback_data.?.p_message});

        if (message_severity.warning_bit_ext or message_severity.error_bit_ext) {
            // Message is important enough to show
        }

        _ = message_type;
        _ = p_user_data;

        return vk.FALSE;
    }
};

pub fn main() !void {
    var app: HelloTriangleApplication = undefined;

    try app.run();
}

// NOTE: Main body function from https://vulkan-tutorial.com/Development_environment
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
