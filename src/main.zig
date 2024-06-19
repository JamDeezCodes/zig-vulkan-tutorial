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
var vkd: DeviceDispatch = undefined;

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
            .createDevice = true,
            .enumeratePhysicalDevices = true,
            .getPhysicalDeviceProperties = true,
            .getPhysicalDeviceFeatures = true,
            .getPhysicalDeviceQueueFamilyProperties = true,
            .destroyInstance = true,
            .getDeviceProcAddr = true,
        },
    },
});

const DeviceDispatch = vk.DeviceWrapper(&.{
    .{
        .device_commands = .{
            .destroyDevice = true,
            .getDeviceQueue = true,
        },
    },
});

const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
};

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
    window: ?*const glfw.Window = null,
    instance: vk.Instance = .null_handle,
    debug_messenger: vk.DebugUtilsMessengerEXT = .null_handle,
    physical_device: vk.PhysicalDevice = .null_handle,
    device: vk.Device = .null_handle,
    graphics_queue: vk.Queue = .null_handle,

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
        try self.createInstance();
        self.setupDebugMessenger();
        try self.pickPhysicalDevice();
        try self.createLogicalDevice();
    }

    fn createLogicalDevice(self: *HelloTriangleApplication) !void {
        const indices = try findQueueFamilies(self.physical_device);

        const queue_priority: [1]f32 = .{1};
        var queue_create_info: [1]vk.DeviceQueueCreateInfo = .{.{
            .queue_family_index = indices.graphics_family.?,
            .queue_count = 1,
            .p_queue_priorities = &queue_priority,
        }};

        const device_features: vk.PhysicalDeviceFeatures = .{};

        var create_info: vk.DeviceCreateInfo = .{
            .p_queue_create_infos = &queue_create_info,
            .queue_create_info_count = 1,
            .p_enabled_features = &device_features,
            .enabled_extension_count = 0,
        };

        var device_extensions = try std.ArrayList([*:0]const u8)
            .initCapacity(allocator, validation_layers.len);
        defer device_extensions.deinit();

        // NOTE: It seems that even though pp_enabled_layer_names is deprecated for devices,
        // a Validation Error is produced with or without appending the portability
        // extensions here. The same error still presents without the portability extensions
        // appended, but then the deprecation message goes away. Maybe this Validation Error
        // shows erroneously with 1.3.283 but the portability extensions would be required
        // for backwards compatibility
        //
        // loader_create_device_chain: Using deprecated and ignored 'ppEnabledLayerNames' member of 'VkDeviceCreateInfo' when creating a Vulkan device.
        // Validation Error: [ VUID-VkDeviceCreateInfo-pProperties-04451 ] Object ... vkCreateDevice():
        // VK_KHR_portability_subset must be enabled because physical device VkPhysicalDevice 0x60000269b5a0[] supports it. The Vulkan spec states:
        // If the VK_KHR_portability_subset extension is included in pProperties of vkEnumerateDeviceExtensionProperties,
        // ppEnabledExtensionNames must include "VK_KHR_portability_subset"
        if (builtin.os.tag == .macos) {
            try device_extensions.append(@ptrCast(
                vk.extensions.khr_portability_subset.name,
            ));
            try device_extensions.append(@ptrCast(
                vk.extensions.khr_portability_enumeration.name,
            ));
        }

        if (enable_validation_layers) {
            create_info.enabled_layer_count = @intCast(device_extensions.items.len);
            create_info.pp_enabled_layer_names = @ptrCast(device_extensions.items);
        } else {
            create_info.enabled_layer_count = 0;
        }

        self.device = try vki.createDevice(self.physical_device, &create_info, null);
        vkd = try DeviceDispatch.load(self.device, vki.dispatch.vkGetDeviceProcAddr);
        self.graphics_queue = vkd.getDeviceQueue(self.device, indices.graphics_family.?, 0);
    }

    fn pickPhysicalDevice(self: *HelloTriangleApplication) !void {
        var device_count: u32 = 0;

        _ = try vki.enumeratePhysicalDevices(self.instance, &device_count, null);

        if (device_count == 0) @panic("failed to find GPUs with Vulkan support!");

        const devices = try allocator.alloc(vk.PhysicalDevice, device_count);
        defer allocator.free(devices);

        _ = try vki.enumeratePhysicalDevices(self.instance, &device_count, devices.ptr);

        var candidates = std.AutoHashMap(vk.PhysicalDevice, i32).init(allocator);
        defer candidates.deinit();

        for (devices) |device| {
            const score = rateDeviceSuitability(device);
            try candidates.put(device, score);
        }

        var it = candidates.iterator();
        var best_score: i32 = 0;
        while (it.next()) |device| {
            if (device.value_ptr.* > best_score) {
                if (try deviceIsSuitable(device.key_ptr.*)) {
                    best_score = device.value_ptr.*;

                    self.physical_device = device.key_ptr.*;
                }
            }
        }

        if (self.physical_device == .null_handle) {
            @panic("failed to find suitable GPU!");
        }
    }

    fn deviceIsSuitable(device: vk.PhysicalDevice) !bool {
        const indices = try findQueueFamilies(device);

        return indices.graphics_family != null;
    }

    fn findQueueFamilies(device: vk.PhysicalDevice) !QueueFamilyIndices {
        var indices: QueueFamilyIndices = .{};

        var queue_family_count: u32 = 0;
        vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

        const queue_families = try allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
        defer allocator.free(queue_families);

        vki.getPhysicalDeviceQueueFamilyProperties(
            device,
            &queue_family_count,
            queue_families.ptr,
        );

        var i: u32 = 0;
        for (queue_families) |family| {
            if (family.queue_flags.graphics_bit) {
                indices.graphics_family = i;
                break;
            }

            // NOTE: The tutorial adds this check to break out if the graphics family was
            // set, but this seems pointless when we can do it inside the assignment block?
            //if (indices.graphics_family) |_| {
            //    break;
            //}

            i += 1;
        }

        return indices;
    }

    fn rateDeviceSuitability(device: vk.PhysicalDevice) i32 {
        var result: i32 = 0;

        const device_props = vki.getPhysicalDeviceProperties(device);
        const device_features = vki.getPhysicalDeviceFeatures(device);

        if (device_props.device_type == .discrete_gpu) {
            result += 1000;
        }

        result += @intCast(device_props.limits.max_image_dimension_2d);

        // NOTE: My laptop does not support geometry shaders
        // if (device_features.geometry_shader != vk.TRUE) {
        if (device_features.tessellation_shader != vk.TRUE) {
            result = 0;
        }

        return result;
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
        while (!self.window.?.shouldClose()) {
            glfw.pollEvents();
        }
    }

    fn cleanup(self: *HelloTriangleApplication) void {
        vkd.destroyDevice(self.device, null);

        if (enable_validation_layers) {
            destroyDebugUtilsMessengerEXT(
                self.instance,
                self.debug_messenger,
                null,
            );
        }

        vki.destroyInstance(self.instance, null);
        // TODO: Why does produce an illegal instruction at address error?
        self.window.?.destroy();
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
    var app: HelloTriangleApplication = .{};

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
