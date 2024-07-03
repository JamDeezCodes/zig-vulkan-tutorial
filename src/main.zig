const std = @import("std");
const builtin = @import("builtin");
const glfw = @import("mach-glfw");
const vk = @import("vulkan");
const mach = @import("mach");

const vert = @embedFile("vert");
const frag = @embedFile("frag");

const assert = std.debug.assert;

const vec4 = mach.math.vec4;
const mat4 = mach.math.mat4x4;
const Vec4 = mach.math.Vec4;
const Mat4x4 = mach.math.Mat4x4;

const width = 800;
const height = 600;
const max_frames_in_flight = 2;
var current_frame: u32 = 0;

const validation_layers = [_][*:0]const u8{
    "VK_LAYER_KHRONOS_validation",
};

const device_extensions = [_][*:0]const u8{
    vk.extensions.khr_swapchain.name,
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
            .enumerateDeviceExtensionProperties = true,
            .getPhysicalDeviceProperties = true,
            .getPhysicalDeviceFeatures = true,
            .getPhysicalDeviceQueueFamilyProperties = true,
            .getPhysicalDeviceSurfaceSupportKHR = true,
            .getPhysicalDeviceSurfaceCapabilitiesKHR = true,
            .getPhysicalDeviceSurfaceFormatsKHR = true,
            .getPhysicalDeviceSurfacePresentModesKHR = true,
            .destroyInstance = true,
            .destroySurfaceKHR = true,
            .getDeviceProcAddr = true,
        },
    },
});

const DeviceDispatch = vk.DeviceWrapper(&.{
    .{
        .device_commands = .{
            .acquireNextImageKHR = true,
            .allocateCommandBuffers = true,
            .beginCommandBuffer = true,
            .cmdBindPipeline = true,
            .cmdBeginRenderPass = true,
            .cmdSetViewport = true,
            .cmdSetScissor = true,
            .cmdDraw = true,
            .cmdEndRenderPass = true,
            .createSwapchainKHR = true,
            .createImageView = true,
            .createShaderModule = true,
            .createRenderPass = true,
            .createPipelineLayout = true,
            .createGraphicsPipelines = true,
            .createFramebuffer = true,
            .createCommandPool = true,
            .createSemaphore = true,
            .createFence = true,
            .destroyDevice = true,
            .destroySwapchainKHR = true,
            .destroyImageView = true,
            .destroyShaderModule = true,
            .destroyPipelineLayout = true,
            .destroyRenderPass = true,
            .destroyPipeline = true,
            .destroyFramebuffer = true,
            .destroyCommandPool = true,
            .destroySemaphore = true,
            .destroyFence = true,
            .deviceWaitIdle = true,
            .endCommandBuffer = true,
            .getDeviceQueue = true,
            .getSwapchainImagesKHR = true,
            .queueSubmit = true,
            .queuePresentKHR = true,
            .resetFences = true,
            .resetCommandBuffer = true,
            .waitForFences = true,
        },
    },
});

const QueueFamilyIndices = struct {
    graphics_family: ?u32 = null,
    present_family: ?u32 = null,
    slice: []u32 = undefined,
};

const SwapChainSupportDetails = struct {
    capabilities: vk.SurfaceCapabilitiesKHR,
    formats: []vk.SurfaceFormatKHR,
    present_modes: []vk.PresentModeKHR,
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
    var layer_count: u32 = undefined;

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

    var extension_count: u32 = undefined;
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

fn querySwapChainSupport(
    app: *HelloTriangleApplication,
    device: vk.PhysicalDevice,
) !SwapChainSupportDetails {
    var result: SwapChainSupportDetails = undefined;

    result.capabilities = try vki.getPhysicalDeviceSurfaceCapabilitiesKHR(device, app.surface);

    var format_count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(device, app.surface, &format_count, null);

    if (format_count != 0) {
        result.formats = try allocator.alloc(vk.SurfaceFormatKHR, format_count);
        _ = try vki.getPhysicalDeviceSurfaceFormatsKHR(device, app.surface, &format_count, result.formats.ptr);
    }

    var present_mode_count: u32 = undefined;
    _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(device, app.surface, &present_mode_count, null);

    if (present_mode_count != 0) {
        result.present_modes = try allocator.alloc(vk.PresentModeKHR, present_mode_count);
        _ = try vki.getPhysicalDeviceSurfacePresentModesKHR(device, app.surface, &present_mode_count, result.present_modes.ptr);
    }

    return result;
}

fn chooseSwapSurfaceFormat(available_formats: []const vk.SurfaceFormatKHR) vk.SurfaceFormatKHR {
    for (available_formats) |format| {
        if (format.format == .b8g8r8a8_srgb and format.color_space == .srgb_nonlinear_khr) {
            return format;
        }
    }

    return available_formats[0];
}

fn chooseSwapPresentMode(available_present_modes: []const vk.PresentModeKHR) vk.PresentModeKHR {
    for (available_present_modes) |present_mode| {
        if (present_mode == .mailbox_khr) {
            return present_mode;
        }
    }

    return vk.PresentModeKHR.fifo_khr;
}

fn chooseSwapExtent(
    app: *HelloTriangleApplication,
    capabilities: *const vk.SurfaceCapabilitiesKHR,
) vk.Extent2D {
    if (capabilities.current_extent.width != std.math.maxInt(u32)) {
        return capabilities.current_extent;
    } else {
        const size = app.window.?.getFramebufferSize();

        var actual_extent = vk.Extent2D{ .width = size.width, .height = size.height };

        actual_extent.width = std.math.clamp(
            actual_extent.width,
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        );
        actual_extent.height = std.math.clamp(
            actual_extent.height,
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        );

        return actual_extent;
    }
}

fn framebufferResizeCallback(window: glfw.Window, _: u32, _: u32) void {
    const maybe_app = window.getUserPointer(HelloTriangleApplication);

    if (maybe_app) |app| app.framebuffer_resized = true;
}

const HelloTriangleApplication = struct {
    window: ?*const glfw.Window = null,
    instance: vk.Instance = .null_handle,
    debug_messenger: vk.DebugUtilsMessengerEXT = .null_handle,
    physical_device: vk.PhysicalDevice = .null_handle,
    device: vk.Device = .null_handle,
    graphics_queue: vk.Queue = .null_handle,
    surface: vk.SurfaceKHR = .null_handle,
    present_queue: vk.Queue = .null_handle,
    swap_chain: vk.SwapchainKHR = .null_handle,
    swap_chain_images: []vk.Image = undefined,
    swap_chain_image_format: vk.Format = .undefined,
    swap_chain_extent: vk.Extent2D = undefined,
    swap_chain_image_views: []vk.ImageView = undefined,
    render_pass: vk.RenderPass = .null_handle,
    pipeline_layout: vk.PipelineLayout = .null_handle,
    graphics_pipeline: vk.Pipeline = .null_handle,
    swap_chain_framebuffers: []vk.Framebuffer = undefined,
    command_pool: vk.CommandPool = .null_handle,
    command_buffers: []vk.CommandBuffer = undefined,
    image_available_semaphores: []vk.Semaphore = undefined,
    render_finished_semaphores: []vk.Semaphore = undefined,
    in_flight_fences: []vk.Fence = undefined,
    framebuffer_resized: bool = false,

    pub fn run(self: *HelloTriangleApplication) !void {
        self.initWindow();
        try self.initVulkan();
        defer self.cleanup();
        try self.mainLoop();
    }

    fn initWindow(self: *HelloTriangleApplication) void {
        if (!glfw.init(.{})) {
            std.log.err("failed to initialize GLFW: {?s}", .{glfw.getErrorString()});
            std.process.exit(1);
        }

        const window = glfw.Window.create(width, height, "Vulkan", null, null, .{
            .client_api = .no_api,
            .resizable = true,
        }) orelse {
            std.log.err("failed to create GLFW window: {?s}", .{glfw.getErrorString()});
            std.process.exit(1);
        };

        self.window = &window;
        self.window.?.setUserPointer(self);
        self.window.?.setFramebufferSizeCallback(framebufferResizeCallback);
    }

    // TODO: Consider a procedural refactoring of this struct and all its member functions, many of which
    // do not seem to see any reuse. Maybe a larger app would see some of these functions being called more
    // than once?
    fn initVulkan(self: *HelloTriangleApplication) !void {
        try self.createInstance();
        self.setupDebugMessenger();
        try self.createSurface();
        try self.pickPhysicalDevice();
        try self.createLogicalDevice();
        try self.createSwapChain();
        try self.createImageViews();
        try self.createRenderPass();
        try self.createGraphicsPipeline();
        try self.createFrameBuffers();
        try self.createCommandPool();
        try self.createCommandBuffers();
        try self.createSyncObjects();
    }

    fn createSyncObjects(self: *HelloTriangleApplication) !void {
        self.image_available_semaphores = try allocator.alloc(vk.Semaphore, max_frames_in_flight);
        self.render_finished_semaphores = try allocator.alloc(vk.Semaphore, max_frames_in_flight);
        self.in_flight_fences = try allocator.alloc(vk.Fence, max_frames_in_flight);

        var semaphore_info = vk.SemaphoreCreateInfo{};

        var fence_info = vk.FenceCreateInfo{
            .flags = .{ .signaled_bit = true },
        };

        for (0..max_frames_in_flight) |i| {
            self.image_available_semaphores[i] = try vkd.createSemaphore(self.device, &semaphore_info, null);
            self.render_finished_semaphores[i] = try vkd.createSemaphore(self.device, &semaphore_info, null);
            self.in_flight_fences[i] = try vkd.createFence(self.device, &fence_info, null);
        }
    }

    fn recordCommandBuffer(
        self: *HelloTriangleApplication,
        command_buffer: vk.CommandBuffer,
        image_index: u32,
    ) !void {
        var begin_info = vk.CommandBufferBeginInfo{
            .flags = .{},
            .p_inheritance_info = null,
        };

        _ = try vkd.beginCommandBuffer(command_buffer, &begin_info);

        var render_pass_info = vk.RenderPassBeginInfo{
            .render_pass = self.render_pass,
            .framebuffer = self.swap_chain_framebuffers[image_index],
            .render_area = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.swap_chain_extent },
            .clear_value_count = 1,
            .p_clear_values = @ptrCast(&vk.ClearValue{ .color = .{ .float_32 = .{ 0, 0, 0, 1 } } }),
        };

        vkd.cmdBeginRenderPass(command_buffer, &render_pass_info, .@"inline");
        vkd.cmdBindPipeline(command_buffer, .graphics, self.graphics_pipeline);

        const viewport = vk.Viewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.swap_chain_extent.width),
            .height = @floatFromInt(self.swap_chain_extent.height),
            .min_depth = 0,
            .max_depth = 1,
        };

        vkd.cmdSetViewport(command_buffer, 0, 1, @ptrCast(&viewport));

        const scissor = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swap_chain_extent,
        };

        vkd.cmdSetScissor(command_buffer, 0, 1, @ptrCast(&scissor));
        vkd.cmdDraw(command_buffer, 3, 1, 0, 0);
        vkd.cmdEndRenderPass(command_buffer);
        _ = try vkd.endCommandBuffer(command_buffer);
    }

    fn createCommandBuffers(self: *HelloTriangleApplication) !void {
        self.command_buffers = try allocator.alloc(vk.CommandBuffer, max_frames_in_flight);

        var alloc_info = vk.CommandBufferAllocateInfo{
            .command_pool = self.command_pool,
            .level = .primary,
            .command_buffer_count = @intCast(self.command_buffers.len),
        };

        _ = try vkd.allocateCommandBuffers(self.device, &alloc_info, @ptrCast(self.command_buffers.ptr));
    }

    fn createCommandPool(self: *HelloTriangleApplication) !void {
        const queue_family_indices = try self.findQueueFamilies(self.physical_device);

        var pool_info = vk.CommandPoolCreateInfo{
            .flags = .{ .reset_command_buffer_bit = true },
            .queue_family_index = queue_family_indices.graphics_family.?,
        };

        self.command_pool = try vkd.createCommandPool(self.device, &pool_info, null);
    }

    fn createFrameBuffers(self: *HelloTriangleApplication) !void {
        self.swap_chain_framebuffers = try allocator.alloc(vk.Framebuffer, self.swap_chain_image_views.len);

        for (self.swap_chain_image_views, self.swap_chain_framebuffers) |image_view, *framebuffer| {
            const framebuffer_info = vk.FramebufferCreateInfo{
                .render_pass = self.render_pass,
                .attachment_count = 1,
                .p_attachments = &.{image_view},
                .width = self.swap_chain_extent.width,
                .height = self.swap_chain_extent.height,
                .layers = 1,
            };

            framebuffer.* = try vkd.createFramebuffer(self.device, &framebuffer_info, null);
        }
    }

    fn createRenderPass(self: *HelloTriangleApplication) !void {
        const color_attachment = vk.AttachmentDescription{
            .format = self.swap_chain_image_format,
            .samples = .{ .@"1_bit" = true },
            .load_op = .clear,
            .store_op = .store,
            .stencil_load_op = .dont_care,
            .stencil_store_op = .dont_care,
            .initial_layout = .undefined,
            .final_layout = .present_src_khr,
        };

        const color_attachment_ref = vk.AttachmentReference{
            .attachment = 0,
            .layout = .color_attachment_optimal,
        };

        const subpass = vk.SubpassDescription{
            .pipeline_bind_point = .graphics,
            .color_attachment_count = 1,
            .p_color_attachments = &.{color_attachment_ref},
        };

        const dependency = vk.SubpassDependency{
            .src_subpass = vk.SUBPASS_EXTERNAL,
            .dst_subpass = 0,
            .src_stage_mask = .{ .color_attachment_output_bit = true },
            .src_access_mask = .{},
            .dst_stage_mask = .{ .color_attachment_output_bit = true },
            .dst_access_mask = .{ .color_attachment_write_bit = true },
        };

        const render_pass_info = vk.RenderPassCreateInfo{
            .attachment_count = 1,
            .p_attachments = &.{color_attachment},
            .subpass_count = 1,
            .p_subpasses = &.{subpass},
            .dependency_count = 1,
            .p_dependencies = @ptrCast(&dependency),
        };

        self.render_pass = try vkd.createRenderPass(self.device, &render_pass_info, null);
    }

    fn createGraphicsPipeline(self: *HelloTriangleApplication) !void {
        //const vert_shader_module = try self.createShaderModule(vert);
        const vert_shader_module = try vkd.createShaderModule(self.device, &.{
            .code_size = vert.len,
            .p_code = @ptrCast(@alignCast(vert)),
        }, null);
        defer vkd.destroyShaderModule(self.device, vert_shader_module, null);

        //const frag_shader_module = try self.createShaderModule(frag);
        const frag_shader_module = try vkd.createShaderModule(self.device, &.{
            .code_size = frag.len,
            .p_code = @ptrCast(@alignCast(frag)),
        }, null);
        defer vkd.destroyShaderModule(self.device, frag_shader_module, null);

        const vert_shader_stage_info = vk.PipelineShaderStageCreateInfo{
            .stage = .{ .vertex_bit = true },
            .module = vert_shader_module,
            .p_name = "main",
        };

        const frag_shader_stage_info = vk.PipelineShaderStageCreateInfo{
            .stage = .{ .fragment_bit = true },
            .module = frag_shader_module,
            .p_name = "main",
        };

        const shader_stages = [_]vk.PipelineShaderStageCreateInfo{ vert_shader_stage_info, frag_shader_stage_info };

        const dynamic_states = [_]vk.DynamicState{ .viewport, .scissor };
        const dynamic_state = vk.PipelineDynamicStateCreateInfo{
            .dynamic_state_count = dynamic_states.len,
            .p_dynamic_states = &dynamic_states,
        };

        const vertex_input_info = vk.PipelineVertexInputStateCreateInfo{
            .vertex_binding_description_count = 0,
            .p_vertex_binding_descriptions = null,
            .vertex_attribute_description_count = 0,
            .p_vertex_attribute_descriptions = null,
        };

        const input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        };

        const viewport = vk.Viewport{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(self.swap_chain_extent.width),
            .height = @floatFromInt(self.swap_chain_extent.height),
            .min_depth = 0,
            .max_depth = 1,
        };

        const scissor = vk.Rect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = self.swap_chain_extent,
        };

        const viewport_state = vk.PipelineViewportStateCreateInfo{
            .viewport_count = 1,
            .p_viewports = &.{viewport},
            .scissor_count = 1,
            .p_scissors = &.{scissor},
        };

        const rasterizer = vk.PipelineRasterizationStateCreateInfo{
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .line_width = 1,
            .cull_mode = .{ .back_bit = true },
            .front_face = .clockwise,
            .depth_bias_enable = vk.FALSE,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
        };

        const multisampling = vk.PipelineMultisampleStateCreateInfo{
            .sample_shading_enable = vk.FALSE,
            .rasterization_samples = .{ .@"1_bit" = true },
            .min_sample_shading = 1,
            .p_sample_mask = null,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        };

        const color_blend_attachment = vk.PipelineColorBlendAttachmentState{
            .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
            //.blend_enable = vk.FALSE,
            //.src_color_blend_factor = .one,
            //.dst_color_blend_factor = .zero,
            .blend_enable = vk.TRUE,
            .src_color_blend_factor = .src_alpha,
            .dst_color_blend_factor = .one_minus_src_alpha,
            .color_blend_op = .add,
            .src_alpha_blend_factor = .one,
            .dst_alpha_blend_factor = .zero,
            .alpha_blend_op = .add,
        };

        const color_blending = vk.PipelineColorBlendStateCreateInfo{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = &.{color_blend_attachment},
            .blend_constants = .{ 0, 0, 0, 0 },
        };

        const pipeline_layout_info = vk.PipelineLayoutCreateInfo{
            .set_layout_count = 0,
            .p_set_layouts = null,
            .push_constant_range_count = 0,
            .p_push_constant_ranges = null,
        };

        self.pipeline_layout = try vkd.createPipelineLayout(self.device, &pipeline_layout_info, null);

        const pipeline_info = vk.GraphicsPipelineCreateInfo{
            .stage_count = 2,
            .p_stages = &shader_stages,
            .p_vertex_input_state = &vertex_input_info,
            .p_input_assembly_state = &input_assembly,
            .p_viewport_state = &viewport_state,
            .p_rasterization_state = &rasterizer,
            .p_multisample_state = &multisampling,
            .p_depth_stencil_state = null,
            .p_color_blend_state = &color_blending,
            .p_dynamic_state = &dynamic_state,
            .layout = self.pipeline_layout,
            .render_pass = self.render_pass,
            .subpass = 0,
            .base_pipeline_handle = .null_handle,
            .base_pipeline_index = 1,
        };

        _ = try vkd.createGraphicsPipelines(
            self.device,
            .null_handle,
            1,
            &.{pipeline_info},
            null,
            @ptrCast(&self.graphics_pipeline),
        );
    }

    // NOTE: For some reason, use of this function produces an incorrect alignment error
    // Is the implicit conversion to []const u8 somehow improper?
    fn createShaderModule(self: *HelloTriangleApplication, code: []const u8) !vk.ShaderModule {
        var create_info = vk.ShaderModuleCreateInfo{
            .code_size = code.len,
            .p_code = @ptrCast(@alignCast(code)),
        };

        const result = try vkd.createShaderModule(self.device, &create_info, null);

        return result;
    }

    fn createImageViews(self: *HelloTriangleApplication) !void {
        self.swap_chain_image_views = try allocator.alloc(vk.ImageView, self.swap_chain_images.len);

        for (self.swap_chain_images, self.swap_chain_image_views) |image, *image_view| {
            var create_info = vk.ImageViewCreateInfo{
                .image = image,
                .view_type = .@"2d",
                .format = self.swap_chain_image_format,
                .components = .{
                    .r = .identity,
                    .g = .identity,
                    .b = .identity,
                    .a = .identity,
                },
                .subresource_range = .{
                    .aspect_mask = .{ .color_bit = true },
                    .base_mip_level = 0,
                    .level_count = 1,
                    .base_array_layer = 0,
                    .layer_count = 1,
                },
            };

            image_view.* = try vkd.createImageView(self.device, &create_info, null);
        }
    }

    fn createSwapChain(self: *HelloTriangleApplication) !void {
        const swap_chain_support = try querySwapChainSupport(self, self.physical_device);
        const surface_format = chooseSwapSurfaceFormat(swap_chain_support.formats);
        const present_mode = chooseSwapPresentMode(swap_chain_support.present_modes);
        const extent = chooseSwapExtent(self, &swap_chain_support.capabilities);
        var image_count = swap_chain_support.capabilities.min_image_count + 1;

        if (swap_chain_support.capabilities.max_image_count > 0 and
            image_count > swap_chain_support.capabilities.max_image_count)
        {
            image_count = swap_chain_support.capabilities.max_image_count;
        }

        var create_info = vk.SwapchainCreateInfoKHR{
            .surface = self.surface,
            .min_image_count = image_count,
            .image_format = surface_format.format,
            .image_color_space = surface_format.color_space,
            .image_extent = extent,
            .image_array_layers = 1,
            .image_usage = .{ .color_attachment_bit = true },
            .image_sharing_mode = .exclusive,
            .pre_transform = swap_chain_support.capabilities.current_transform,
            .composite_alpha = .{ .opaque_bit_khr = true },
            .present_mode = present_mode,
            .clipped = vk.TRUE,
            .old_swapchain = self.swap_chain,
        };

        const indices = try findQueueFamilies(self, self.physical_device);
        defer allocator.free(indices.slice);

        if (indices.graphics_family.? != indices.present_family.?) {
            create_info.image_sharing_mode = .concurrent;
            create_info.queue_family_index_count = 2;
            create_info.p_queue_family_indices = @ptrCast(indices.slice.ptr);
        } else {
            create_info.image_sharing_mode = .exclusive;
            create_info.queue_family_index_count = 0;
            create_info.p_queue_family_indices = null;
        }

        self.swap_chain = try vkd.createSwapchainKHR(self.device, &create_info, null);
        _ = try vkd.getSwapchainImagesKHR(self.device, self.swap_chain, &image_count, null);
        self.swap_chain_images = try allocator.alloc(vk.Image, image_count);
        _ = try vkd.getSwapchainImagesKHR(self.device, self.swap_chain, &image_count, self.swap_chain_images.ptr);
        self.swap_chain_image_format = surface_format.format;
        self.swap_chain_extent = extent;
    }

    fn cleanupSwapChain(self: *HelloTriangleApplication) !void {
        for (self.swap_chain_framebuffers) |framebuffer| {
            vkd.destroyFramebuffer(self.device, framebuffer, null);
        }

        for (self.swap_chain_image_views) |image_view| {
            vkd.destroyImageView(self.device, image_view, null);
        }

        vkd.destroySwapchainKHR(self.device, self.swap_chain, null);
    }

    fn recreateSwapChain(self: *HelloTriangleApplication) !void {
        // NOTE: This call is causing the program to crash, similarly to how it crashes
        // when calling destroy() on cleanup. Something must be incorrect with how we've
        // set up the glfw window
        var size = self.window.?.getFramebufferSize();

        while (size.width == 0 or size.height == 0) {
            size = self.window.?.getFramebufferSize();
            glfw.waitEvents();
        }

        try vkd.deviceWaitIdle(self.device);
        try self.createSwapChain();
        try self.createImageViews();
        try self.createFrameBuffers();
    }

    fn createSurface(self: *HelloTriangleApplication) !void {
        if (glfw.createWindowSurface(
            self.instance,
            self.window.?.*,
            null,
            &self.surface,
        ) != @intFromEnum(vk.Result.success)) {
            @panic("failed to create window surface!");
        }
    }

    fn createLogicalDevice(self: *HelloTriangleApplication) !void {
        const indices = try findQueueFamilies(self, self.physical_device);
        defer allocator.free(indices.slice);

        var queue_create_infos = try allocator.alloc(vk.DeviceQueueCreateInfo, indices.slice.len);
        defer allocator.free(queue_create_infos);
        const queue_priority: [1]f32 = .{1};

        for (indices.slice, 0..) |queue_family, i| {
            queue_create_infos.ptr[i] = .{
                .queue_family_index = queue_family,
                .queue_count = 1,
                .p_queue_priorities = &queue_priority,
            };
        }

        const device_features: vk.PhysicalDeviceFeatures = .{};

        var create_info: vk.DeviceCreateInfo = .{
            .queue_create_info_count = @intCast(queue_create_infos.len),
            .p_queue_create_infos = queue_create_infos.ptr,
            .p_enabled_features = &device_features,
            .enabled_extension_count = device_extensions.len,
            .pp_enabled_extension_names = &device_extensions,
        };

        var layers = try std.ArrayList([*:0]const u8)
            .initCapacity(allocator, validation_layers.len + 2);
        defer layers.deinit();

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
            try layers.append(@ptrCast(
                vk.extensions.khr_portability_subset.name,
            ));
            try layers.append(@ptrCast(
                vk.extensions.khr_portability_enumeration.name,
            ));
        }

        if (enable_validation_layers) {
            // NOTE: pp_enabled_layer names are inherited and therefore deprecated
            //create_info.enabled_layer_count = @intCast(layers.items.len);
            //create_info.pp_enabled_layer_names = @ptrCast(layers.items);
        } else {
            create_info.enabled_layer_count = 0;
        }

        self.device = try vki.createDevice(self.physical_device, &create_info, null);
        vkd = try DeviceDispatch.load(self.device, vki.dispatch.vkGetDeviceProcAddr);
        self.graphics_queue = vkd.getDeviceQueue(self.device, indices.graphics_family.?, 0);
        self.present_queue = vkd.getDeviceQueue(self.device, indices.present_family.?, 0);
    }

    fn pickPhysicalDevice(self: *HelloTriangleApplication) !void {
        var device_count: u32 = undefined;

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
                // TODO: This check should be baked into rateDeviceSuitability()
                if (try deviceIsSuitable(self, device.key_ptr.*)) {
                    best_score = device.value_ptr.*;

                    self.physical_device = device.key_ptr.*;
                }
            }
        }

        if (self.physical_device == .null_handle) {
            @panic("failed to find suitable GPU!");
        }
    }

    fn deviceIsSuitable(self: *HelloTriangleApplication, device: vk.PhysicalDevice) !bool {
        const indices = try findQueueFamilies(self, device);
        defer allocator.free(indices.slice);

        const extensions_supported = try checkDeviceExtensionSupport(device);

        var swap_chain_adequate = false;

        if (extensions_supported) {
            const swap_chain_support = try querySwapChainSupport(self, device);
            defer allocator.free(swap_chain_support.formats);
            defer allocator.free(swap_chain_support.present_modes);
            swap_chain_adequate = swap_chain_support.formats.len > 0 and
                swap_chain_support.present_modes.len > 0;
        }

        return indices.graphics_family != null and
            indices.present_family != null and
            extensions_supported and
            swap_chain_adequate;
    }

    fn checkDeviceExtensionSupport(device: vk.PhysicalDevice) !bool {
        var result = true;

        var extension_count: u32 = undefined;
        _ = try vki.enumerateDeviceExtensionProperties(device, null, &extension_count, null);
        const available_extensions = try allocator.alloc(vk.ExtensionProperties, extension_count);
        defer allocator.free(available_extensions);
        _ = try vki.enumerateDeviceExtensionProperties(device, null, &extension_count, available_extensions.ptr);

        for (device_extensions) |required_extension| {
            var extension_is_available = false;

            for (available_extensions) |available_extension| {
                const len = std.mem.indexOfScalar(u8, &available_extension.extension_name, 0).?;
                const extension_name = available_extension.extension_name[0..len];

                if (std.mem.eql(u8, extension_name, std.mem.span(required_extension))) {
                    extension_is_available = true;
                    break;
                }
            }

            if (!extension_is_available) {
                result = false;
                break;
            }
        }

        return result;
    }

    fn findQueueFamilies(self: *HelloTriangleApplication, device: vk.PhysicalDevice) !QueueFamilyIndices {
        var indices: QueueFamilyIndices = .{};

        var queue_family_count: u32 = undefined;
        vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, null);

        const queue_families = try allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
        defer allocator.free(queue_families);

        vki.getPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.ptr);

        var i: u32 = 0;
        for (queue_families) |family| {
            if (family.queue_flags.graphics_bit) {
                indices.graphics_family = i;
            }

            const present_support = try vki.getPhysicalDeviceSurfaceSupportKHR(device, i, self.surface);

            if (present_support == vk.TRUE) {
                indices.present_family = i;
            }

            // TODO: Determine the best way to prevent reusing the same index for each family
            if (indices.graphics_family != null and
                indices.present_family != null)
            {
                break;
            }

            i += 1;
        }

        // NOTE: It appears that in newer versions of Vulkan, using the same queue family index across
        // multiple create infos would be regarded as bad practice, and the following Validation Error(s)
        // are produced when doing so to point this out:
        // ADDENDUM: Using the same index multiple times causes the app to fail altogether outside of
        // macos (MoltenVK), confirming the above
        //
        // Validation Error: [ VUID-VkDeviceCreateInfo-queueFamilyIndex-02802 ] Object 0: handle = 0x6000026d5880, type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
        //      | MessageID = 0x29498778 | vkCreateDevice(): pCreateInfo->pQueueCreateInfos[1].queueFamilyIndex (0) is not unique and was also used in
        //      pCreateInfo->pQueueCreateInfos[0]. The Vulkan spec states: The queueFamilyIndex member of each element of pQueueCreateInfos must be unique
        //      within pQueueCreateInfos , except that two members can share the same queueFamilyIndex if one describes protected-capable queues and one
        //      describes queues that are not protected-capable
        //      (https://vulkan.lunarg.com/doc/view/1.3.283.0/mac/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-queueFamilyIndex-02802)
        //
        // Validation Error: [ VUID-VkDeviceCreateInfo-pQueueCreateInfos-06755 ] Object 0: handle = 0x6000026d5880, type = VK_OBJECT_TYPE_PHYSICAL_DEVICE;
        //      | MessageID = 0x4180bcf6 | vkCreateDevice(): pCreateInfo Total queue count requested from queue family index 0 is 2, which is greater than
        //      queue count available in the queue family (1). The Vulkan spec states: If multiple elements of pQueueCreateInfos share the same queueFamilyIndex,
        //      the sum of their queueCount members must be less than or equal to the queueCount member of the VkQueueFamilyProperties structure, as returned
        //      by vkGetPhysicalDeviceQueueFamilyProperties in the pQueueFamilyProperties[queueFamilyIndex]
        //      (https://vulkan.lunarg.com/doc/view/1.3.283.0/mac/1.3-extensions/vkspec.html#VUID-VkDeviceCreateInfo-pQueueCreateInfos-06755)
        //
        // Here we store a slice of one or more unique indices for later use, ensuring we do not encounter
        // the above validation errors
        if (indices.graphics_family) |graphics_family| {
            if (indices.present_family) |present_family| {
                if (graphics_family == present_family) {
                    indices.slice = try allocator.alloc(u32, 1);
                    indices.slice[0] = graphics_family;
                } else {
                    indices.slice = try allocator.alloc(u32, 2);
                    indices.slice[0] = graphics_family;
                    indices.slice[1] = present_family;
                }
            } else unreachable;
        } else unreachable;

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

    fn mainLoop(self: *HelloTriangleApplication) !void {
        while (!self.window.?.shouldClose()) {
            glfw.pollEvents();
            try self.drawFrame();
        }

        try vkd.deviceWaitIdle(self.device);
    }

    fn drawFrame(self: *HelloTriangleApplication) !void {
        _ = try vkd.waitForFences(self.device, 1, @ptrCast(&self.in_flight_fences[current_frame]), vk.TRUE, std.math.maxInt(u64));

        const image_result = try vkd.acquireNextImageKHR(
            self.device,
            self.swap_chain,
            std.math.maxInt(u64),
            self.image_available_semaphores[current_frame],
            .null_handle,
        );

        if (image_result.result == .error_out_of_date_khr) {
            try self.recreateSwapChain();
            return;
        } else if (image_result.result != .success and image_result.result != .suboptimal_khr) {
            @panic("failed to acquire swap chain image!");
        }

        _ = try vkd.resetFences(self.device, 1, @ptrCast(&self.in_flight_fences[current_frame]));
        _ = try vkd.resetCommandBuffer(self.command_buffers[current_frame], .{});
        _ = try self.recordCommandBuffer(self.command_buffers[current_frame], image_result.image_index);

        const wait_semaphores = [_]vk.Semaphore{self.image_available_semaphores[current_frame]};
        const wait_stages = [_]vk.PipelineStageFlags{.{ .color_attachment_output_bit = true }};
        const signal_semaphores = [_]vk.Semaphore{self.render_finished_semaphores[current_frame]};
        var submit_info = vk.SubmitInfo{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = &wait_semaphores,
            .p_wait_dst_stage_mask = &wait_stages,
            .command_buffer_count = 1,
            .p_command_buffers = @ptrCast(&self.command_buffers[current_frame]),
            .signal_semaphore_count = 1,
            .p_signal_semaphores = &signal_semaphores,
        };

        try vkd.queueSubmit(self.graphics_queue, 1, @ptrCast(&submit_info), self.in_flight_fences[current_frame]);

        const swap_chains = [_]vk.SwapchainKHR{self.swap_chain};
        const present_info = vk.PresentInfoKHR{
            .wait_semaphore_count = 1,
            .p_wait_semaphores = &signal_semaphores,
            .swapchain_count = 1,
            .p_swapchains = &swap_chains,
            .p_image_indices = @ptrCast(&image_result.image_index),
            .p_results = null,
        };

        const present_result = try vkd.queuePresentKHR(self.present_queue, &present_info);

        if (present_result == .error_out_of_date_khr or
            present_result == .suboptimal_khr or
            self.framebuffer_resized)
        {
            self.framebuffer_resized = false;
            try self.recreateSwapChain();
        } else if (present_result != .success) {
            @panic("failed to present swap chain image");
        }

        current_frame = (current_frame + 1) % max_frames_in_flight;
    }

    fn cleanup(self: *HelloTriangleApplication) void {
        try self.cleanupSwapChain();

        allocator.free(self.command_buffers);
        allocator.free(self.image_available_semaphores);
        allocator.free(self.render_finished_semaphores);
        allocator.free(self.in_flight_fences);

        vkd.destroyPipeline(self.device, self.graphics_pipeline, null);
        vkd.destroyPipelineLayout(self.device, self.pipeline_layout, null);
        vkd.destroyRenderPass(self.device, self.render_pass, null);

        for (0..max_frames_in_flight) |i| {
            vkd.destroySemaphore(self.device, self.image_available_semaphores[i], null);
            vkd.destroySemaphore(self.device, self.render_finished_semaphores[i], null);
            vkd.destroyFence(self.device, self.in_flight_fences[i], null);
        }

        vkd.destroyCommandPool(self.device, self.command_pool, null);
        vkd.destroyDevice(self.device, null);

        if (enable_validation_layers) {
            destroyDebugUtilsMessengerEXT(
                self.instance,
                self.debug_messenger,
                null,
            );
        }

        vki.destroySurfaceKHR(self.instance, self.surface, null);
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

    var extension_count: u32 = undefined;
    _ = try base_dispatch.enumerateInstanceExtensionProperties(null, &extension_count, null);

    const matrix: Mat4x4 = undefined;
    const vec = Vec4.splat(0);

    _ = matrix.mulVec(&vec);

    defer window.destroy();

    while (!window.shouldClose()) {
        glfw.pollEvents();
    }
}
