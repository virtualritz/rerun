struct FrameUniformBuffer {
    view_from_world: mat4x3f,
    projection_from_view: mat4x4f,
    projection_from_world: mat4x4f,

    /// Camera position in world space.
    camera_position: vec3f,

    /// For perspective: Multiply this with a camera distance to get a measure of how wide a pixel is in world units.
    /// For orthographic: This is the world size value, independent of distance.
    pixel_world_size_from_camera_distance: f32,

    /// Camera direction in world space.
    /// Same as -vec3f(view_from_world[0].z, view_from_world[1].z, view_from_world[2].z)
    camera_forward: vec3f,

    /// How many pixels there are per point.
    /// I.e. the UI zoom factor.
    pixels_from_point: f32,

    /// (tan(fov_y / 2) * aspect_ratio, tan(fov_y /2)), i.e. half ratio of screen dimension to screen distance in x & y.
    /// Both values are set to f32max for orthographic projection
    tan_half_fov: vec2f,

    /// re_renderer defined device tier.
    device_tier: u32,

    /// boolean (0/1): set to true for snapshot tests to minimize
    /// GPU/driver-specific stuff like alpha-to-coverage.
    deterministic_rendering: u32,

    /// Screen resolution in pixels.
    framebuffer_resolution: vec2f,

    /// Focal length in pixels: `framebuffer_resolution / (2 * tan_half_fov)`.
    /// Zero for orthographic projection.
    focal_length_in_pixels: vec2f,
};

@group(0) @binding(0)
var<uniform> frame: FrameUniformBuffer;

@group(0) @binding(1)
var nearest_sampler_repeat: sampler;
@group(0) @binding(2)
var nearest_sampler_clamped: sampler;
@group(0) @binding(3)
var trilinear_sampler_repeat: sampler;

// Screen-space ambient occlusion, one value per pixel in `.r` (akatela
// SPEC-123). A 1x1 white texture when the view has none. Read it with
// `textureLoad` and a clamped coordinate: see `occlusion_at` in
// `instanced_mesh_common.wgsl`.
@group(0) @binding(4)
var occlusion_texture: texture_2d<f32>;

// The horizon method's bent normal (akatela SPEC-123 D3a): `.rgb` the
// view-space direction mapped to [0, 1], `.a` 1 where it was computed. An
// all-zero texture when the view has none, so `.a` doubles as the flag.
@group(0) @binding(5)
var bent_normal_texture: texture_2d<f32>;

// See config.rs#DeviceTier
const DEVICE_TIER_GLES = 0u;
const DEVICE_TIER_WEBGPU = 1u;
