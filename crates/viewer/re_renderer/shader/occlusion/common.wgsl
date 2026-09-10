// Shared by the occlusion passes (akatela SPEC-123).
//
// Keep `OcclusionUniformBuffer` in sync with `draw_phases/occlusion.rs`.

struct OcclusionUniformBuffer {
    view_from_projection: mat4x4f,
    framebuffer_resolution: vec2f,
    // Pixels per world unit: at unit view depth for a perspective projection,
    // outright for an orthographic one.
    pixels_per_world_unit: f32,
    // 1 for a perspective projection, 0 for an orthographic one.
    perspective: u32,
    world_radius: f32,
    pixel_radius_min: f32,
    pixel_radius_max: f32,
    strength: f32,
    sample_count: u32,
    // There is more padding in the buffer; the shader needs none of it.
};

// The view-space position of a pixel, from its depth.
//
// Never call this with depth 0: reverse-Z puts the far plane there, and a
// perspective projection sends it to infinity.
fn view_position(params: OcclusionUniformBuffer, pixel: vec2i, depth: f32) -> vec3f {
    let uv = (vec2f(pixel) + 0.5) / params.framebuffer_resolution;
    let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let position = params.view_from_projection * vec4f(ndc, depth, 1.0);
    return position.xyz / position.w;
}
