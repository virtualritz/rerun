// FORK DIVERGENCE: akatela SPEC-109 composite pass.
//
// Draw Fidget's GPU-resident voxel result into the live scene. Fidget
// evaluates the field and writes one `GeometryPixel { normal, depth }` per
// pixel (see `fidget-wgpu`); this pass reads that buffer, reconstructs each
// hit's world position from its voxel depth, and writes scene colour,
// projected reverse-Z depth and picking ids in the same phases the mesh
// passes use. The buffer is read on the GPU: no CPU readback, no per-frame
// texture upload.
//
// `depth` is Fidget's voxel depth -- 0 means empty, larger means nearer to the
// camera -- NOT reverse-Z. It is converted through `world_from_voxel` and the
// frame projection here.
//
// The proxy box in the SDF's local space is rasterized so the fragment runs
// only where the field can be, and a hit is rejected when it falls outside
// the validated bounds.

#import <./global_bindings.wgsl>

struct SdfVoxelUniformBuffer {
    /// Proxy box minimum corner, in the SDF's local space; `xyz` used.
    bounds_min: vec4f,

    /// Proxy box maximum corner, in the SDF's local space; `xyz` used.
    bounds_max: vec4f,

    /// Flat base colour until the mesh material path lands (SPEC-109 T014).
    color: vec4f,

    /// `(object_lo, object_hi, instance_lo, instance_hi)` for picking.
    picking_layer_id: vec4u,

    /// Fidget render size in pixels: `(width, height, 0, 0)`.
    ///
    /// The fragment maps its framebuffer position into this grid, so one
    /// evaluation serves targets of any resolution -- e.g. the viewport at
    /// full size and the downscaled picking target in the same frame.
    size: vec4u,

    /// Maps Fidget voxel coordinates `(px, py, depth, 1)` to world space.
    world_from_voxel: mat4x4<f32>,

    /// Placement of the SDF's local space in world space.
    world_from_local: mat4x4<f32>,

    /// Inverse of `world_from_local`. The field gradient is a covector, so the
    /// world normal uses its transpose.
    local_from_world: mat4x4<f32>,
}

@group(1) @binding(0)
var<uniform> config: SdfVoxelUniformBuffer;

struct GeometryPixel {
    normal: vec3f,
    depth: u32,
}

/// Fidget's `GeometryPixel` output, one entry per pixel, row-major.
@group(1) @binding(1)
var<storage, read> geometry: array<GeometryPixel>;

// Corner index bits: bit 0 = x, bit 1 = y, bit 2 = z.
fn corner(index: u32) -> vec3f {
    return vec3f(
        select(config.bounds_min.x, config.bounds_max.x, (index & 1u) != 0u),
        select(config.bounds_min.y, config.bounds_max.y, (index & 2u) != 0u),
        select(config.bounds_min.z, config.bounds_max.z, (index & 4u) != 0u),
    );
}

// Six quads, each as four corner indices.
const FACE_CORNERS: array<array<u32, 4>, 6> = array<array<u32, 4>, 6>(
    array<u32, 4>(0u, 1u, 3u, 2u), // z = min
    array<u32, 4>(4u, 5u, 7u, 6u), // z = max
    array<u32, 4>(0u, 1u, 5u, 4u), // y = min
    array<u32, 4>(2u, 3u, 7u, 6u), // y = max
    array<u32, 4>(0u, 2u, 6u, 4u), // x = min
    array<u32, 4>(1u, 3u, 7u, 5u), // x = max
);

// Two triangles per quad: 0-1-2, 0-2-3.
const QUAD_TRIANGLE: array<u32, 6> = array<u32, 6>(
    0u, 1u, 2u, 0u, 2u, 3u,
);

struct VertexOutput {
    @builtin(position)
    position: vec4f,
}

@vertex
fn main_vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let face = vertex_index / 6u;
    let triangle = (vertex_index % 6u) / 3u;
    let corner_in_quad = QUAD_TRIANGLE[triangle * 3u + vertex_index % 3u];
    let local_position = corner(FACE_CORNERS[face][corner_in_quad]);
    let world_position =
        (config.world_from_local * vec4f(local_position, 1.0)).xyz;

    var out: VertexOutput;
    out.position = frame.projection_from_world * vec4f(world_position, 1.0);
    return out;
}

struct Hit {
    /// Hit position in world space.
    world_position: vec3f,

    /// Outward surface normal in world space.
    world_normal: vec3f,

    /// False when the pixel has no hit, or the hit is outside the proxy.
    found: bool,
}

/// Reconstruct the Fidget hit at a framebuffer pixel.
///
/// A hit outside the validated proxy bounds is treated as a miss rather than
/// silently clipped: the SDF is expected to be positive outside its bounds, so
/// reaching here means the bounds were wrong and the caller should have taken
/// the mesh fallback.
fn surface_hit(framebuffer_position: vec4f) -> Hit {
    // Fidget's grid is not necessarily the target's grid: the picking pass
    // renders at a reduced resolution. Map into the Fidget grid so both
    // passes read the same evaluation.
    let buffer_position =
        vec2u(framebuffer_position.xy
            * vec2f(config.size.xy) / frame.framebuffer_resolution);
    let px = buffer_position.x;
    let py = buffer_position.y;
    if (px >= config.size.x || py >= config.size.y) {
        return Hit(vec3f(0.0), vec3f(0.0, 0.0, 1.0), false);
    }

    let pixel = geometry[px + py * config.size.x];
    if (pixel.depth == 0u) {
        return Hit(vec3f(0.0), vec3f(0.0, 0.0, 1.0), false);
    }

    let voxel = vec4f(f32(px), f32(py), f32(pixel.depth), 1.0);
    let world = config.world_from_voxel * voxel;
    let world_position = world.xyz / world.w;
    let local_position =
        (config.local_from_world * vec4f(world_position, 1.0)).xyz;
    if (any(local_position < config.bounds_min.xyz)
        || any(local_position > config.bounds_max.xyz)) {
        return Hit(vec3f(0.0), vec3f(0.0, 0.0, 1.0), false);
    }

    // Fidget's gradient is in the SDF's model space. The render transform maps
    // model space to the voxel reconstruction space, and a gradient is a
    // covector, so `transpose(local_from_world)` carries it across.
    let world_normal =
        normalize((vec4f(pixel.normal, 0.0) * config.local_from_world).xyz);
    return Hit(world_position, world_normal, true);
}

struct ShadedFragment {
    @location(0)
    color: vec4f,

    @builtin(frag_depth)
    depth: f32,
}

@fragment
fn fs_main_shaded(@builtin(position) position: vec4f) -> ShadedFragment {
    let hit = surface_hit(position);
    if (!hit.found) {
        discard;
    }

    let clip = frame.projection_from_world * vec4f(hit.world_position, 1.0);

    var out: ShadedFragment;
    // Placeholder shading: the normal tint makes warped depth and transforms
    // visible until SPEC-109 T014 reuses the mesh matcap/AO path.
    out.color = config.color * vec4f(hit.world_normal * 0.5 + 0.5, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return out;
}

struct PickingFragment {
    @location(0)
    picking_layer_id: vec4u,

    @builtin(frag_depth)
    depth: f32,
}

@fragment
fn fs_main_picking(@builtin(position) position: vec4f) -> PickingFragment {
    let hit = surface_hit(position);
    if (!hit.found) {
        discard;
    }

    let clip = frame.projection_from_world * vec4f(hit.world_position, 1.0);

    var out: PickingFragment;
    out.picking_layer_id = config.picking_layer_id;
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return out;
}
