// FORK DIVERGENCE: direct SDF display pass for akatela SPEC-109.
//
// This is Slice A of the spec: an analytic sphere drawn by rasterizing its
// finite proxy box, reconstructing a world-space ray per covered fragment and
// writing the projected hit depth. The box proxy and the sphere are the test
// fixture the depth/picking contract is pinned against; the Fidget tape
// interpreter and its interval pre-pass arrive on top of this pass without
// changing its phase, depth or picking contract.
//
// The fragment only runs where the proxy box covers a pixel, so a ray outside
// the proxy never evaluates the field.

#import <./global_bindings.wgsl>
#import <./utils/camera.wgsl>

struct SdfTapeUniformBuffer {
    /// Proxy box minimum corner, in world space, `xyz` used.
    bounds_min: vec4f,

    /// Proxy box maximum corner, in world space, `xyz` used.
    bounds_max: vec4f,

    /// Sphere centre in `xyz` and radius in `w`.
    center_radius: vec4f,

    /// Placeholder flat colour until Slice D reuses the mesh matcap/AO path.
    color: vec4f,

    /// `(object_lo, object_hi, instance_lo, instance_hi)` for the picking layer.
    picking_layer_id: vec4u,
}

@group(1) @binding(0)
var<uniform> config: SdfTapeUniformBuffer;

struct VertexOutput {
    @builtin(position)
    position: vec4f,

    @location(0)
    world_position: vec3f,
}

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

@vertex
fn main_vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let face = vertex_index / 6u;
    let triangle = (vertex_index % 6u) / 3u;
    let corner_in_quad = QUAD_TRIANGLE[triangle * 3u + vertex_index % 3u];
    let position = corner(FACE_CORNERS[face][corner_in_quad]);

    var out: VertexOutput;
    out.position = frame.projection_from_world * vec4f(position, 1.0);
    out.world_position = position;
    return out;
}

struct ShadedFragment {
    @location(0)
    color: vec4f,

    @builtin(frag_depth)
    depth: f32,
}

@fragment
fn fs_main_shaded(in: VertexOutput) -> ShadedFragment {
    let ray = camera_ray_to_world_pos(in.world_position);
    let distance = ray_sphere_distance(ray, config.center_radius.xyz, config.center_radius.w).y;
    if distance < 0.0 {
        discard;
    }

    let hit = ray.origin + ray.direction * distance;
    let clip = frame.projection_from_world * vec4f(hit, 1.0);
    let normal = normalize(hit - config.center_radius.xyz);

    var out: ShadedFragment;
    out.color = config.color * vec4f(normal * 0.5 + 0.5, 1.0);
    out.depth = clip.z / clip.w;
    return out;
}

struct PickingFragment {
    @location(0)
    picking_layer_id: vec4u,

    @builtin(frag_depth)
    depth: f32,
}

@fragment
fn fs_main_picking(in: VertexOutput) -> PickingFragment {
    let ray = camera_ray_to_world_pos(in.world_position);
    let distance = ray_sphere_distance(ray, config.center_radius.xyz, config.center_radius.w).y;
    if distance < 0.0 {
        discard;
    }

    let hit = ray.origin + ray.direction * distance;
    let clip = frame.projection_from_world * vec4f(hit, 1.0);

    var out: PickingFragment;
    out.picking_layer_id = config.picking_layer_id;
    out.depth = clip.z / clip.w;
    return out;
}
