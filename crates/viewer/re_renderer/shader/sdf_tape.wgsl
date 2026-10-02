// FORK DIVERGENCE: direct SDF display pass for akatela SPEC-109.
//
// Rasterize the SDF's finite proxy box, reconstruct a world-space camera ray
// per covered fragment, march the Fidget tape (group 1 bindings 1 and 2) with
// a bounded linear search plus bisection, and write the projected hit depth.
// No Lipschitz assumption: the search never advances by `|f|` (SPEC-109 R4).
// The interval pre-pass and gradient-tape normals build on this pass without
// changing its phase, depth or picking contract.

#import <./global_bindings.wgsl>
#import <./utils/camera.wgsl>
#import <./fidget_ops.wgsl>

struct SdfTapeUniformBuffer {
    /// Proxy box minimum corner, in world space, `xyz` used.
    bounds_min: vec4f,

    /// Proxy box maximum corner, in world space, `xyz` used.
    bounds_max: vec4f,

    /// Placeholder flat colour until Slice D reuses the mesh matcap/AO path.
    color: vec4f,

    /// `(object_lo, object_hi, instance_lo, instance_hi)` for the picking layer.
    picking_layer_id: vec4u,

    /// Variable indices of the `x`, `y`, `z` inputs; `0xFFFFFFFF` when absent.
    axes: vec4u,

    /// `(search_steps, bisection_steps, 0, 0)`.
    counts: vec4u,
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

/// Evaluate the root tape at a world-space point. Negative is inside.
fn field(position: vec3f) -> f32 {
    let result = run_tape(
        0u,
        array<Value, 3>(build_imm(position.x), build_imm(position.y), build_imm(position.z)),
        config.axes.xyz,
    );
    return result.value.v;
}

/// Normal from central differences of the root tape. Replaced by fidget's
/// gradient tape before the pass leaves Slice A (SPEC-109 R6).
fn field_normal(position: vec3f, distance_to_camera: f32) -> vec3f {
    // Scale the offset with the distance so the difference is well conditioned
    // close to the surface and does not underflow far away.
    let h = max(1e-4, 1e-3 * distance_to_camera);
    let dx = field(position + vec3f(h, 0.0, 0.0)) - field(position - vec3f(h, 0.0, 0.0));
    let dy = field(position + vec3f(0.0, h, 0.0)) - field(position - vec3f(0.0, h, 0.0));
    let dz = field(position + vec3f(0.0, 0.0, h)) - field(position - vec3f(0.0, 0.0, h));
    return normalize(vec3f(dx, dy, dz));
}

/// Ray/box intersection: `(near, far)` along the ray, or a negative range on a
/// miss. Rays are clipped to the validated proxy so the field is never
/// evaluated outside it (SPEC-109 R3/R13).
fn ray_box(ray: Ray) -> vec2f {
    let inverse = 1.0 / ray.direction;
    let t0 = (config.bounds_min.xyz - ray.origin) * inverse;
    let t1 = (config.bounds_max.xyz - ray.origin) * inverse;
    let t_min = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
    let t_max = min(min(max(t0.x, t1.x), max(t0.y, t1.y)), max(t0.z, t1.z));
    return vec2f(t_min, t_max);
}

/// Distance along the ray to the first field root, negative on a miss.
fn march(ray: Ray) -> f32 {
    let interval = ray_box(ray);
    let t_near = max(interval.x, 0.0);
    if interval.y <= t_near {
        return -1.0;
    }

    let steps = max(config.counts.x, 1u);
    let bisections = config.counts.y;
    let step = (interval.y - t_near) / f32(steps);
    var previous_t = t_near;
    if field(ray.origin + ray.direction * t_near) <= 0.0 {
        return t_near;
    }

    for (var i = 1u; i <= steps; i = i + 1u) {
        let t = t_near + f32(i) * step;
        if field(ray.origin + ray.direction * t) <= 0.0 {
            var lower = previous_t;
            var upper = t;
            for (var j = 0u; j < bisections; j = j + 1u) {
                let middle = 0.5 * (lower + upper);
                if field(ray.origin + ray.direction * middle) <= 0.0 {
                    upper = middle;
                } else {
                    lower = middle;
                }
            }
            return 0.5 * (lower + upper);
        }
        previous_t = t;
    }
    return -1.0;
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
    let distance = march(ray);
    if distance < 0.0 {
        discard;
    }

    let hit = ray.origin + ray.direction * distance;
    let clip = frame.projection_from_world * vec4f(hit, 1.0);
    let normal = field_normal(hit, distance);

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
    let distance = march(ray);
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
