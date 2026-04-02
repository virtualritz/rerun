#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./mesh_vertex.wgsl>
#import <./utils/srgb.wgsl>

@group(1) @binding(0)
var albedo_texture: texture_2d<f32>;

// Keep in sync with `gpu_data::TextureFormat` in mesh.rs
const FORMAT_RGBA: u32 = 0;
const FORMAT_GRAYSCALE: u32 = 1;

// Keep in sync with `gpu_data::MaterialUniformBuffer` in mesh.rs
struct MaterialUniformBuffer {
    albedo_factor: vec4f,
    texture_format: u32,
    use_matcap: u32,
};

@group(1) @binding(1)
var<uniform> material: MaterialUniformBuffer;

struct VertexOut {
    @builtin(position)
    position: vec4f,

    @location(0)
    color: vec3f, // 0-1 linear space with unmultiplied/separate alpha

    @location(1)
    texcoord: vec2f,

    @location(2)
    normal_world_space: vec3f,

    @location(3) @interpolate(flat)
    additive_tint_rgba: vec4f, // 0-1 linear space with unmultiplied/separate alpha

    @location(4) @interpolate(flat)
    outline_mask_ids: vec2u,

    @location(5) @interpolate(flat)
    picking_layer_id: vec4u,

    @location(6) @interpolate(flat)
    element_id: u32,

    @location(7) @interpolate(flat)
    hover_element_id: u32,

    @location(8) @interpolate(flat)
    selected_element_count: u32,
    @location(9) @interpolate(flat)
    selected_ids_0: vec4u,
    @location(10) @interpolate(flat)
    selected_ids_1: vec4u,
    @location(11) @interpolate(flat)
    selected_ids_2: vec4u,
    @location(12) @interpolate(flat)
    selected_ids_3: vec4u,
};

@vertex
fn vs_main(in_vertex: VertexIn, in_instance: InstanceIn) -> VertexOut {
    let world_position = vec3f(
        dot(in_instance.world_from_mesh_row_0.xyz, in_vertex.position) + in_instance.world_from_mesh_row_0.w,
        dot(in_instance.world_from_mesh_row_1.xyz, in_vertex.position) + in_instance.world_from_mesh_row_1.w,
        dot(in_instance.world_from_mesh_row_2.xyz, in_vertex.position) + in_instance.world_from_mesh_row_2.w,
    );
    let world_normal = vec3f(
        dot(in_instance.world_from_mesh_normal_row_0.xyz, in_vertex.normal),
        dot(in_instance.world_from_mesh_normal_row_1.xyz, in_vertex.normal),
        dot(in_instance.world_from_mesh_normal_row_2.xyz, in_vertex.normal),
    );

    var out: VertexOut;
    out.position = frame.projection_from_world * vec4f(world_position, 1.0);
    out.color = linear_from_srgb(in_vertex.color.rgb);
    out.texcoord = in_vertex.texcoord;
    out.normal_world_space = world_normal;
    // Instance encoded is with pre-multiplied alpha in sRGB.
    out.additive_tint_rgba = vec4f(linear_from_srgb(in_instance.additive_tint_srgba.rgb / in_instance.additive_tint_srgba.a),
                                    in_instance.additive_tint_srgba.a);
    out.outline_mask_ids = in_instance.outline_mask_ids;
    out.picking_layer_id = in_instance.picking_layer_id;
    out.element_id = in_vertex.element_id;
    out.hover_element_id = in_instance.hover_element_id;
    out.selected_element_count = in_instance.selected_element_count;
    out.selected_ids_0 = in_instance.selected_element_ids_0;
    out.selected_ids_1 = in_instance.selected_element_ids_1;
    out.selected_ids_2 = in_instance.selected_element_ids_2;
    out.selected_ids_3 = in_instance.selected_element_ids_3;

    return out;
}

/// Linear search for `element_id` in up to 16 packed selection IDs.
fn is_selected(element_id: u32, count: u32, ids0: vec4u, ids1: vec4u, ids2: vec4u, ids3: vec4u) -> bool {
    for (var i = 0u; i < min(count, 4u); i++) { if ids0[i] == element_id { return true; } }
    for (var i = 0u; i < min(max(count, 4u) - 4u, 4u); i++) { if ids1[i] == element_id { return true; } }
    for (var i = 0u; i < min(max(count, 8u) - 8u, 4u); i++) { if ids2[i] == element_id { return true; } }
    for (var i = 0u; i < min(max(count, 12u) - 12u, 4u); i++) { if ids3[i] == element_id { return true; } }
    return false;
}

@fragment
fn fs_main_shaded(in: VertexOut) -> @location(0) vec4f {
    // Always use matcap shading. Fallback to +Z if a normal is missing.
    let has_normal = any(in.normal_world_space != vec3f(0.0, 0.0, 0.0));
    let normal_world = normalize(select(
        vec3f(0.0, 0.0, 1.0),
        in.normal_world_space,
        vec3<bool>(has_normal, has_normal, has_normal),
    ));

    // view_from_world is a mat4x3f, so extract the 3x3 rotation part.
    let view_normal = normalize(vec3f(
        dot(vec3f(frame.view_from_world[0].x, frame.view_from_world[1].x, frame.view_from_world[2].x), normal_world),
        dot(vec3f(frame.view_from_world[0].y, frame.view_from_world[1].y, frame.view_from_world[2].y), normal_world),
        dot(vec3f(frame.view_from_world[0].z, frame.view_from_world[1].z, frame.view_from_world[2].z), normal_world)
    ));

    // Map view-space normal XY from [-1,1] to [0,1] for texture lookup.
    let matcap_uv = view_normal.xy * 0.5 + 0.5;

    // Sample matcap texture (passed as albedo_texture).
    let matcap_sample = textureSample(albedo_texture, trilinear_sampler_repeat, matcap_uv);
    var matcap_color = linear_from_srgb(matcap_sample.rgb);

    // Apply albedo factor for tinting.
    matcap_color *= material.albedo_factor.rgb;

    // Apply additive tint.
    matcap_color += in.additive_tint_rgba.rgb;
    matcap_color *= in.additive_tint_rgba.a;

    // Selection tint: blue-ish tint for selected faces.
    if in.element_id != 0u && in.selected_element_count > 0u && is_selected(in.element_id, in.selected_element_count, in.selected_ids_0, in.selected_ids_1, in.selected_ids_2, in.selected_ids_3) {
        matcap_color = matcap_color * vec3f(0.6, 0.85, 1.3);
    }

    // Hover tint: brighten the face under the cursor.
    if in.hover_element_id != 0u && in.element_id == in.hover_element_id {
        matcap_color = matcap_color * 1.35 + vec3f(0.08, 0.08, 0.12);
    }

    return vec4f(matcap_color, matcap_sample.a);
}

@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
    // Sentinel 0xFFFFFFFF = discard (suppress face IDs for edge/vertex modes).
    if in.picking_layer_id.x == 0xFFFFFFFFu {
        discard;
    }
    // Non-zero picking_layer_id overrides element_id (used for body mode).
    if in.picking_layer_id.x != 0u {
        return vec4u(in.picking_layer_id.x, 0u, 0u, 0u);
    }
    // Per-vertex element_id (face mode).
    if in.element_id != 0u {
        return vec4u(in.element_id, 0u, 0u, 0u);
    }
    discard;
}

@fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
    return in.outline_mask_ids;
}
