// Shared body of the instanced mesh shader.
//
// This file is not a standalone shader module: it expects the importing file
// to declare `selected_ids` (group 2, binding 0) and `fn is_selected(u32) ->
// bool` on top. Two variants exist:
// - `instanced_mesh.wgsl`: storage-buffer selection (full WebGPU tier).
// - `instanced_mesh_limited.wgsl`: fixed-size uniform-buffer selection for the
//   `Limited` device tier (WebGL2 has no storage buffers).

#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./mesh_vertex.wgsl>
#import <./utils/srgb.wgsl>

@group(1) @binding(0)
var albedo_texture: texture_2d<f32>;

// Keep in sync with `gpu_data::TextureFormat` in mesh.rs
const FORMAT_RGBA: u32 = 0;
const FORMAT_GRAYSCALE: u32 = 1;

// Keep in sync with `gpu_data::MaterialUniformBuffer` in mesh.rs.
//
// Both flags are `wgpu_buffer_types::U32RowPadded` on the Rust side: a `u32`
// followed by three padding words, so each occupies a full 16-byte row. They
// are declared `vec4u` here to match, and read through `.x`.
//
// Declaring them as bare `u32` put `use_matcap` at byte 20 while Rust wrote it
// at byte 32 -- byte 20 being `texture_format`'s first padding word, which is
// always zero. The shader therefore saw `use_matcap == 0` for every mesh and
// took the textured path unconditionally, which is what broke face matcap
// shading. `texture_format` happens to sit at byte 16 under both layouts, so
// half the struct kept working and hid the other half.
struct MaterialUniformBuffer {
    albedo_factor: vec4f,
    /// `.x` is the `TextureFormat` discriminant; `.yzw` is padding.
    texture_format: vec4u,
    /// `.x` is 1 to light with the matcap texture, 0 for standard shading;
    /// `.yzw` is padding.
    use_matcap: vec4u,
};

@group(1) @binding(1)
var<uniform> material: MaterialUniformBuffer;

// Wireframe-mode face-selection cue. `.x` is the cue alpha: when > 0, the
// shaded pass draws ONLY selected faces, at this alpha, with everything else
// fully transparent -- giving selected faces a shading cue even when the
// opaque shaded pass is off. 0 (the default) leaves the shaded pass unchanged.
@group(2) @binding(1)
var<uniform> selection_cue: vec4f;

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
    selection_tint: vec3f,
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
    out.selection_tint = in_instance.selection_tint;

    return out;
}

// Matcap albedo, used when `material.use_matcap.x != 0`. The bound texture is a
// matcap, sampled by the view-space normal rather than by the mesh's texture
// coordinates, so a mesh needs no UVs at all on this path. Returns linear
// unmultiplied rgb in `.rgb` and separate alpha in `.a`.
fn shade_matcap(normal_world_space: vec3f, additive_tint_rgba: vec4f) -> vec4f {
    // Fallback to +Z if a normal is missing.
    let has_normal = any(normal_world_space != vec3f(0.0, 0.0, 0.0));
    let normal_world = normalize(select(
        vec3f(0.0, 0.0, 1.0),
        normal_world_space,
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
    matcap_color += additive_tint_rgba.rgb;
    matcap_color *= additive_tint_rgba.a;
    matcap_color *= material.albedo_factor.a;

    let alpha = matcap_sample.a * material.albedo_factor.a * additive_tint_rgba.a;

    return vec4f(matcap_color, alpha);
}

// Textured albedo, used when `material.use_matcap.x == 0`. The bound texture is a
// base-color map sampled at the interpolated corner UV, lit by a fixed two-light
// diffuse rig so that surface form still reads. This restores the shading path
// that the matcap work replaced, rather than inventing a second one. Returns
// linear unmultiplied rgb in `.rgb` and separate alpha in `.a`.
fn shade_textured(texcoord: vec2f, vertex_color: vec3f, normal_world_space: vec3f, additive_tint_rgba: vec4f) -> vec4f {
    let sample = textureSample(albedo_texture, trilinear_sampler_repeat, texcoord);
    var texture_color: vec3f;
    switch material.texture_format.x {
        case FORMAT_RGBA: { texture_color = linear_from_srgb(sample.rgb); }
        case FORMAT_GRAYSCALE: { texture_color = linear_from_srgb(sample.rrr); }
        default: { texture_color = vec3f(0.0); }
    }

    // Texture alpha is deliberately ignored: the CPU side flags a mesh as
    // transparent from `albedo_factor.a` alone, so honouring texture alpha here
    // would surprise-enable transparency on a mesh nothing sorted.
    var albedo = vec4f(texture_color * vertex_color, 1.0) * material.albedo_factor;

    // The additive tint is linear space with unmultiplied/separate (!!) alpha.
    albedo += vec4f(additive_tint_rgba.rgb, 0.0);
    albedo *= additive_tint_rgba.a;

    // Two lights, so that every side of the mesh picks up some shading. A mesh
    // without normals stays unshaded rather than going black.
    var shading = 1.0;
    if any(normal_world_space != vec3f(0.0, 0.0, 0.0)) {
        let normal = normalize(normal_world_space);
        shading = 0.2;
        shading += 1.0 * clamp(dot(normalize(vec3f(1.0, 2.0, 3.0)), normal), 0.0, 1.0);
        shading += 0.5 * clamp(dot(normalize(vec3f(-1.0, -3.0, -5.0)), normal), 0.0, 1.0);
        shading = clamp(shading, 0.0, 1.0);
    }

    return vec4f(albedo.rgb * shading, albedo.a);
}

@fragment
fn fs_main_shaded(in: VertexOut) -> @location(0) vec4f {
    // Matcap is the default and stays the untextured path; `use_matcap == 0`
    // opts into sampling the albedo texture at the interpolated corner UV.
    var shaded: vec4f;
    if material.use_matcap.x != 0u {
        shaded = shade_matcap(in.normal_world_space, in.additive_tint_rgba);
    } else {
        shaded = shade_textured(in.texcoord, in.color, in.normal_world_space, in.additive_tint_rgba);
    }

    var shaded_color = shaded.rgb;

    // Selection tint: blend towards geometry type color.
    if in.element_id != 0u && is_selected(in.element_id) {
        shaded_color = mix(shaded_color, in.selection_tint, 0.4);
    }

    // Hover tint: stronger blend towards geometry type color.
    if in.hover_element_id != 0u && in.element_id == in.hover_element_id {
        shaded_color = mix(shaded_color, in.selection_tint * 1.3, 0.5);
    }

    var alpha = shaded.a;

    // Wireframe-mode selection cue: when active, this draw shows only selected
    // faces (at cue alpha); every other fragment is fully transparent.
    if selection_cue.x > 0.0 {
        let selected = in.element_id != 0u && is_selected(in.element_id);
        alpha = select(0.0, selection_cue.x, selected);
    }

    return vec4f(shaded_color, alpha);
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
    // Per-vertex element_id (face mode). Carry the instance's picking layer
    // `instance` half (z/w) through so the element id stays unique across
    // separate mesh instances -- without it, face id N collides between every
    // object. `object.x` is unchanged (= element_id), so single-object picking
    // is byte-identical.
    if in.element_id != 0u {
        return vec4u(in.element_id, 0u, in.picking_layer_id.z, in.picking_layer_id.w);
    }
    discard;
    // Unreachable after `discard`, but WGSL's browser validator requires
    // a return on every path.
    return vec4u(0u, 0u, 0u, 0u);
}

@fragment
fn fs_main_outline_mask(in: VertexOut) -> @location(0) vec2u {
    return in.outline_mask_ids;
}
