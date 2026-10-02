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
#import <./utils/dither.wgsl>
#import <./utils/matcap.wgsl>

@group(1) @binding(0)
var albedo_texture: texture_2d<f32>;

// The ADDED matcap lobe, sampled by the same view-space normal as the diffuse
// lobe. A 1x1 black texture when the matcap has no specular group, so the
// added term contributes nothing and the binding never varies.
@group(1) @binding(2)
var specular_matcap_texture: texture_2d<f32>;

// Keep in sync with `gpu_data::TextureFormat` in mesh.rs
const FORMAT_RGBA: u32 = 0;
const FORMAT_GRAYSCALE: u32 = 1;


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

    // For the per-pixel view vector (akatela SPEC-123 D3b).
    @location(9)
    position_view: vec3f,
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
    out.position_view = frame.view_from_world * vec4f(world_position, 1.0);

    return out;
}

// The mesh's matcap path: the shared lobes bound to this shader's material.
fn shade_matcap(normal_world_space: vec3f, position_view: vec3f, additive_tint_rgba: vec4f, occlusion: f32, bent: vec4f) -> vec4f {
    return shade_matcap_lobes(
        albedo_texture,
        specular_matcap_texture,
        trilinear_sampler_repeat,
        material.albedo_factor,
        material.specular_roughness,
        normal_world_space,
        position_view,
        additive_tint_rgba,
        occlusion,
        bent,
    );
}


// Textured albedo, used when `material.use_matcap == 0`. The bound texture is a
// base-color map sampled at the interpolated corner UV, lit by a fixed two-light
// diffuse rig so that surface form still reads. This restores the shading path
// that the matcap work replaced, rather than inventing a second one. Returns
// linear unmultiplied rgb in `.rgb` and separate alpha in `.a`.
fn shade_textured(texcoord: vec2f, vertex_color: vec3f, normal_world_space: vec3f, position_view: vec3f, additive_tint_rgba: vec4f, occlusion: f32) -> vec4f {
    let sample = textureSample(albedo_texture, trilinear_sampler_repeat, texcoord);
    var texture_color: vec3f;
    switch material.texture_format {
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
        let normal = facing_world_normal(normal_world_space, position_view);
        shading = 0.2;
        shading += 1.0 * clamp(dot(normalize(vec3f(1.0, 2.0, 3.0)), normal), 0.0, 1.0);
        shading += 0.5 * clamp(dot(normalize(vec3f(-1.0, -3.0, -5.0)), normal), 0.0, 1.0);
        shading = clamp(shading, 0.0, 1.0);
    }

    // Occlusion belongs to the scene, not the material (SPEC-123 R5).
    return vec4f(albedo.rgb * shading * occlusion, albedo.a);
}

// The shared body of the two shaded entry points.
fn shade(in: VertexOut, occlusion: f32, bent: vec4f) -> vec4f {
    // A debug view shows the occlusion term itself, so what you see is what
    // masks the lobe. The selection and hover tints are skipped below: the
    // grey IS the value.
    if frame.occlusion_debug == OCCLUSION_DEBUG_AMBIENT {
        return vec4f(vec3f(occlusion), 1.0);
    }

    // Matcap is the default and stays the untextured path; `use_matcap == 0`
    // opts into sampling the albedo texture at the interpolated corner UV.
    var shaded: vec4f;
    if material.use_matcap != 0u {
        shaded = shade_matcap(in.normal_world_space, in.position_view, in.additive_tint_rgba, occlusion, bent);
    } else {
        // A textured surface has no specular lobe, so nothing occludes it.
        if frame.occlusion_debug == OCCLUSION_DEBUG_SPECULAR {
            return vec4f(1.0);
        }
        shaded = shade_textured(in.texcoord, in.color, in.normal_world_space, in.position_view, in.additive_tint_rgba, occlusion);
    }

    if frame.occlusion_debug != 0u {
        return shaded;
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

// Dithers the shaded color before the 8-bit main target rounds it.
fn dithered(shaded: vec4f, frag_position: vec4f) -> vec4f {
    return vec4f(dither_linear_for_srgb8(shaded.rgb, frag_position.xy), shaded.a);
}

@fragment
fn fs_main_shaded(in: VertexOut) -> @location(0) vec4f {
    return dithered(shade(in, occlusion_at(in.position), bent_normal_at(in.position)), in.position);
}

// Transparent geometry neither writes nor receives occlusion (SPEC-123 R9).
// The occlusion behind a transparent surface belongs to what is behind it.
@fragment
fn fs_main_shaded_unoccluded(in: VertexOut) -> @location(0) vec4f {
    return dithered(shade(in, 1.0, vec4f(0.0)), in.position);
}

// The occlusion prepass (SPEC-123): the view-space normal, mapped to [0, 1].
@fragment
fn fs_main_occlusion_prepass(in: VertexOut) -> @location(0) vec4f {
    return vec4f(facing_view_normal(in.normal_world_space, in.position_view) * 0.5 + 0.5, 1.0);
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
