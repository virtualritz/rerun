// Shared matcap shading, used by the instanced mesh shader and the SDF
// composite (akatela SPEC-109 R6/R14).
//
// The functions here are binding-agnostic: `shade_matcap_lobes` takes the two
// matcap textures, the sampler, and the material scalars as parameters, so a
// caller can declare its own bind group. `occlusion_at` and `bent_normal_at`
// read the view's occlusion outputs from the group-0 global bindings, which
// every view shares.
//
// The WGSL `MaterialUniformBuffer` lives here so the mesh shader and the SDF
// shader cannot drift apart; the Rust twin is
// `gpu_data::MaterialUniformBuffer` in `mesh.rs`.

#import <../global_bindings.wgsl>
#import <../types.wgsl>

// Keep in sync with `gpu_data::MaterialUniformBuffer` in mesh.rs.
//
// Both scalars are a `U32RowPadded` on the Rust side -- a u32 followed by
// three words of padding -- so each occupies a full 16-byte row and the next
// field starts 16 bytes later, not 4. WGSL gives a bare `u32` an alignment of
// 4, so writing them back to back put `use_matcap` at offset 20 while Rust
// wrote it at 32; the shader read `texture_format`'s first padding word, which
// is always zero, and every mesh took the `use_matcap == 0` branch. Matcap
// shading could never appear. `texture_format` was unaffected only because
// offset 16 happens to be correct for both.
//
// `tests/shader_validation.rs` pins these offsets against the Rust struct.
struct MaterialUniformBuffer {
    albedo_factor: vec4f,
    texture_format: u32,
    // Padding out `texture_format`'s row. Three separate scalars, not a
    // `vec3u` (alignment 16 -- it would jump to offset 32 itself and push
    // `use_matcap` to 44) and not an `array<u32, 3>` (in the UNIFORM address
    // space WGSL rounds array stride up to 16, making it 48 bytes wide).
    _texture_format_padding_0: u32,
    _texture_format_padding_1: u32,
    _texture_format_padding_2: u32,
    use_matcap: u32,
    // `use_matcap` is a `U32RowPadded` too, so its row must be filled before
    // the next field, for the same reason as above.
    _use_matcap_padding_0: u32,
    _use_matcap_padding_1: u32,
    _use_matcap_padding_2: u32,
    specular_roughness: f32,
};

// This pixel's ambient occlusion (akatela SPEC-123). When the view has none
// the texture is 1x1 white, so the coordinate is clamped: reading outside a
// texture returns an indeterminate value.
fn occlusion_at(frag_position: vec4f) -> f32 {
    let size = textureDimensions(occlusion_texture);
    let pixel = min(vec2u(frag_position.xy), size - vec2u(1u));
    return textureLoad(occlusion_texture, pixel, 0).r;
}

// This pixel's bent normal and its presence flag (SPEC-123 D3a). See
// `bent_normal_texture` in `global_bindings.wgsl`.
fn bent_normal_at(frag_position: vec4f) -> vec4f {
    let size = textureDimensions(bent_normal_texture);
    let pixel = min(vec2u(frag_position.xy), size - vec2u(1u));
    return textureLoad(bent_normal_texture, pixel, 0);
}

// The solid angle where two spherical caps overlap, from the cosines of their
// half-angles and of the angle between their axes. After Oat and Sander
// (2007), as Unity's SRP core implements it. Keep in sync with the Rust
// reference in `draw_phases/occlusion.rs`.
fn spherical_cap_intersection(cos_c1: f32, cos_c2: f32, cos_b: f32) -> f32 {
    let r1 = acos(clamp(cos_c1, -1.0, 1.0));
    let r2 = acos(clamp(cos_c2, -1.0, 1.0));
    let rd = acos(clamp(cos_b, -1.0, 1.0));
    let smaller_cap = 6.283185307 - 6.283185307 * max(cos_c1, cos_c2);
    if rd <= max(r1, r2) - min(r1, r2) {
        return smaller_cap; // one cap lies inside the other
    }
    if rd >= r1 + r2 {
        return 0.0; // the caps do not meet
    }
    let diff = abs(r1 - r2);
    let den = r1 + r2 - diff;
    let x = 1.0 - saturate((rd - diff) / max(den, 0.0001));
    return smoothstep(0.0, 1.0, x) * smaller_cap;
}

// `frame.occlusion_debug`, in sync with `OcclusionDebugView`.
const OCCLUSION_DEBUG_AMBIENT: u32 = 1u;
const OCCLUSION_DEBUG_SPECULAR: u32 = 2u;

// Specular occlusion from a bent cone (SPEC-123 D3a), after Jimenez et al.,
// "Practical Realtime Strategies for Accurate Indirect Occlusion" (SIGGRAPH
// 2016), slide 129: the share of the reflection cone inside the visible cone
// around the bent normal. Keep in sync with the Rust reference.
fn specular_occlusion_cone(eye: vec3f, bent_normal: vec3f, normal: vec3f, ao: f32, roughness: f32) -> f32 {
    // Occlusion is cosine weighted, so the visible cone's half-angle follows.
    let cos_visible = sqrt(max(1.0 - ao, 0.0));
    let r = max(roughness, 0.01);
    // 10^(-r^2): the reflection cone widens with roughness.
    let cos_reflection = exp2(-3.321928 * r * r);
    let cos_between = dot(bent_normal, reflect(-eye, normal));
    return saturate(spherical_cap_intersection(cos_visible, cos_reflection, cos_between)
        / (6.283185307 * (1.0 - cos_reflection)));
}

// Specular occlusion from ambient occlusion (SPEC-123 R4), after Lagarde and
// de Rousiers, "Moving Frostbite to PBR" (2014). Roughness 1 gives `ao`. A
// smooth surface occludes less facing the viewer and more at grazing angles.
fn specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    return saturate(pow(n_dot_v + ao, exp2(-16.0 * roughness - 1.0)) - 1.0 + ao);
}

fn view_normal(normal_world: vec3f) -> vec3f {
    // view_from_world is a mat4x3f, so extract the 3x3 rotation part.
    return normalize(vec3f(
        dot(vec3f(frame.view_from_world[0].x, frame.view_from_world[1].x, frame.view_from_world[2].x), normal_world),
        dot(vec3f(frame.view_from_world[0].y, frame.view_from_world[1].y, frame.view_from_world[2].y), normal_world),
        dot(vec3f(frame.view_from_world[0].z, frame.view_from_world[1].z, frame.view_from_world[2].z), normal_world)
    ));
}

// The surface normal in world space, turned toward the eye (SPEC-123 R11).
// The eye test uses the supplied normal instead of triangle winding: imported
// normals, analytic SDF/B-rep normals, and reflected instances can disagree
// with the rasterizer's front-face flag even when each normal is valid.
fn facing_world_normal(normal_world_space: vec3f, position_view: vec3f) -> vec3f {
    let has_normal = any(normal_world_space != vec3f(0.0, 0.0, 0.0));
    let normal_world = normalize(select(
        vec3f(0.0, 0.0, 1.0),
        normal_world_space,
        vec3<bool>(has_normal, has_normal, has_normal),
    ));

    let normal_view = view_normal(normal_world);
    let eye = eye_vector(position_view);
    return select(-normal_world, normal_world, dot(normal_view, eye) >= 0.0);
}

// The same eye-facing normal in view space. Falls back to world +Z when the
// mesh carries no normal.
fn facing_view_normal(normal_world_space: vec3f, position_view: vec3f) -> vec3f {
    return view_normal(facing_world_normal(normal_world_space, position_view));
}

// The unit vector from the surface toward the eye, in view space (SPEC-123
// D3b). View space here looks down -Z, so it lies in the +Z hemisphere.
// `tan_half_fov` is `f32::MAX` for an orthographic camera, where every pixel
// shares one view direction.
fn eye_vector(position_view: vec3f) -> vec3f {
    let orthographic = frame.tan_half_fov.y > 1.0e30;
    return select(normalize(-position_view), vec3f(0.0, 0.0, 1.0), orthographic);
}

// Blender's `matcap_uv_compute(I, N)` (workbench_matcap_lib.glsl), verbatim
// with its vertical `flipped` convention. It builds an orthonormal basis around the
// eye vector, so a perspective camera looks the matcap up correctly away from
// the screen centre. With I = +Z it reduces to `N.xy`. The basis is singular
// at I.z = -1, which the eye vector never reaches.
fn matcap_uv_compute(eye: vec3f, normal: vec3f) -> vec2f {
    let a = 1.0 / (1.0 + eye.z);
    let b = -eye.x * eye.y * a;
    let b1 = vec3f(1.0 - eye.x * eye.x * a, b, -eye.x);
    let b2 = vec3f(b, 1.0 - eye.y * eye.y * a, -eye.y);
    // Texture V grows downward while view-space +Y grows upward. Blender's
    // matcap lookup therefore flips the basis' second coordinate. The first
    // coordinate already agrees with the camera's screen-right axis.
    return vec2f(dot(b1, normal), -dot(b2, normal)) * 0.496 + 0.5;
}

// Cubic B-spline filtered sample, from four bilinear fetches (Sigg and
// Hadwiger, GPU Gems 2 ch. 20).
//
// A matcap is magnified: one texel covers many pixels of a large mesh.
// Bilinear filtering joins texels with straight segments, and the eye reads
// the slope change at every texel as a band even in an f16 matcap. The
// B-spline is smooth across texels and never overshoots, so a highlight does
// not ring; it softens a little, which a matcap does not mind.
//
// Keep in sync with `bspline_4tap_1d` in `mesh_renderer.rs`.
fn texture_sample_bicubic(t: texture_2d<f32>, s: sampler, uv: vec2f) -> vec4f {
    let size = vec2f(textureDimensions(t));
    let texel = uv * size - 0.5;
    let base = floor(texel);
    let f = texel - base;
    let f2 = f * f;
    let f3 = f2 * f;
    let w0 = (1.0 - 3.0 * f + 3.0 * f2 - f3) / 6.0;
    let w1 = (4.0 - 6.0 * f2 + 3.0 * f3) / 6.0;
    let w2 = (1.0 + 3.0 * f + 3.0 * f2 - 3.0 * f3) / 6.0;
    let w3 = f3 / 6.0;
    // Each pair of taps folds into one bilinear fetch between them.
    let g0 = w0 + w1;
    let g1 = w2 + w3;
    let p0 = (base - 0.5 + w1 / g0) / size;
    let p1 = (base + 1.5 + w3 / g1) / size;
    return g0.y * (g0.x * textureSample(t, s, vec2f(p0.x, p0.y)) + g1.x * textureSample(t, s, vec2f(p1.x, p0.y)))
        + g1.y * (g0.x * textureSample(t, s, vec2f(p0.x, p1.y)) + g1.x * textureSample(t, s, vec2f(p1.x, p1.y)));
}

// Matcap albedo, used when `material.use_matcap != 0`. The bound texture is a
// matcap, sampled by the view-space normal rather than by the mesh's texture
// coordinates, so a mesh needs no UVs at all on this path. Returns linear
// unmultiplied rgb in `.rgb` and separate alpha in `.a`.
fn shade_matcap_lobes(
    diffuse_texture: texture_2d<f32>,
    specular_texture: texture_2d<f32>,
    matcap_sampler: sampler,
    albedo_factor: vec4f,
    specular_roughness: f32,
    normal_world_space: vec3f,
    position_view: vec3f,
    additive_tint_rgba: vec4f,
    occlusion: f32,
    bent: vec4f,
) -> vec4f {
    let facing_normal = facing_view_normal(normal_world_space, position_view);
    let eye = eye_vector(position_view);

    // Map view-space normal XY from [-1,1] to [0,1] for texture lookup.
    let matcap_uv = matcap_uv_compute(eye, facing_normal);

    // No sRGB decode here. An `Rgba8UnormSrgb` texture is linearised by the
    // sampler, and an EXR lobe is linear already, so decoding would darken
    // both. The previous `linear_from_srgb` call was a second decode.
    let matcap_sample = texture_sample_bicubic(diffuse_texture, matcap_sampler, matcap_uv);
    let diffuse_lobe = matcap_sample.rgb;
    let specular_lobe = texture_sample_bicubic(specular_texture, matcap_sampler, matcap_uv).rgb;

    // Blender's rule: multiply the diffuse lobe, then ADD the specular lobe.
    // The base colour therefore tints the body without washing out the
    // highlights, which is what lets one matcap read as ceramic and another
    // as steel.
    //
    // Each lobe is masked on its own (SPEC-123 R2): ambient occlusion darkens
    // the body, and the specular lobe takes the specular occlusion derived
    // from it. With a bent normal from the horizon method that is the cone
    // overlap (D3a); without one, the analytic R4 formula.
    let n_dot_v = saturate(dot(facing_normal, eye));
    let cone_mask = specular_occlusion_cone(
        eye, normalize(bent.rgb * 2.0 - 1.0), facing_normal, occlusion, specular_roughness);
    let analytic_mask = specular_occlusion(n_dot_v, occlusion, specular_roughness);
    let specular_mask = select(analytic_mask, cone_mask, bent.a > 0.5);
    if frame.occlusion_debug == OCCLUSION_DEBUG_SPECULAR {
        return vec4f(vec3f(specular_mask), 1.0);
    }
    var matcap_color = diffuse_lobe * albedo_factor.rgb * occlusion
        + specular_lobe * specular_mask;

    // Apply additive tint.
    matcap_color += additive_tint_rgba.rgb;
    matcap_color *= additive_tint_rgba.a;
    matcap_color *= albedo_factor.a;

    let alpha = matcap_sample.a * albedo_factor.a * additive_tint_rgba.a;

    return vec4f(matcap_color, alpha);
}
