// Horizon-based ambient occlusion with a bent normal (akatela SPEC-123 D3a).
//
// A port of Intel's XeGTAO (MIT licence), after Jimenez et al., "Practical
// Realtime Strategies for Accurate Indirect Occlusion" (SIGGRAPH 2016). The
// arithmetic is XeGTAO's, line for line, with its default settings. XeGTAO
// works in a left-handed view space (+Z forward) and ours is right-handed
// (-Z forward), so positions and normals flip Z on the way in and the bent
// normal flips back on the way out.
//
// Two departures, both because this path has no history buffer. The slice and
// step noise is a 4x4 Bayer pattern the blur cancels, not XeGTAO's
// spatio-temporal noise. And there is no depth MIP chain, so every tap reads
// full-resolution depth.

#import <../types.wgsl>
#import <../screen_triangle_vertex.wgsl>
#import <./common.wgsl>

@group(0) @binding(0)
var depth_texture: texture_depth_2d;
@group(0) @binding(1)
var normal_texture: texture_2d<f32>;
@group(0) @binding(2)
var<uniform> params: OcclusionUniformBuffer;

const GTAO_PI: f32 = 3.1415926535897932;
const GTAO_PI_HALF: f32 = 1.5707963267948966;

// XeGTAO's defaults (XeGTAO.h, XE_GTAO_DEFAULT_*).
const RADIUS_MULTIPLIER: f32 = 1.457;
const FALLOFF_RANGE: f32 = 0.615;
const SAMPLE_DISTRIBUTION_POWER: f32 = 2.0;
const THIN_OCCLUDER_COMPENSATION: f32 = 0.0;
const FINAL_VALUE_POWER: f32 = 2.2;

// XeGTAO's `pixelTooCloseThreshold`: taps closer than this, in pixels, are
// skipped.
const PIXEL_TOO_CLOSE_THRESHOLD: f32 = 1.3;

// Our right-handed view space to XeGTAO's left-handed one, and back.
fn flip_z(v: vec3f) -> vec3f {
    return vec3f(v.x, v.y, -v.z);
}

// A 4x4 ordered pattern in [0, 1). The blur's 4x4 window sees each value once,
// so the pattern cancels.
fn bayer(pixel: vec2i) -> f32 {
    var pattern = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    return f32(pattern[(u32(pixel.y) & 3u) * 4u + (u32(pixel.x) & 3u)]) / 16.0;
}

// `acos` with the input kept in its domain; XeGTAO's FastACos clamps too.
fn safe_acos(x: f32) -> f32 {
    return acos(clamp(x, -1.0, 1.0));
}

// XeGTAO_RotFromToMatrix, applied to a vector. HLSL's mul(matrix, vector)
// dots each matrix row with the vector, so the rows are written out here
// rather than packed into WGSL's column-major mat3x3.
// `from` is a reserved word in WGSL, hence the longer names.
fn rotate_from_to(from_dir: vec3f, to_dir: vec3f, v: vec3f) -> vec3f {
    let e = dot(from_dir, to_dir);
    let f = abs(e);
    if f > 1.0 - 0.0003 {
        return v;
    }
    let c = cross(from_dir, to_dir);
    let h = 1.0 / (1.0 + e);
    let hvx = h * c.x;
    let hvz = h * c.z;
    let hvxy = hvx * c.y;
    let hvxz = hvx * c.z;
    let hvyz = hvz * c.y;
    let row0 = vec3f(e + hvx * c.x, hvxy - c.z, hvxz + c.y);
    let row1 = vec3f(hvxy + c.z, e + h * c.y * c.y, hvyz - c.x);
    let row2 = vec3f(hvxz - c.y, hvyz + c.x, e + hvz * c.z);
    return vec3f(dot(row0, v), dot(row1, v), dot(row2, v));
}

struct Output {
    @location(0) occlusion: f32,
    @location(1) bent_normal: vec4f,
};

@fragment
fn main(@builtin(position) frag_position: vec4f) -> Output {
    var out: Output;
    let pixel = vec2i(frag_position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    // Reverse-Z clears to 0: nothing was drawn here.
    if depth <= 0.0 {
        out.occlusion = 1.0;
        out.bent_normal = vec4f(0.0);
        return out;
    }

    // XeGTAO moves the centre slightly toward the camera before it compares
    // taps. Without this FP32-depth offset, quantisation lets the visible
    // surface occlude itself, with an error that follows its facing angle.
    let pix_center_pos = flip_z(view_position(params, pixel, depth)) * 0.99999;
    let view_vec = normalize(-pix_center_pos);
    let viewspace_normal = normalize(flip_z(textureLoad(normal_texture, pixel, 0).xyz * 2.0 - 1.0));

    // The radius in pixels, clamped at both ends (R7), and the world radius
    // the clamp leaves.
    let view_depth = max(abs(pix_center_pos.z), 1e-6);
    let unclamped_pixels = params.world_radius * RADIUS_MULTIPLIER * params.pixels_per_world_unit
        / select(1.0, view_depth, params.perspective != 0u);
    let screenspace_radius = clamp(unclamped_pixels, params.pixel_radius_min, params.pixel_radius_max);
    let effect_radius = params.world_radius * RADIUS_MULTIPLIER * screenspace_radius
        / max(unclamped_pixels, 1e-6);

    let falloff_range = FALLOFF_RANGE * effect_radius;
    let falloff_from = effect_radius * (1.0 - FALLOFF_RANGE);
    let falloff_mul = -1.0 / falloff_range;
    let falloff_add = falloff_from / falloff_range + 1.0;
    let min_s = PIXEL_TOO_CLOSE_THRESHOLD / screenspace_radius;

    let noise_slice = bayer(pixel);
    let noise_sample = bayer(pixel + vec2i(2, 1));
    let resolution = vec2i(params.framebuffer_resolution);
    let slice_count = f32(params.sample_count);
    let steps_per_slice = f32(params.steps_per_slice);

    var visibility = 0.0;
    var bent_normal = vec3f(0.0);
    for (var slice = 0u; slice < params.sample_count; slice += 1u) {
        let slice_k = (f32(slice) + noise_slice) / slice_count;
        let phi = slice_k * GTAO_PI;
        let cos_phi = cos(phi);
        let sin_phi = sin(phi);
        // Screen space: pixels, with y pointing down.
        let omega = vec2f(cos_phi, -sin_phi) * screenspace_radius;

        let direction_vec = vec3f(cos_phi, sin_phi, 0.0);
        let ortho_direction_vec = direction_vec - dot(direction_vec, view_vec) * view_vec;
        let axis_vec = normalize(cross(ortho_direction_vec, view_vec));
        let projected_normal_vec = viewspace_normal - axis_vec * dot(viewspace_normal, axis_vec);
        let sign_norm = sign(dot(ortho_direction_vec, projected_normal_vec));
        let projected_normal_vec_length = length(projected_normal_vec);
        let cos_norm = saturate(dot(projected_normal_vec, view_vec) / projected_normal_vec_length);
        let n = sign_norm * safe_acos(cos_norm);

        let low_horizon_cos0 = cos(n + GTAO_PI_HALF);
        let low_horizon_cos1 = cos(n - GTAO_PI_HALF);
        var horizon_cos0 = low_horizon_cos0;
        var horizon_cos1 = low_horizon_cos1;

        for (var step_index = 0u; step_index < params.steps_per_slice; step_index += 1u) {
            let step_base_noise = f32(slice + step_index * params.steps_per_slice) * 0.6180339887498948482;
            let step_noise = fract(noise_sample + step_base_noise);
            var s = (f32(step_index) + step_noise) / steps_per_slice;
            s = pow(s, SAMPLE_DISTRIBUTION_POWER);
            s += min_s;
            let sample_offset = vec2i(round(s * omega));

            let tap0 = clamp(pixel + sample_offset, vec2i(0), resolution - 1);
            let tap1 = clamp(pixel - sample_offset, vec2i(0), resolution - 1);
            let sz0 = textureLoad(depth_texture, tap0, 0);
            let sz1 = textureLoad(depth_texture, tap1, 0);
            // A tap on nothing is infinitely far away and cannot occlude. Its
            // weight is zeroed below, so the stand-in depth is never used.
            let sample_pos0 = flip_z(view_position(params, tap0, max(sz0, 1e-7)));
            let sample_pos1 = flip_z(view_position(params, tap1, max(sz1, 1e-7)));

            let sample_delta0 = sample_pos0 - pix_center_pos;
            let sample_delta1 = sample_pos1 - pix_center_pos;
            let sample_dist0 = length(sample_delta0);
            let sample_dist1 = length(sample_delta1);
            let sample_horizon_vec0 = sample_delta0 / sample_dist0;
            let sample_horizon_vec1 = sample_delta1 / sample_dist1;

            let falloff_base0 = length(vec3f(sample_delta0.x, sample_delta0.y, sample_delta0.z * (1.0 + THIN_OCCLUDER_COMPENSATION)));
            let falloff_base1 = length(vec3f(sample_delta1.x, sample_delta1.y, sample_delta1.z * (1.0 + THIN_OCCLUDER_COMPENSATION)));
            let weight0 = select(saturate(falloff_base0 * falloff_mul + falloff_add), 0.0, sz0 <= 0.0);
            let weight1 = select(saturate(falloff_base1 * falloff_mul + falloff_add), 0.0, sz1 <= 0.0);

            var shc0 = dot(sample_horizon_vec0, view_vec);
            var shc1 = dot(sample_horizon_vec1, view_vec);
            shc0 = mix(low_horizon_cos0, shc0, weight0);
            shc1 = mix(low_horizon_cos1, shc1, weight1);
            horizon_cos0 = max(horizon_cos0, shc0);
            horizon_cos1 = max(horizon_cos1, shc1);
        }

        // XeGTAO's own fudge against slight overdarkening on steep slopes; its
        // training set scored 0.05 close to disabled.
        let normal_weight = mix(projected_normal_vec_length, 1.0, 0.05);

        let h0 = -safe_acos(horizon_cos1);
        let h1 = safe_acos(horizon_cos0);
        let iarc0 = (cos_norm + 2.0 * h0 * sin(n) - cos(2.0 * h0 - n)) / 4.0;
        let iarc1 = (cos_norm + 2.0 * h1 * sin(n) - cos(2.0 * h1 - n)) / 4.0;
        visibility += normal_weight * (iarc0 + iarc1);

        let t0 = (6.0 * sin(h0 - n) - sin(3.0 * h0 - n) + 6.0 * sin(h1 - n) - sin(3.0 * h1 - n)
            + 16.0 * sin(n) - 3.0 * (sin(h0 + n) + sin(h1 + n))) / 12.0;
        let t1 = (-cos(3.0 * h0 - n) - cos(3.0 * h1 - n) + 8.0 * cos(n)
            - 3.0 * (cos(h0 + n) + cos(h1 + n))) / 12.0;
        let local_bent_normal = vec3f(direction_vec.x * t0, direction_vec.y * t0, -t1);
        bent_normal += rotate_from_to(vec3f(0.0, 0.0, -1.0), view_vec, local_bent_normal)
            * normal_weight;
    }

    visibility /= slice_count;
    // pow of a negative is undefined in WGSL, and rounding can dip below 0.
    visibility = pow(max(visibility, 0.0), FINAL_VALUE_POWER);
    visibility = max(0.03, visibility);
    bent_normal = normalize(bent_normal);

    out.occlusion = saturate(visibility);
    // Back to our right-handed view space, mapped to [0, 1]. Alpha 1 says a
    // bent normal is present.
    out.bent_normal = vec4f(flip_z(bent_normal) * 0.5 + 0.5, 1.0);
    return out;
}
