// Depth-aware blur of the occlusion estimate (akatela SPEC-123).
//
// Removes the estimate's rotation pattern without bleeding a crease's
// occlusion onto the surface in front of it. Also applies the strength
// exponent (R8), so the mesh shader reads a finished value.

#import <../types.wgsl>
#import <../screen_triangle_vertex.wgsl>
#import <./common.wgsl>

@group(0) @binding(0)
var raw_occlusion: texture_2d<f32>;
@group(0) @binding(1)
var depth_texture: texture_depth_2d;
@group(0) @binding(2)
var<uniform> params: OcclusionUniformBuffer;

// Relative view-depth difference past which a neighbour stops counting.
const DEPTH_TOLERANCE: f32 = 0.05;

// Interleaved gradient noise (Jimenez, SIGGRAPH 2014) in [-1, 1]. The same
// formula as `dithereens::InterleavedGradientNoise` with seed 0, so the
// viewport and akatela's menu swatches dither alike.
fn interleaved_gradient_noise(pixel: vec2i) -> f32 {
    let value = fract(52.982918 * fract(0.06711056 * f32(pixel.x) + 0.00583715 * f32(pixel.y)));
    return value * 2.0 - 1.0;
}

@fragment
fn main(@builtin(position) frag_position: vec4f) -> @location(0) f32 {
    let pixel = vec2i(frag_position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    if depth <= 0.0 {
        return 1.0;
    }
    let centre_depth = abs(view_position(params, pixel, depth).z);
    let resolution = vec2i(params.framebuffer_resolution);

    // A 4x4 window covers one period of the estimate's rotation pattern.
    var sum = 0.0;
    var weight_sum = 0.0;
    for (var y = -1; y <= 2; y += 1) {
        for (var x = -1; x <= 2; x += 1) {
            let tap = clamp(pixel + vec2i(x, y), vec2i(0), resolution - 1);
            let tap_depth = textureLoad(depth_texture, tap, 0);
            if tap_depth <= 0.0 {
                continue;
            }
            let difference = abs(abs(view_position(params, tap, tap_depth).z) - centre_depth);
            let weight = max(1.0 - difference / (DEPTH_TOLERANCE * centre_depth), 0.0);
            sum += textureLoad(raw_occlusion, tap, 0).r * weight;
            weight_sum += weight;
        }
    }
    // The centre pixel always weighs 1, so `weight_sum` is never zero here.
    let occlusion = pow(sum / weight_sum, params.strength);
    // Half a step of noise before the 8-bit target rounds, as
    // `dithereens::simple_dither_2d` does. A smooth crease would band without.
    return clamp(occlusion + interleaved_gradient_noise(pixel) * (0.5 / 255.0), 0.0, 1.0);
}
