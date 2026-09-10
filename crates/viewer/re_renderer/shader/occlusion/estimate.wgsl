// Screen-space ambient occlusion estimate (akatela SPEC-123).
//
// McGuire et al., "Scalable Ambient Obscurance" (HPG 2012). Each tap is a
// nearby pixel's view-space position. A tap above the surface and inside the
// radius occludes it, and a NEAR tap occludes far more than a distant one:
// the term divides the tap's rise by its squared distance. That weighting is
// what makes a crease dark. An earlier version averaged the bare cosine and
// left a right-angled crease only 13 % darker.
//
// SAO's constants are in metres. Here each is a fraction of the radius, and
// the term is multiplied by the radius, so it has no units. The result holds
// across the viewport's whole scale range, which SPEC-123 R7 requires.

#import <../types.wgsl>
#import <../screen_triangle_vertex.wgsl>
#import <./common.wgsl>

@group(0) @binding(0)
var depth_texture: texture_depth_2d;
@group(0) @binding(1)
var normal_texture: texture_2d<f32>;
@group(0) @binding(2)
var<uniform> params: OcclusionUniformBuffer;

const TAU: f32 = 6.28318530718;

// Spiral turns across the taps. Seven keeps consecutive taps apart in angle
// for 8 to 32 taps (SAO's choice).
const SPIRAL_TURNS: f32 = 7.0;

// A tap must rise this far above the tangent plane to count, as a fraction
// of the radius (SAO: 0.012 m). Removes the self-occlusion a tessellated flat
// surface would otherwise show.
const BIAS: f32 = 0.02;

// Keeps a tap right beside the centre from dividing by almost nothing, as a
// fraction of the radius squared (SAO: 0.01 square metres).
const EPSILON: f32 = 0.01;

// SAO's intensity. One is the paper's value; the strength exponent tunes the
// rest.
const INTENSITY: f32 = 1.0;

// A 4x4 ordered pattern of rotations. The blur's 4x4 window sees each
// rotation exactly once, so the pattern cancels instead of smearing.
fn rotation_at(pixel: vec2i) -> f32 {
    var bayer = array<u32, 16>(0u, 8u, 2u, 10u, 12u, 4u, 14u, 6u, 3u, 11u, 1u, 9u, 15u, 7u, 13u, 5u);
    let index = (u32(pixel.y) & 3u) * 4u + (u32(pixel.x) & 3u);
    return f32(bayer[index]) / 16.0 * TAU;
}

@fragment
fn main(@builtin(position) frag_position: vec4f) -> @location(0) f32 {
    let pixel = vec2i(frag_position.xy);
    let depth = textureLoad(depth_texture, pixel, 0);
    // Reverse-Z clears to 0: nothing was drawn here.
    if depth <= 0.0 {
        return 1.0;
    }
    let position = view_position(params, pixel, depth);
    let normal = normalize(textureLoad(normal_texture, pixel, 0).xyz * 2.0 - 1.0);

    // The world radius as pixels at this depth, clamped at both ends (R7).
    let view_depth = max(abs(position.z), 1e-6);
    let unclamped_pixels = params.world_radius * params.pixels_per_world_unit
        / select(1.0, view_depth, params.perspective != 0u);
    let radius_pixels = clamp(unclamped_pixels, params.pixel_radius_min, params.pixel_radius_max);
    // The clamp changes the world radius the taps span. The falloff must use
    // that radius, or clamped pixels would over- or under-occlude.
    let radius = params.world_radius * radius_pixels / max(unclamped_pixels, 1e-6);
    let radius_squared = radius * radius;

    let resolution = vec2i(params.framebuffer_resolution);
    let rotation = rotation_at(pixel);
    let count = params.sample_count;
    var sum = 0.0;
    for (var i = 0u; i < count; i += 1u) {
        let fraction = (f32(i) + 0.5) / f32(count);
        let angle = fraction * SPIRAL_TURNS * TAU + rotation;
        // The square root spreads the taps evenly over the disk's area.
        let offset = vec2f(cos(angle), sin(angle)) * radius_pixels * sqrt(fraction);
        let tap = clamp(pixel + vec2i(round(offset)), vec2i(0), resolution - 1);
        let tap_depth = textureLoad(depth_texture, tap, 0);
        if tap_depth <= 0.0 {
            continue;
        }
        let to_tap = view_position(params, tap, tap_depth) - position;
        let distance_squared = dot(to_tap, to_tap);
        let rise = dot(to_tap, normal);
        // SAO's term with its units removed: the falloff cubed, times the
        // rise over the squared distance, times the radius.
        let falloff = max(1.0 - distance_squared / radius_squared, 0.0);
        sum += falloff * falloff * falloff * max(rise - BIAS * radius, 0.0) * radius
            / (distance_squared + EPSILON * radius_squared);
    }
    // SAO's normalisation: five over the tap count.
    return max(0.0, 1.0 - INTENSITY * 5.0 * sum / f32(count));
}
