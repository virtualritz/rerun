// Dithering for 8-bit sRGB render targets.

#import <../global_bindings.wgsl>
#import <./srgb.wgsl>

// Interleaved gradient noise (Jimenez, SIGGRAPH 2014) in [-1, 1]. The same
// formula as `dithereens::InterleavedGradientNoise` with seed 0, so the
// viewport and akatela's menu swatches dither alike.
fn interleaved_gradient_noise_signed(pixel: vec2f) -> f32 {
    let value = fract(52.982918 * fract(0.06711056 * pixel.x + 0.00583715 * pixel.y));
    return value * 2.0 - 1.0;
}

// Adds half a quantization step of noise to a linear color, in sRGB space,
// before an `Rgba8UnormSrgb` target rounds it.
//
// A smooth gradient -- a matcap falloff, an occlusion band -- spans only a
// few 8-bit steps across many pixels and shows each step as a band. The noise
// breaks those contours into grain the eye averages away. It is applied in
// sRGB space because that is where the target's steps are evenly spaced.
//
// Deterministic rendering (snapshot tests) skips it.
fn dither_linear_for_srgb8(color_linear: vec3f, pixel: vec2f) -> vec3f {
    if frame.deterministic_rendering == 1u {
        return color_linear;
    }
    let noise = interleaved_gradient_noise_signed(pixel) * (0.5 / 255.0);
    let srgb = saturate(srgb_from_linear(saturate(color_linear)) + noise);
    return linear_from_srgb(srgb);
}
