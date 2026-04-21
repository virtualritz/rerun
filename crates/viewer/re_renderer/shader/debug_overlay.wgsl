// Debug overlay shader
//
// Works together with `debug_overlay.rs` to display a texture on top of the screen.
// It is meant to be used as last part of the compositor phase in order to present the debug output unfiltered.
// It's sole purpose is for developing new rendering features and it should not be used in production!
//
// The fragment shader is a blueprint for handling different texture outputs.
// *Do* edit it on the fly for debugging purposes!

#import <./types.wgsl>
#import <./global_bindings.wgsl>
#import <./utils/camera.wgsl>

struct UniformBuffer {
    screen_resolution: vec2f,
    position_in_pixel: vec2f,
    extent_in_pixel: vec2f,
    mode: u32,
    _padding: u32,
};
@group(1) @binding(0)
var<uniform> uniforms: UniformBuffer;

@group(1) @binding(1)
var debug_texture_float: texture_2d<f32>;
@group(1) @binding(2)
var debug_texture_uint: texture_2d<u32>;

// Mode options, see `DebugOverlayMode` in `debug_overlay.rs`
const ShowFloatTexture: u32 = 0u;
const ShowUintTexture: u32 = 1u;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) texcoord: vec2f,
};

@vertex
fn main_vs(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    let texcoord = vec2f(f32(vertex_index / 2u), f32(vertex_index % 2u));

    // This calculation could be simplified by pre-computing things on the CPU.
    // But this is not the point here - we want to debug this and other things rapidly by editing the shader.
    let screen_fraction = texcoord * (uniforms.extent_in_pixel / uniforms.screen_resolution) +
                        uniforms.position_in_pixel / uniforms.screen_resolution;
    let screen_ndc = screenuv_to_ndc(screen_fraction);

    var out: VertexOutput;
    out.position = vec4f(screen_ndc, 0.0, 1.0);
    out.texcoord = texcoord;
    return out;
}

// Pick ID encoding: bits [31:30] = type (0=body,1=face,2=edge,3=vertex),
// bits [29:0] = 1-indexed element ID.
const TYPE_SHIFT: u32 = 30u;
const ID_MASK: u32 = 0x3FFFFFFFu;

// HSV to RGB.
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> vec3f {
    let c = v * s;
    let x = c * (1.0 - abs(((h / 60.0) % 2.0) - 1.0));
    let m = v - c;
    var rgb: vec3f;
    if h < 60.0 { rgb = vec3f(c, x, 0.0); }
    else if h < 120.0 { rgb = vec3f(x, c, 0.0); }
    else if h < 180.0 { rgb = vec3f(0.0, c, x); }
    else if h < 240.0 { rgb = vec3f(0.0, x, c); }
    else if h < 300.0 { rgb = vec3f(x, 0.0, c); }
    else { rgb = vec3f(c, 0.0, x); }
    return rgb + vec3f(m, m, m);
}

// Decode a pick ID and map to a color.
// Hue from element ID (golden-angle), saturation/value from type.
fn pick_id_to_rgb(raw: u32) -> vec3f {
    let element_id = raw & ID_MASK;
    let pick_type = raw >> TYPE_SHIFT;
    let hue = f32(element_id) * 137.508;
    // Type-based saturation/value: face=warm, edge=cool, vertex=bright, body=gray.
    if pick_type == 1u {
        // Face: high saturation, warm.
        return hsv_to_rgb(hue % 360.0, 0.8, 0.9);
    } else if pick_type == 2u {
        // Edge: shifted hue, cooler.
        return hsv_to_rgb((hue + 180.0) % 360.0, 0.9, 0.95);
    } else if pick_type == 3u {
        // Vertex: bright, high value.
        return hsv_to_rgb((hue + 90.0) % 360.0, 0.6, 1.0);
    }
    // Body: desaturated.
    return hsv_to_rgb(hue % 360.0, 0.3, 0.7);
}

@fragment
fn main_fs(in: VertexOutput) -> @location(0) vec4f {
    if uniforms.mode == ShowFloatTexture {
        return vec4f(textureSample(debug_texture_float, nearest_sampler_clamped, in.texcoord).rgb, 1.0);
    } else if uniforms.mode == ShowUintTexture {
        let coords = vec2i(in.texcoord * vec2f(textureDimensions(debug_texture_uint).xy));
        let raw_values = textureLoad(debug_texture_uint, coords, 0);

        let id = raw_values.r;
        if id == 0u {
            // Magenta checkerboard for "no data" -- easy to distinguish from broken overlay.
            let checker = (coords.x / 8 + coords.y / 8) % 2;
            if checker == 0 {
                return vec4f(0.15, 0.0, 0.15, 1.0);
            }
            return vec4f(0.25, 0.0, 0.25, 1.0);
        }
        return vec4f(pick_id_to_rgb(id), 0.85);
    }
    return vec4f(1.0, 0.0, 1.0, 1.0);
}
