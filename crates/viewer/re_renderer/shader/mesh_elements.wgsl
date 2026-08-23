// Draws a mesh's cage edges and vertices without any line or point geometry.
//
// The mesh is already resident on the GPU for the shaded pass, so this pass
// pulls its vertices rather than binding a second copy: it draws
// `3 * triangle_count` vertices, fetches the index and the per-vertex channels
// itself, and lets the fragment stage decide what is an edge.
//
// An edge id of 0 means "this triangle side is not a cage edge" -- it is a
// triangulation diagonal or an n-gon interior cut -- so diagonals are skipped
// structurally rather than guessed at from geometry (SPEC-100 T011/T020).

#import <./global_bindings.wgsl>
#import <./utils/srgb.wgsl>

struct ElementStyle {
    // Cage edge colour, and the accent used for the hovered element.
    edge_color: vec4f,
    hover_color: vec4f,

    // Half-width in pixels for an ordinary edge and for the hovered one.
    edge_half_width_px: f32,
    hover_half_width_px: f32,

    // Radius in pixels of a vertex marker, or 0 to draw no vertices.
    vertex_radius_px: f32,

    // Encoded id of the hovered element, or 0 for none. Compared per fragment,
    // exactly as the face path already does, so hover costs O(1) rather than a
    // rebuild.
    hover_element_id: u32,
};

@group(1) @binding(0) var<storage, read> indices: array<u32>;
@group(1) @binding(1) var<storage, read> positions: array<f32>;
@group(1) @binding(2) var<storage, read> edge_ids: array<u32>;
@group(1) @binding(3) var<storage, read> vertex_ids: array<u32>;
@group(1) @binding(4) var<uniform> style: ElementStyle;

struct VertexOut {
    @builtin(position) clip_position: vec4f,

    // Which corner of this triangle we are: (1,0,0), (0,1,0) or (0,0,1).
    // Interpolated, this is the barycentric coordinate the fragment needs.
    @location(0) barycentric: vec3f,

    // Flat, so every fragment of the triangle sees the same ids rather than an
    // interpolated nonsense value.
    @location(1) @interpolate(flat) edge_id_ab: u32,
    @location(2) @interpolate(flat) edge_id_bc: u32,
    @location(3) @interpolate(flat) edge_id_ca: u32,
    @location(4) @interpolate(flat) vertex_id_a: u32,
    @location(5) @interpolate(flat) vertex_id_b: u32,
    @location(6) @interpolate(flat) vertex_id_c: u32,
};

// Positions are packed three floats to a vertex; reading them as a flat f32
// array sidesteps the std430 padding a vec3 array would carry.
fn position_at(vertex: u32) -> vec3f {
    let base = vertex * 3u;
    return vec3f(positions[base], positions[base + 1u], positions[base + 2u]);
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    let triangle = vertex_index / 3u;
    let corner = vertex_index % 3u;

    let index_a = indices[triangle * 3u];
    let index_b = indices[triangle * 3u + 1u];
    let index_c = indices[triangle * 3u + 2u];

    var out: VertexOut;

    let index = indices[vertex_index];
    out.clip_position =
        frame.projection_from_world * vec4f(position_at(index), 1.0);

    out.barycentric = vec3f(
        f32(corner == 0u),
        f32(corner == 1u),
        f32(corner == 2u),
    );

    // The edge leaving each corner toward the next one of the same face. The
    // channel is authored that way, so side a-b carries corner a's id.
    out.edge_id_ab = edge_ids[index_a];
    out.edge_id_bc = edge_ids[index_b];
    out.edge_id_ca = edge_ids[index_c];

    out.vertex_id_a = vertex_ids[index_a];
    out.vertex_id_b = vertex_ids[index_b];
    out.vertex_id_c = vertex_ids[index_c];

    return out;
}

// Distance in pixels from this fragment to the triangle side opposite `corner`.
//
// The barycentric coordinate falls linearly to 0 on that side, so dividing it
// by its own screen-space rate of change converts it to a pixel distance --
// which is what keeps stroke width constant regardless of how the triangle is
// foreshortened.
fn pixel_distance_to_side(barycentric: vec3f, corner: i32) -> f32 {
    let derivatives = fwidth(barycentric);
    let scaled = barycentric / max(derivatives, vec3f(1e-8));
    return scaled[corner];
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4f {
    // A vertex marker wins over an edge: it sits at the join of two edges, and
    // the artist is aiming at the point, not the pair of lines.
    if style.vertex_radius_px > 0.0 {
        let to_a = pixel_distance_to_side(in.barycentric, 1)
            + pixel_distance_to_side(in.barycentric, 2);
        let to_b = pixel_distance_to_side(in.barycentric, 0)
            + pixel_distance_to_side(in.barycentric, 2);
        let to_c = pixel_distance_to_side(in.barycentric, 0)
            + pixel_distance_to_side(in.barycentric, 1);

        let corner_ids = array<u32, 3>(
            in.vertex_id_a, in.vertex_id_b, in.vertex_id_c);
        let corner_distances = array<f32, 3>(to_a, to_b, to_c);

        for (var corner = 0; corner < 3; corner += 1) {
            let id = corner_ids[corner];
            if id != 0u && corner_distances[corner] <= style.vertex_radius_px {
                if id == style.hover_element_id {
                    return style.hover_color;
                }
                return style.edge_color;
            }
        }
    }

    // Side a-b is opposite corner c, and so on: the barycentric coordinate that
    // vanishes on a side is the one belonging to the corner facing it.
    let side_ids = array<u32, 3>(in.edge_id_ab, in.edge_id_bc, in.edge_id_ca);
    let opposite = array<i32, 3>(2, 0, 1);

    var nearest = 1e30;
    var nearest_id = 0u;
    for (var side = 0; side < 3; side += 1) {
        let id = side_ids[side];
        // 0 means diagonal: never stroked, so an interior cut cannot show up
        // as topology.
        if id == 0u {
            continue;
        }
        let distance = pixel_distance_to_side(in.barycentric, opposite[side]);
        if distance < nearest {
            nearest = distance;
            nearest_id = id;
        }
    }

    if nearest_id == 0u {
        discard;
    }

    let hovered = nearest_id == style.hover_element_id;
    let half_width = select(
        style.edge_half_width_px, style.hover_half_width_px, hovered);
    if nearest > half_width {
        discard;
    }

    return select(style.edge_color, style.hover_color, hovered);
}

// Picking arm: the same geometric test, writing ids instead of colour.
//
// The point of drawing edges into the ID buffer at all is that an edge then
// picks with the same rasterization that drew it -- no fat-line proxy geometry
// on the CPU, and no separate notion of "close enough" to fall out of step
// with what the user can see (SPEC-100 T030).
@fragment
fn fs_main_picking_layer(in: VertexOut) -> @location(0) vec4u {
    // A vertex wins over an edge, matching the colour pass: the marker sits on
    // the join of two edges and the user is aiming at the point.
    if style.vertex_radius_px > 0.0 {
        let to_a = pixel_distance_to_side(in.barycentric, 1)
            + pixel_distance_to_side(in.barycentric, 2);
        let to_b = pixel_distance_to_side(in.barycentric, 0)
            + pixel_distance_to_side(in.barycentric, 2);
        let to_c = pixel_distance_to_side(in.barycentric, 0)
            + pixel_distance_to_side(in.barycentric, 1);

        let corner_ids = array<u32, 3>(
            in.vertex_id_a, in.vertex_id_b, in.vertex_id_c);
        let corner_distances = array<f32, 3>(to_a, to_b, to_c);

        for (var corner = 0; corner < 3; corner += 1) {
            let id = corner_ids[corner];
            if id != 0u && corner_distances[corner] <= style.vertex_radius_px {
                return vec4u(id, 0u, 0u, 0u);
            }
        }
    }

    let side_ids = array<u32, 3>(in.edge_id_ab, in.edge_id_bc, in.edge_id_ca);
    let opposite = array<i32, 3>(2, 0, 1);

    var nearest = 1e30;
    var nearest_id = 0u;
    for (var side = 0; side < 3; side += 1) {
        let id = side_ids[side];
        if id == 0u {
            continue;
        }
        let distance = pixel_distance_to_side(in.barycentric, opposite[side]);
        if distance < nearest {
            nearest = distance;
            nearest_id = id;
        }
    }

    // Picking uses the HOVER width for every edge, not the drawn width: a
    // one-pixel line is drawn thin on purpose and would be miserable to hit if
    // the pick target were equally thin.
    if nearest_id == 0u || nearest > style.hover_half_width_px {
        discard;
    }

    return vec4u(nearest_id, 0u, 0u, 0u);
}
