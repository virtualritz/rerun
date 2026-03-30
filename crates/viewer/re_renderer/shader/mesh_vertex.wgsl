// See mesh.rs#MeshVertex
struct VertexIn {
    @location(0) position: vec3f,
    @location(1) color: vec4f, // gamma-space 0-1, unmultiplied
    @location(2) normal: vec3f,
    @location(3) texcoord: vec2f,
    @location(4) element_id: u32,
};

// See mesh_renderer.rs
struct InstanceIn {
    // We could alternatively store projection_from_mesh, but world position might be useful
    // in the future and this saves us a vec4f and simplifies dataflow on the cpu side.
    @location(5) world_from_mesh_row_0: vec4f,
    @location(6) world_from_mesh_row_1: vec4f,
    @location(7) world_from_mesh_row_2: vec4f,
    @location(8) world_from_mesh_normal_row_0: vec3f,
    @location(9) world_from_mesh_normal_row_1: vec3f,
    @location(10) world_from_mesh_normal_row_2: vec3f,
    @location(11) additive_tint_srgba: vec4f,
    @location(12) picking_layer_id: vec4u,
    @location(13) outline_mask_ids: vec2u,
};
