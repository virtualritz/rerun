// Instanced mesh shader, full-WebGPU-tier variant: selection ids live in a
// runtime-sized storage buffer. See `instanced_mesh_common.wgsl` for the
// shared body and `instanced_mesh_limited.wgsl` for the WebGL2 variant.

@group(2) @binding(0)
var<storage, read> selected_ids: array<u32>;

/// Check if element_id is in the selection SSBO.
fn is_selected(element_id: u32) -> bool {
    let len = arrayLength(&selected_ids);
    for (var i = 0u; i < len; i++) {
        if selected_ids[i] == element_id {
            return true;
        }
    }
    return false;
}

#import <./instanced_mesh_common.wgsl>
