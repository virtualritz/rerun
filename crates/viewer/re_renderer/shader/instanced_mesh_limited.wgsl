// Instanced mesh shader, `Limited`-device-tier variant (WebGL2): storage
// buffers don't exist on this tier, so selection ids live in a fixed-capacity
// uniform buffer instead. See `instanced_mesh_common.wgsl` for the shared body
// and `instanced_mesh.wgsl` for the full-WebGPU variant.

// Keep in sync with `MAX_SELECTED_IDS_LIMITED_TIER` in mesh_renderer.rs:
// 16 header bytes + 1023 * 16 id bytes = 16384 bytes, the guaranteed minimum
// uniform buffer size on WebGL2. Capacity: 1023 * 4 = 4092 ids.
struct SelectedIds {
    // `.x` = number of valid ids; `.yzw` unused padding.
    count: vec4u,
    ids: array<vec4u, 1023>,
};

@group(2) @binding(0)
var<uniform> selected_ids: SelectedIds;

/// Check if element_id is in the selection uniform buffer.
fn is_selected(element_id: u32) -> bool {
    let len = selected_ids.count.x;
    for (var i = 0u; i < len; i++) {
        if selected_ids.ids[i / 4u][i % 4u] == element_id {
            return true;
        }
    }
    return false;
}

#import <./instanced_mesh_common.wgsl>
