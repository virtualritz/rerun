//! FORK DIVERGENCE: composite pass for akatela SPEC-109.
//!
//! Draw Fidget's GPU-resident voxel output into the live scene. Fidget
//! evaluates the field and writes one `GeometryPixel { normal, depth }` per
//! pixel (see the `fidget-wgpu` crate); this pass rasterizes the SDF's proxy
//! box, reads that buffer at each covered pixel, reconstructs the hit's world
//! position from its voxel depth, and writes scene colour, projected reverse-Z
//! depth and picking ids in the same phases the mesh passes use.
//!
//! Fidget owns field evaluation; `re_renderer` owns composition. The buffer is
//! read on the GPU, with no CPU readback and no per-frame texture upload.
//!
//! The `GeometryPixel` buffer belongs to Fidget's workspace, not to this
//! crate's resource pools, so the bind group is created directly from the raw
//! `wgpu` handles and the draw data holds a reference to the buffer for the
//! frame. Resizing the Fidget workspace produces a new buffer and the caller
//! must rebuild the draw data.

use glam::{Mat4, Vec3};
use smallvec::smallvec;

use super::{DrawData, DrawError, RenderContext, Renderer};
use crate::draw_phases::{DrawPhase, PickingLayerProcessor};
use crate::renderer::{DrawDataDrawable, DrawInstruction, DrawableCollectionViewInfo};
use crate::wgpu_resources::{
    BindGroupLayoutDesc, BufferDesc, GpuBindGroupLayoutHandle, GpuBuffer, GpuRenderPipelineHandle,
    GpuRenderPipelinePoolAccessor, PipelineLayoutDesc, RenderPipelineDesc,
};
use crate::{DrawableCollector, PickingLayerId, Rgba, ViewBuilder, include_shader_module};

/// One SDF composite draw: where Fidget's voxel result lives, and how to place
/// it in the scene.
///
/// `world_from_voxel` maps Fidget's voxel coordinates `(px, py, depth, 1)` to
/// world space. It must be built for the same view and projection as the view
/// it is drawn in; the fragment still projects with the frame's own
/// `projection_from_world`, so depth is byte-consistent with the mesh passes.
pub struct SdfVoxelConfiguration<'a> {
    /// Fidget's `GeometryPixel` storage buffer, one entry per pixel of
    /// [`Self::size`], row-major.
    pub geometry: &'a wgpu::Buffer,

    /// Fidget render size in pixels. The composite maps framebuffer positions
    /// into this grid, so it need not match the target's resolution.
    pub size: [u32; 2],

    /// Conservative proxy box minimum corner, in the SDF's LOCAL space.
    pub bounds_min: Vec3,

    /// Conservative proxy box maximum corner, in the SDF's LOCAL space.
    pub bounds_max: Vec3,

    /// Placement of the SDF's local space in world space.
    pub world_from_local: Mat4,

    /// Maps Fidget voxel coordinates to world space.
    pub world_from_voxel: Mat4,

    /// Placeholder flat colour until Slice D reuses the mesh matcap path.
    pub color: Rgba,

    /// Picking id written into the `PickingLayer` pass.
    pub picking_layer_id: PickingLayerId,
}

mod gpu_data {
    use crate::wgpu_buffer_types;

    /// Keep in sync with `shader/sdf_voxel.wgsl`.
    ///
    /// The trailing rows pad the uniform to 512 bytes, a multiple of the
    /// 256-byte alignment `wgpu` requires of a uniform buffer binding.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct SdfVoxelUniformBuffer {
        pub bounds_min: [f32; 4],
        pub bounds_max: [f32; 4],
        pub color: [f32; 4],
        pub picking_layer_id: [u32; 4],
        pub size: [u32; 4],
        /// Column-major, matching WGSL's `mat4x4<f32>` layout.
        pub world_from_voxel: [f32; 16],
        pub world_from_local: [f32; 16],
        pub local_from_world: [f32; 16],
        pub end_padding: [wgpu_buffer_types::PaddingRow; 15],
    }

    #[cfg(test)]
    mod tests {
        use super::SdfVoxelUniformBuffer;
        use core::mem::{offset_of, size_of};

        /// The WGSL twin is `tests::sdf_voxel_uniform_layout_matches_the_rust_struct`
        /// in `tests/shader_validation.rs`; both name the same offsets, so
        /// drift on either side fails one of them.
        ///
        /// A mismatch here writes one field where the shader reads another,
        /// with no validation error: the transform, the picking id or the
        /// proxy bounds would simply be wrong on screen.
        #[test]
        fn uniform_offsets_match_the_shader() {
            assert_eq!(offset_of!(SdfVoxelUniformBuffer, bounds_min), 0);
            assert_eq!(offset_of!(SdfVoxelUniformBuffer, bounds_max), 16);
            assert_eq!(offset_of!(SdfVoxelUniformBuffer, color), 32);
            assert_eq!(offset_of!(SdfVoxelUniformBuffer, picking_layer_id), 48);
            assert_eq!(offset_of!(SdfVoxelUniformBuffer, size), 64);
            assert_eq!(offset_of!(SdfVoxelUniformBuffer, world_from_voxel), 80);
            assert_eq!(offset_of!(SdfVoxelUniformBuffer, world_from_local), 144);
            assert_eq!(offset_of!(SdfVoxelUniformBuffer, local_from_world), 208);
            assert_eq!(
                size_of::<SdfVoxelUniformBuffer>(),
                512,
                "the uniform must stay a multiple of 256 bytes for wgpu"
            );
        }
    }
}

/// The SDF composite pass's pipelines and bind group layout.
pub struct SdfVoxelRenderer {
    shaded_pipeline: GpuRenderPipelineHandle,
    picking_pipeline: GpuRenderPipelineHandle,
    bind_group_layout: GpuBindGroupLayoutHandle,
}

/// Draw data for one SDF composite.
#[derive(Clone)]
pub struct SdfVoxelDrawData {
    /// Created directly rather than through [`crate::wgpu_resources::GpuBindGroupPool`]
    /// because the `GeometryPixel` buffer is Fidget's, not a pooled resource.
    bind_group: wgpu::BindGroup,
    /// Kept alive so the pooled uniform is not reclaimed under the raw bind
    /// group.
    _uniform_buffer: GpuBuffer,
    /// Kept alive alongside the bind group.
    _geometry_buffer: wgpu::Buffer,
    /// World-space centre, for the `Opaque` distance sort key.
    center: Vec3,
}

impl DrawData for SdfVoxelDrawData {
    type Renderer = SdfVoxelRenderer;

    fn collect_drawables(
        &self,
        view_info: &DrawableCollectionViewInfo,
        collector: &mut DrawableCollector<'_>,
    ) {
        collector.add_drawable(
            DrawPhase::Opaque,
            DrawDataDrawable::from_world_position(view_info, self.center.into(), 0),
        );
        collector.add_drawable(
            DrawPhase::PickingLayer,
            DrawDataDrawable::from_world_position(view_info, self.center.into(), 0),
        );
    }
}

impl SdfVoxelDrawData {
    /// Build one SDF composite's uniform and bind group over Fidget's
    /// `GeometryPixel` buffer.
    ///
    /// Call once per placement per frame: the camera, and therefore
    /// `world_from_voxel`, changes every frame.
    pub fn new(ctx: &RenderContext, config: &SdfVoxelConfiguration<'_>) -> Result<Self, DrawError> {
        let renderer = ctx.renderer::<SdfVoxelRenderer>()?;

        let object = config.picking_layer_id.object.0;
        let instance = config.picking_layer_id.instance.0;
        let uniform_buffer = ctx.gpu_resources.buffers.alloc(
            &ctx.device,
            &BufferDesc {
                label: "SdfVoxelDrawData".into(),
                size: std::mem::size_of::<gpu_data::SdfVoxelUniformBuffer>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        ctx.queue.write_buffer(
            &uniform_buffer,
            0,
            bytemuck::bytes_of(&gpu_data::SdfVoxelUniformBuffer {
                bounds_min: [
                    config.bounds_min.x,
                    config.bounds_min.y,
                    config.bounds_min.z,
                    0.0,
                ],
                bounds_max: [
                    config.bounds_max.x,
                    config.bounds_max.y,
                    config.bounds_max.z,
                    0.0,
                ],
                color: [
                    config.color.r(),
                    config.color.g(),
                    config.color.b(),
                    config.color.a(),
                ],
                picking_layer_id: [
                    object as u32,
                    (object >> 32) as u32,
                    instance as u32,
                    (instance >> 32) as u32,
                ],
                size: [config.size[0], config.size[1], 0, 0],
                world_from_voxel: config.world_from_voxel.to_cols_array(),
                world_from_local: config.world_from_local.to_cols_array(),
                local_from_world: config.world_from_local.inverse().to_cols_array(),
                end_padding: Default::default(),
            }),
        );

        let bind_group = {
            let layouts = ctx.gpu_resources.bind_group_layouts.resources();
            let layout = layouts.get(renderer.bind_group_layout)?;
            ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("SdfVoxel"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: config.geometry.as_entire_binding(),
                    },
                ],
            })
        };

        Ok(Self {
            bind_group,
            _uniform_buffer: uniform_buffer,
            _geometry_buffer: config.geometry.clone(),
            // The sorter compares world positions, so the local proxy centre
            // is mapped through the placement, not used raw.
            center: config
                .world_from_local
                .transform_point3((config.bounds_min + config.bounds_max) * 0.5),
        })
    }
}

impl Renderer for SdfVoxelRenderer {
    type RendererDrawData = SdfVoxelDrawData;

    fn create_renderer(ctx: &RenderContext) -> Self {
        re_tracing::profile_function!();

        let bind_group_layout = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "SdfVoxel::bind_group_layout".into(),
                entries: vec![
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                                gpu_data::SdfVoxelUniformBuffer,
                            >(
                            )
                                as _),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            // One `GeometryPixel`.
                            min_binding_size: std::num::NonZeroU64::new(16),
                        },
                        count: None,
                    },
                ],
            },
        );

        let pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &PipelineLayoutDesc {
                label: "SdfVoxel".into(),
                entries: vec![ctx.global_bindings.layout, bind_group_layout],
            },
        );

        let shader_module = ctx
            .gpu_resources
            .shader_modules
            .get_or_create(ctx, &include_shader_module!("../../shader/sdf_voxel.wgsl"));

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        };

        let shaded_pipeline = ctx.gpu_resources.render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "SdfVoxel::shaded".into(),
                pipeline_layout,
                vertex_entrypoint: "main_vs".into(),
                vertex_handle: shader_module,
                fragment_entrypoint: "fs_main_shaded".into(),
                fragment_handle: shader_module,
                vertex_buffers: smallvec![],
                render_targets: smallvec![Some(wgpu::ColorTargetState {
                    format: ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                primitive,
                depth_stencil: Some(ViewBuilder::MAIN_TARGET_DEFAULT_DEPTH_STATE),
                multisample: ViewBuilder::main_target_default_msaa_state(
                    ctx.render_config(),
                    false,
                ),
            },
        );

        let picking_pipeline = ctx.gpu_resources.render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "SdfVoxel::picking".into(),
                pipeline_layout,
                vertex_entrypoint: "main_vs".into(),
                vertex_handle: shader_module,
                fragment_entrypoint: "fs_main_picking".into(),
                fragment_handle: shader_module,
                vertex_buffers: smallvec![],
                render_targets: smallvec![Some(PickingLayerProcessor::PICKING_LAYER_FORMAT.into())],
                primitive,
                depth_stencil: PickingLayerProcessor::PICKING_LAYER_DEPTH_STATE,
                multisample: PickingLayerProcessor::PICKING_LAYER_MSAA_STATE,
            },
        );

        Self {
            shaded_pipeline,
            picking_pipeline,
            bind_group_layout,
        }
    }

    fn draw(
        &self,
        render_pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        phase: DrawPhase,
        pass: &mut wgpu::RenderPass<'_>,
        draw_instructions: &[DrawInstruction<'_, Self::RendererDrawData>],
    ) -> Result<(), DrawError> {
        let pipeline = match phase {
            DrawPhase::PickingLayer => self.picking_pipeline,
            _ => self.shaded_pipeline,
        };
        let pipeline = render_pipelines.get(pipeline)?;

        pass.set_pipeline(pipeline);
        for DrawInstruction { draw_data, .. } in draw_instructions {
            pass.set_bind_group(1, &draw_data.bind_group, &[]);
            pass.draw(0..36, 0..1);
        }

        Ok(())
    }
}
