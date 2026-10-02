//! FORK DIVERGENCE: direct SDF display pass for akatela SPEC-109.
//!
//! Slice A of the spec: an analytic sphere drawn by rasterizing its finite
//! proxy box, reconstructing a world-space ray per covered fragment and
//! writing the projected hit depth. The phase, depth and picking contract is
//! what the Fidget tape pass will reuse; only the fragment's field evaluation
//! changes. See `shader/sdf_tape.wgsl`.

use smallvec::smallvec;

use super::{DrawData, DrawError, RenderContext, Renderer};
use crate::allocator::create_and_fill_uniform_buffer;
use crate::draw_phases::{DrawPhase, PickingLayerProcessor};
use crate::renderer::{DrawDataDrawable, DrawInstruction, DrawableCollectionViewInfo};
use crate::wgpu_resources::{
    BindGroupDesc, BindGroupLayoutDesc, GpuBindGroup, GpuBindGroupLayoutHandle,
    GpuRenderPipelineHandle, GpuRenderPipelinePoolAccessor, PipelineLayoutDesc, RenderPipelineDesc,
};
use crate::{DrawableCollector, PickingLayerId, Rgba, ViewBuilder, include_shader_module};

/// One direct SDF draw: the finite proxy box, the analytic sphere used as the
/// Slice A fixture, its placeholder colour and its picking id.
pub struct SdfTapeConfiguration {
    /// Conservative proxy box minimum corner, in world space.
    pub bounds_min: glam::Vec3,
    /// Conservative proxy box maximum corner, in world space.
    pub bounds_max: glam::Vec3,
    /// Sphere centre in world space.
    pub center: glam::Vec3,
    /// Sphere radius.
    pub radius: f32,
    /// Flat placeholder colour until Slice D reuses the mesh matcap path.
    pub color: Rgba,
    /// Picking id written into the `PickingLayer` pass.
    pub picking_layer_id: PickingLayerId,
}

mod gpu_data {
    use crate::wgpu_buffer_types;

    /// Keep in sync with `shader/sdf_tape.wgsl`.
    ///
    /// The trailing rows pad the uniform to the 256-byte size `wgpu` requires
    /// of a uniform buffer binding.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct SdfTapeUniformBuffer {
        pub bounds_min: [f32; 4],
        pub bounds_max: [f32; 4],
        pub center_radius: [f32; 4],
        pub color: [f32; 4],
        pub picking_layer_id: [u32; 4],
        pub end_padding: [wgpu_buffer_types::PaddingRow; 11],
    }
}

/// The SDF pass's pipelines and bind group layout.
pub struct SdfTapeRenderer {
    shaded_pipeline: GpuRenderPipelineHandle,
    picking_pipeline: GpuRenderPipelineHandle,
    bind_group_layout: GpuBindGroupLayoutHandle,
}

/// Draw data for one SDF payload.
#[derive(Clone)]
pub struct SdfTapeDrawData {
    bind_group: GpuBindGroup,
    /// World-space centre, for the `Opaque` distance sort key.
    center: glam::Vec3,
}

impl DrawData for SdfTapeDrawData {
    type Renderer = SdfTapeRenderer;

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

impl SdfTapeDrawData {
    /// Upload one SDF draw's uniforms.
    pub fn new(
        ctx: &RenderContext,
        config: &SdfTapeConfiguration,
    ) -> Result<Self, crate::RendererRegistrationError> {
        let renderer = ctx.renderer::<SdfTapeRenderer>()?;

        let object = config.picking_layer_id.object.0;
        let instance = config.picking_layer_id.instance.0;
        let uniform_buffer = create_and_fill_uniform_buffer(
            ctx,
            "SdfTapeDrawData".into(),
            gpu_data::SdfTapeUniformBuffer {
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
                center_radius: [
                    config.center.x,
                    config.center.y,
                    config.center.z,
                    config.radius,
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
                end_padding: Default::default(),
            },
        );

        Ok(Self {
            bind_group: ctx.gpu_resources.bind_groups.alloc(
                &ctx.device,
                &ctx.gpu_resources,
                &BindGroupDesc {
                    label: "SdfTape".into(),
                    entries: smallvec![uniform_buffer],
                    layout: renderer.bind_group_layout,
                },
            ),
            center: config.center,
        })
    }
}

impl Renderer for SdfTapeRenderer {
    type RendererDrawData = SdfTapeDrawData;

    fn create_renderer(ctx: &RenderContext) -> Self {
        re_tracing::profile_function!();

        let bind_group_layout = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "SdfTape::bind_group_layout".into(),
                entries: vec![wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                            gpu_data::SdfTapeUniformBuffer,
                        >()
                            as _),
                    },
                    count: None,
                }],
            },
        );

        let pipeline_layout = ctx.gpu_resources.pipeline_layouts.get_or_create(
            ctx,
            &PipelineLayoutDesc {
                label: "SdfTape".into(),
                entries: vec![ctx.global_bindings.layout, bind_group_layout],
            },
        );

        let shader_module = ctx
            .gpu_resources
            .shader_modules
            .get_or_create(ctx, &include_shader_module!("../../shader/sdf_tape.wgsl"));

        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: None,
            ..Default::default()
        };

        let shaded_pipeline = ctx.gpu_resources.render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "SdfTape::shaded".into(),
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
                label: "SdfTape::picking".into(),
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
