//! FORK DIVERGENCE: direct SDF display pass for akatela SPEC-109.
//!
//! Rasterize an SDF's finite proxy box, reconstruct a world-space camera ray
//! per covered fragment, march a Fidget bytecode tape with a bounded linear
//! search plus bisection, and write the projected hit depth. The Fidget
//! opcode table, tape format and interpreter are vendored in
//! `shader/fidget_ops.wgsl`; the pass, its depth and its picking are ours.
//! `shader/sdf_tape.wgsl` is the entry point.

use smallvec::smallvec;

use super::{DrawData, DrawError, RenderContext, Renderer};
use crate::allocator::create_and_fill_uniform_buffer;
use crate::draw_phases::{DrawPhase, PickingLayerProcessor};
use crate::renderer::{DrawDataDrawable, DrawInstruction, DrawableCollectionViewInfo};
use crate::wgpu_resources::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BufferDesc, GpuBindGroup,
    GpuBindGroupLayoutHandle, GpuBuffer, GpuRenderPipelineHandle, GpuRenderPipelinePoolAccessor,
    PipelineLayoutDesc, RenderPipelineDesc,
};
use crate::{DrawableCollector, PickingLayerId, Rgba, ViewBuilder, include_shader_module};

/// Default number of linear search samples between the proxy box entry and
/// exit.
///
/// The interval pre-pass replaces this with per-cell work (T013); until then
/// this is the bounded search that keeps the pass free of any Lipschitz
/// assumption.
pub const DEFAULT_SEARCH_STEPS: u32 = 128;

/// Default bisection refinements once a sign change is bracketed.
pub const DEFAULT_BISECTION_STEPS: u32 = 16;

/// One direct SDF draw: a Fidget bytecode tape, its variable values, the
/// finite proxy box that clips rays, and the picking id.
pub struct SdfTapeConfiguration {
    /// Conservative proxy box minimum corner, in the SDF's LOCAL space.
    pub bounds_min: glam::Vec3,
    /// Conservative proxy box maximum corner, in the SDF's LOCAL space.
    pub bounds_max: glam::Vec3,
    /// Placement of the SDF's local space in world space. The proxy box and
    /// the tape are local; rays are mapped into local space to march and the
    /// hit is mapped back for depth, so a transformed or instanced SDF follows
    /// its placement without rebuilding the tape.
    pub world_from_local: glam::Mat4,
    /// Placeholder flat colour until Slice D reuses the mesh matcap path.
    pub color: Rgba,
    /// Picking id written into the `PickingLayer` pass.
    pub picking_layer_id: PickingLayerId,
    /// Linear search samples between the proxy entry and exit.
    pub search_steps: u32,
    /// Bisection refinements after a bracketed sign change.
    pub bisection_steps: u32,
}

/// The immutable GPU half of an SDF: its bytecode and free-variable buffers,
/// plus the axis variable indices.
///
/// Cache this on `sdf_hash` and reuse it across placements: moving or
/// re-picking an SDF only rewrites the small per-draw uniform in
/// [`SdfTapeDrawData`], never the tape.
#[derive(Clone)]
pub struct SdfTapeResources {
    tape_buffer: GpuBuffer,
    variables_buffer: GpuBuffer,
    /// Variable indices of `x`, `y`, `z`; `u32::MAX` when the tape has none.
    pub axes: [u32; 3],
}

impl SdfTapeResources {
    /// Upload a tape, its free-variable values and the axis variable indices.
    pub fn new(ctx: &RenderContext, tape: &[u32], variables: &[f32], axes: [u32; 3]) -> Self {
        let tape_buffer = ctx.gpu_resources.buffers.alloc(
            &ctx.device,
            &BufferDesc {
                label: "SdfTapeResources::tape".into(),
                size: (tape.len() * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        ctx.queue
            .write_buffer(&tape_buffer, 0, bytemuck::cast_slice(tape));

        // An absent or empty variable list still needs a bound buffer.
        let empty_variables = [0.0_f32];
        let variables = if variables.is_empty() {
            &empty_variables[..]
        } else {
            variables
        };
        let variables_buffer = ctx.gpu_resources.buffers.alloc(
            &ctx.device,
            &BufferDesc {
                label: "SdfTapeResources::variables".into(),
                size: (variables.len() * std::mem::size_of::<f32>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        );
        ctx.queue
            .write_buffer(&variables_buffer, 0, bytemuck::cast_slice(variables));

        Self {
            tape_buffer,
            variables_buffer,
            axes,
        }
    }
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
        pub color: [f32; 4],
        pub picking_layer_id: [u32; 4],
        pub axes: [u32; 4],
        pub counts: [u32; 4],
        /// Column-major, matching WGSL's `mat4x4<f32>` layout.
        pub world_from_local: [f32; 16],
        pub local_from_world: [f32; 16],
        pub end_padding: [wgpu_buffer_types::PaddingRow; 2],
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
    /// Build one SDF draw's uniform and bind group over cached `resources`.
    ///
    /// Call once per placement per frame: the tape buffers are shared, so this
    /// only allocates the small uniform and the bind group that joins it to
    /// the tape.
    pub fn new(
        ctx: &RenderContext,
        resources: &SdfTapeResources,
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
                axes: [resources.axes[0], resources.axes[1], resources.axes[2], 0],
                counts: [config.search_steps, config.bisection_steps, 0, 0],
                world_from_local: config.world_from_local.to_cols_array(),
                local_from_world: config.world_from_local.inverse().to_cols_array(),
                end_padding: Default::default(),
            },
        );

        Ok(Self {
            bind_group: ctx.gpu_resources.bind_groups.alloc(
                &ctx.device,
                &ctx.gpu_resources,
                &BindGroupDesc {
                    label: "SdfTape".into(),
                    entries: smallvec![
                        uniform_buffer,
                        BindGroupEntry::Buffer {
                            handle: resources.tape_buffer.handle,
                            offset: 0,
                            size: None,
                        },
                        BindGroupEntry::Buffer {
                            handle: resources.variables_buffer.handle,
                            offset: 0,
                            size: None,
                        },
                    ],
                    layout: renderer.bind_group_layout,
                },
            ),
            // The sorter compares world positions, so the local proxy centre
            // is mapped through the placement, not used raw.
            center: config
                .world_from_local
                .transform_point3((config.bounds_min + config.bounds_max) * 0.5),
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
                entries: vec![
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                                gpu_data::SdfTapeUniformBuffer,
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
                            // One bytecode word pair.
                            min_binding_size: std::num::NonZeroU64::new(8),
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: std::num::NonZeroU64::new(4),
                        },
                        count: None,
                    },
                ],
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
