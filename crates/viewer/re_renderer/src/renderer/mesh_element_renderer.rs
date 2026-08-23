//! Draws a mesh's cage edges and vertices from the mesh already on the GPU.
//!
//! The app used to build whole-mesh line geometry on the CPU and upload it
//! every frame, which is why a dense mesh froze the viewport as soon as the
//! wireframe was on: camera motion means redraw, and every redraw re-walked
//! and re-uploaded millions of segments. This pass draws the same edges from
//! the resident mesh, so the per-frame cost is a draw call (SPEC-100 T020).

use smallvec::smallvec;

use super::{
    DrawData, DrawError, DrawPhase, GpuRenderPipelinePoolAccessor, RenderContext, Renderer,
};
use crate::renderer::{DrawDataDrawable, DrawInstruction, DrawableCollectionViewInfo};
use crate::view_builder::ViewBuilder;
use crate::wgpu_resources::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, BufferDesc, GpuBindGroup,
    GpuBindGroupLayoutHandle, GpuRenderPipelineHandle, PipelineLayoutDesc, RenderPipelineDesc,
};
use crate::{DrawableCollector, include_shader_module};

/// Style and hover state, mirroring what the CPU path resolved per segment.
#[repr(C, align(16))]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ElementStyleUniform {
    edge_color: [f32; 4],
    hover_color: [f32; 4],
    edge_half_width_px: f32,
    hover_half_width_px: f32,
    vertex_radius_px: f32,
    hover_element_id: u32,
}

/// What one mesh contributes to the pass.
pub struct MeshElementBatch {
    bind_group: GpuBindGroup,

    /// `3 * triangle_count`: the pass pulls vertices rather than indexing.
    num_vertices: u32,
}

#[derive(Clone)]
pub struct MeshElementDrawData {
    batches: std::sync::Arc<Vec<MeshElementBatch>>,
}

impl DrawData for MeshElementDrawData {
    type Renderer = MeshElementRenderer;

    fn collect_drawables(
        &self,
        _view_info: &DrawableCollectionViewInfo,
        collector: &mut DrawableCollector<'_>,
    ) {
        if self.batches.is_empty() {
            return;
        }
        // Transparent: edges sit on top of the shaded surface, and stroking
        // discards everywhere else, so depth-writing them would punch holes.
        for phase in [DrawPhase::Transparent, DrawPhase::PickingLayer] {
            collector.add_drawable(
                phase,
                DrawDataDrawable {
                    distance_sort_key: 0.0,
                    secondary_sort_key: 0.0,
                    draw_data_payload: 0,
                },
            );
        }
    }
}

/// How one mesh should be stroked.
pub struct MeshElementStyle {
    pub edge_color: [f32; 4],
    pub hover_color: [f32; 4],
    pub edge_half_width_px: f32,
    pub hover_half_width_px: f32,

    /// 0 draws no vertex markers.
    pub vertex_radius_px: f32,

    /// Encoded id of the hovered element, or 0. Compared per fragment, so
    /// hovering costs nothing beyond a uniform write.
    pub hover_element_id: u32,
}

impl MeshElementDrawData {
    pub fn new(
        ctx: &RenderContext,
        meshes: impl IntoIterator<Item = (std::sync::Arc<crate::mesh::GpuMesh>, MeshElementStyle)>,
    ) -> Self {
        let renderer = ctx.renderer::<MeshElementRenderer>();

        let batches = meshes
            .into_iter()
            .map(|(mesh, style)| {
                let uniform = ElementStyleUniform {
                    edge_color: style.edge_color,
                    hover_color: style.hover_color,
                    edge_half_width_px: style.edge_half_width_px,
                    hover_half_width_px: style.hover_half_width_px,
                    vertex_radius_px: style.vertex_radius_px,
                    hover_element_id: style.hover_element_id,
                };
                let style_buffer = ctx.gpu_resources.buffers.alloc(
                    &ctx.device,
                    &BufferDesc {
                        label: "MeshElementDrawData::style".into(),
                        size: std::mem::size_of::<ElementStyleUniform>() as u64,
                        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    },
                );
                ctx.queue
                    .write_buffer(&style_buffer, 0, bytemuck::bytes_of(&uniform));

                let bind_group = ctx.gpu_resources.bind_groups.alloc(
                    &ctx.device,
                    &ctx.gpu_resources,
                    &BindGroupDesc {
                        label: "MeshElementDrawData::bind_group".into(),
                        entries: smallvec![
                            BindGroupEntry::Buffer {
                                handle: mesh.index_buffer.handle,
                                offset: mesh.index_buffer_range.start,
                                size: std::num::NonZeroU64::new(
                                    mesh.index_buffer_range.end - mesh.index_buffer_range.start
                                ),
                            },
                            BindGroupEntry::Buffer {
                                handle: mesh.vertex_buffer_combined.handle,
                                offset: mesh.vertex_buffer_positions_range.start,
                                size: std::num::NonZeroU64::new(
                                    mesh.vertex_buffer_positions_range.end
                                        - mesh.vertex_buffer_positions_range.start
                                ),
                            },
                            BindGroupEntry::Buffer {
                                handle: mesh.vertex_buffer_combined.handle,
                                offset: mesh.vertex_buffer_edge_ids_range.start,
                                size: std::num::NonZeroU64::new(
                                    mesh.vertex_buffer_edge_ids_range.end
                                        - mesh.vertex_buffer_edge_ids_range.start
                                ),
                            },
                            BindGroupEntry::Buffer {
                                handle: mesh.vertex_buffer_combined.handle,
                                offset: mesh.vertex_buffer_topology_ids_range.start,
                                size: std::num::NonZeroU64::new(
                                    mesh.vertex_buffer_topology_ids_range.end
                                        - mesh.vertex_buffer_topology_ids_range.start
                                ),
                            },
                            BindGroupEntry::Buffer {
                                handle: style_buffer.handle,
                                offset: 0,
                                size: std::num::NonZeroU64::new(
                                    std::mem::size_of::<ElementStyleUniform>() as u64
                                ),
                            },
                        ],
                        layout: renderer.bind_group_layout,
                    },
                );

                let num_indices = (mesh.index_buffer_range.end - mesh.index_buffer_range.start)
                    / std::mem::size_of::<u32>() as u64;

                MeshElementBatch {
                    bind_group,
                    num_vertices: num_indices as u32,
                }
            })
            .collect();

        Self {
            batches: std::sync::Arc::new(batches),
        }
    }
}

pub struct MeshElementRenderer {
    render_pipeline: GpuRenderPipelineHandle,
    picking_pipeline: GpuRenderPipelineHandle,
    bind_group_layout: GpuBindGroupLayoutHandle,
}

impl Renderer for MeshElementRenderer {
    type RendererDrawData = MeshElementDrawData;

    fn create_renderer(ctx: &RenderContext) -> Self {
        let storage = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let bind_group_layout = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "MeshElementRenderer::bind_group_layout".into(),
                entries: vec![
                    storage(0),
                    storage(1),
                    storage(2),
                    storage(3),
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: (std::mem::size_of::<ElementStyleUniform>() as u64)
                                .try_into()
                                .ok(),
                        },
                        count: None,
                    },
                ],
            },
        );

        let shader = ctx.gpu_resources.shader_modules.get_or_create(
            ctx,
            &include_shader_module!("../../shader/mesh_elements.wgsl"),
        );

        let render_pipeline_desc = RenderPipelineDesc {
                label: "MeshElementRenderer::render_pipeline".into(),
                pipeline_layout: ctx.gpu_resources.pipeline_layouts.get_or_create(
                    ctx,
                    &PipelineLayoutDesc {
                        label: "MeshElementRenderer".into(),
                        entries: vec![ctx.global_bindings.layout, bind_group_layout],
                    },
                ),
                vertex_entrypoint: "vs_main".into(),
                vertex_handle: shader,
                fragment_entrypoint: "fs_main".into(),
                fragment_handle: shader,
                // Nothing is bound as a vertex attribute: the shader pulls.
                vertex_buffers: smallvec![],
                render_targets: smallvec![Some(wgpu::ColorTargetState {
                    format: ViewBuilder::MAIN_TARGET_COLOR_FORMAT,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    // Edges are drawn from both sides: a back face's cage edge
                    // is still an edge of the model.
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: ViewBuilder::MAIN_TARGET_DEPTH_FORMAT,
                    // Equal-or-nearer, so an edge lying exactly on the surface
                    // it belongs to is not z-fought out of existence.
                    depth_compare: Some(wgpu::CompareFunction::GreaterEqual),
                    depth_write_enabled: Some(false),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: ViewBuilder::main_target_default_msaa_state(ctx.render_config(), false),
        };
        let render_pipeline = ctx
            .gpu_resources
            .render_pipelines
            .get_or_create(ctx, &render_pipeline_desc);

        // Same geometry, same test, different target: ids instead of colour.
        // Picking writes depth normally -- an edge behind a surface must not
        // win the pick -- where the colour pass deliberately does not.
        let picking_pipeline = ctx.gpu_resources.render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "MeshElementRenderer::picking_pipeline".into(),
                fragment_entrypoint: "fs_main_picking_layer".into(),
                render_targets: smallvec![Some(
                    crate::PickingLayerProcessor::PICKING_LAYER_FORMAT.into()
                )],
                depth_stencil: crate::PickingLayerProcessor::PICKING_LAYER_DEPTH_STATE,
                multisample: crate::PickingLayerProcessor::PICKING_LAYER_MSAA_STATE,
                ..render_pipeline_desc
            },
        );

        Self {
            render_pipeline,
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
        let handle = match phase {
            DrawPhase::PickingLayer => self.picking_pipeline,
            _ => self.render_pipeline,
        };
        let pipeline = render_pipelines.get(handle)?;
        pass.set_pipeline(pipeline);

        for instruction in draw_instructions {
            for batch in instruction.draw_data.batches.iter() {
                if batch.num_vertices == 0 {
                    continue;
                }
                pass.set_bind_group(1, &batch.bind_group, &[]);
                pass.draw(0..batch.num_vertices, 0..1);
            }
        }

        Ok(())
    }
}
