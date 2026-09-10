//! Screen-space ambient occlusion (akatela SPEC-123).
//!
//! A matcap bakes a lighting environment into an image of a sphere. It knows
//! nothing about the scene, so two parts that touch show no contact darkening
//! and an assembly reads as floating shells. This supplies what the matcap
//! cannot.
//!
//! Three passes, all before the main pass:
//!
//! 1. A prepass draws opaque meshes into a single-sampled depth target and a
//!    view-space normal target ([`DrawPhase::OcclusionPrepass`]). The main
//!    depth buffer cannot serve: it is multisampled, and no shader may sample
//!    it.
//! 2. An estimate pass reads both and writes raw occlusion, sampling a disk
//!    around each pixel, oriented by the normal.
//! 3. A depth-aware blur removes the per-pixel sampling pattern and applies
//!    the strength exponent.
//!
//! The result is bound as `occlusion_texture` in the global bind group. The
//! mesh shader applies it to the diffuse lobe and derives specular occlusion
//! from it, so each lobe is masked on its own.
//!
//! Neither fullscreen pass binds the global bind group. The blur writes the
//! very texture that group carries, and a texture cannot be an attachment and
//! a binding in one pass. The camera terms they need travel in their own
//! uniform instead.
//!
//! [`DrawPhase::OcclusionPrepass`]: crate::DrawPhase::OcclusionPrepass

use smallvec::smallvec;

use crate::allocator::create_and_fill_uniform_buffer;
use crate::renderer::screen_triangle_vertex_shader;
use crate::view_builder::ViewBuilder;
use crate::wgpu_resources::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, GpuBindGroup, GpuRenderPipelineHandle,
    GpuRenderPipelinePoolAccessor, GpuTexture, PipelineLayoutDesc, PoolError, RenderPipelineDesc,
    TextureDesc,
};
use crate::{Label, RenderContext, include_shader_module};

/// How a view estimates ambient occlusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionConfig {
    /// Samples per pixel. More is smoother and slower.
    pub sample_count: u32,

    /// Sampling radius in world units.
    pub world_radius: f32,

    /// The pixel radius the world radius is clamped to, as `[min, max]`.
    ///
    /// Too small degenerates to noise. Too large reads most of the screen when
    /// the camera is close, and the cost falls off a cliff.
    pub pixel_radius_range: [f32; 2],

    /// Exponent on the occlusion term. `1.0` leaves it unchanged.
    pub strength: f32,
}

/// Specular occlusion from ambient occlusion (akatela SPEC-123 R4).
///
/// After Lagarde and de Rousiers, "Moving Frostbite to PBR" (2014). This is
/// the reference for `specular_occlusion` in `instanced_mesh_common.wgsl`,
/// which a shader cannot unit-test. Keep the two identical.
///
/// Roughness 1 gives `ao`. A smooth surface occludes less than `ao` facing
/// the viewer and more at grazing angles, where its reflection reaches the
/// horizon that neighbouring geometry blocks.
#[cfg(test)]
fn specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    ((n_dot_v + ao).powf((-16.0 * roughness - 1.0).exp2()) - 1.0 + ao).clamp(0.0, 1.0)
}

mod gpu_data {
    use crate::wgpu_buffer_types;

    /// Keep in sync with `occlusion/common.wgsl`.
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    pub struct OcclusionUniformBuffer {
        pub view_from_projection: wgpu_buffer_types::Mat4,

        pub framebuffer_resolution: [f32; 2],

        /// Pixels per world unit: at unit view depth for a perspective
        /// projection, outright for an orthographic one.
        pub pixels_per_world_unit: f32,

        /// 1 for a perspective projection, 0 for an orthographic one.
        pub perspective: u32,

        pub world_radius: f32,
        pub pixel_radius_min: f32,
        pub pixel_radius_max: f32,
        pub strength: f32,

        pub sample_count: u32,
        pub _padding: [u32; 3],

        pub end_padding: [wgpu_buffer_types::PaddingRow; 16 - 7],
    }
}

pub struct OcclusionProcessor {
    label: Label,
    prepass_depth: GpuTexture,
    prepass_normal: GpuTexture,
    raw_occlusion: GpuTexture,
    occlusion: GpuTexture,
    bind_group_estimate: GpuBindGroup,
    bind_group_blur: GpuBindGroup,
    render_pipeline_estimate: GpuRenderPipelineHandle,
    render_pipeline_blur: GpuRenderPipelineHandle,
}

impl OcclusionProcessor {
    /// Format of the prepass normal target: the view-space normal mapped from
    /// `[-1, 1]` to `[0, 1]`.
    ///
    /// Eight bits per axis are plenty for occlusion, and the format renders on
    /// WebGL 2 without float extensions.
    pub const NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    /// Format of the occlusion targets: one value per pixel.
    pub const OCCLUSION_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R8Unorm;

    pub fn new(
        ctx: &RenderContext,
        config: &OcclusionConfig,
        view_name: &Label,
        resolution_in_pixel: [u32; 2],
        projection_from_view: glam::Mat4,
    ) -> Self {
        re_tracing::profile_function!();
        let label: Label = format!("{view_name} - OcclusionProcessor").into();

        // ------------- Textures -------------
        let texture_pool = &ctx.gpu_resources.textures;

        let normal_desc = TextureDesc {
            label: format!("{label}::prepass_normal").into(),
            size: wgpu::Extent3d {
                width: resolution_in_pixel[0],
                height: resolution_in_pixel[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::NORMAL_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        };
        let prepass_normal = texture_pool.alloc(&ctx.device, &normal_desc);
        let prepass_depth = texture_pool.alloc(
            &ctx.device,
            &TextureDesc {
                label: format!("{label}::prepass_depth").into(),
                format: ViewBuilder::MAIN_TARGET_DEPTH_FORMAT,
                ..normal_desc
            },
        );
        let occlusion_desc = TextureDesc {
            label: format!("{label}::raw_occlusion").into(),
            format: Self::OCCLUSION_FORMAT,
            ..normal_desc
        };
        let raw_occlusion = texture_pool.alloc(&ctx.device, &occlusion_desc);
        let occlusion = texture_pool.alloc(
            &ctx.device,
            &TextureDesc {
                label: format!("{label}::occlusion").into(),
                ..occlusion_desc
            },
        );

        // ------------- Uniform -------------
        //
        // A perspective matrix carries -1 in its z column's w; an orthographic
        // one carries 0.
        let perspective = projection_from_view.z_axis.w != 0.0;
        let resolution = glam::vec2(resolution_in_pixel[0] as f32, resolution_in_pixel[1] as f32);
        let uniform_content = gpu_data::OcclusionUniformBuffer {
            view_from_projection: projection_from_view.inverse().into(),
            framebuffer_resolution: resolution.to_array(),
            // Clip-space y per view-space unit, in pixels. Unlike
            // `focal_length_in_pixels` this includes the viewport zoom, which
            // the projection matrix carries.
            pixels_per_world_unit: projection_from_view.y_axis.y * 0.5 * resolution.y,
            perspective: u32::from(perspective),
            world_radius: config.world_radius,
            pixel_radius_min: config.pixel_radius_range[0],
            pixel_radius_max: config.pixel_radius_range[1],
            strength: config.strength,
            sample_count: config.sample_count.max(1),
            _padding: [0; 3],
            end_padding: bytemuck::Zeroable::zeroed(),
        };
        let uniform = create_and_fill_uniform_buffer(
            ctx,
            format!("{label}::uniform").into(),
            uniform_content,
        );

        // ------------- Bind Groups -------------
        let depth_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let float_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let uniform_entry = |binding: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: std::num::NonZeroU64::new(std::mem::size_of::<
                    gpu_data::OcclusionUniformBuffer,
                >() as _),
            },
            count: None,
        };

        let layout_estimate = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "OcclusionProcessor::layout_estimate".into(),
                entries: vec![depth_entry(0), float_entry(1), uniform_entry(2)],
            },
        );
        let layout_blur = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: "OcclusionProcessor::layout_blur".into(),
                entries: vec![float_entry(0), depth_entry(1), uniform_entry(2)],
            },
        );

        let bind_group_estimate = ctx.gpu_resources.bind_groups.alloc(
            &ctx.device,
            &ctx.gpu_resources,
            &BindGroupDesc {
                label: format!("{label}::estimate").into(),
                entries: smallvec![
                    BindGroupEntry::DefaultTextureView(prepass_depth.handle),
                    BindGroupEntry::DefaultTextureView(prepass_normal.handle),
                    uniform.clone(),
                ],
                layout: layout_estimate,
            },
        );
        let bind_group_blur = ctx.gpu_resources.bind_groups.alloc(
            &ctx.device,
            &ctx.gpu_resources,
            &BindGroupDesc {
                label: format!("{label}::blur").into(),
                entries: smallvec![
                    BindGroupEntry::DefaultTextureView(raw_occlusion.handle),
                    BindGroupEntry::DefaultTextureView(prepass_depth.handle),
                    uniform,
                ],
                layout: layout_blur,
            },
        );

        // ------------- Render Pipelines -------------
        let estimate_desc = RenderPipelineDesc {
            label: "OcclusionProcessor::estimate".into(),
            pipeline_layout: ctx.gpu_resources.pipeline_layouts.get_or_create(
                ctx,
                &PipelineLayoutDesc {
                    label: "OcclusionProcessor::estimate".into(),
                    entries: vec![layout_estimate],
                },
            ),
            vertex_entrypoint: "main".into(),
            vertex_handle: screen_triangle_vertex_shader(ctx),
            fragment_entrypoint: "main".into(),
            fragment_handle: ctx.gpu_resources.shader_modules.get_or_create(
                ctx,
                &include_shader_module!("../../shader/occlusion/estimate.wgsl"),
            ),
            vertex_buffers: smallvec![],
            render_targets: smallvec![Some(Self::OCCLUSION_FORMAT.into())],
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
        };
        let render_pipeline_estimate = ctx
            .gpu_resources
            .render_pipelines
            .get_or_create(ctx, &estimate_desc);
        let render_pipeline_blur = ctx.gpu_resources.render_pipelines.get_or_create(
            ctx,
            &RenderPipelineDesc {
                label: "OcclusionProcessor::blur".into(),
                pipeline_layout: ctx.gpu_resources.pipeline_layouts.get_or_create(
                    ctx,
                    &PipelineLayoutDesc {
                        label: "OcclusionProcessor::blur".into(),
                        entries: vec![layout_blur],
                    },
                ),
                fragment_handle: ctx.gpu_resources.shader_modules.get_or_create(
                    ctx,
                    &include_shader_module!("../../shader/occlusion/blur.wgsl"),
                ),
                ..estimate_desc
            },
        );

        Self {
            label,
            prepass_depth,
            prepass_normal,
            raw_occlusion,
            occlusion,
            bind_group_estimate,
            bind_group_blur,
            render_pipeline_estimate,
            render_pipeline_blur,
        }
    }

    /// Begins the prepass. Draw [`crate::DrawPhase::OcclusionPrepass`] into it.
    pub fn begin_prepass<'a>(
        &'a self,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> wgpu::RenderPass<'a> {
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Label::from(format!("{} - prepass", self.label)).wgpu_label(),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.prepass_normal.default_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    // The estimate skips pixels nothing was drawn to by depth,
                    // so the normal's clear value is never read.
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.prepass_depth.default_view,
                depth_ops: Some(wgpu::Operations {
                    // The main pass's own clear, so the two depths agree.
                    load: ViewBuilder::DEFAULT_DEPTH_CLEAR,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        })
    }

    /// Estimates occlusion from the prepass, then blurs it into
    /// [`Self::occlusion_texture`].
    pub fn compute_occlusion(
        &self,
        pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), PoolError> {
        for (target, pipeline, bind_group, name) in [
            (
                &self.raw_occlusion,
                self.render_pipeline_estimate,
                &self.bind_group_estimate,
                "estimate",
            ),
            (
                &self.occlusion,
                self.render_pipeline_blur,
                &self.bind_group_blur,
                "blur",
            ),
        ] {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Label::from(format!("{} - {name}", self.label)).wgpu_label(),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &target.default_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(pipelines.get(pipeline)?);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        Ok(())
    }

    /// The finished occlusion, one value per pixel in `.r`.
    pub fn occlusion_texture(&self) -> &GpuTexture {
        &self.occlusion
    }
}

#[cfg(test)]
mod tests {
    use super::specular_occlusion;

    /// Offsets `tests/shader_validation.rs` pins against `occlusion/common.wgsl`.
    ///
    /// Two halves, as for the material uniform: the struct is private, so the
    /// integration test cannot `offset_of!` it. A mismatch in exactly this kind
    /// of uniform once hid all matcap shading.
    #[test]
    fn uniform_offsets_match_the_shader() {
        use super::gpu_data::OcclusionUniformBuffer as U;
        use std::mem::{offset_of, size_of};
        assert_eq!(offset_of!(U, view_from_projection), 0);
        assert_eq!(offset_of!(U, framebuffer_resolution), 64);
        assert_eq!(offset_of!(U, pixels_per_world_unit), 72);
        assert_eq!(offset_of!(U, perspective), 76);
        assert_eq!(offset_of!(U, world_radius), 80);
        assert_eq!(offset_of!(U, pixel_radius_min), 84);
        assert_eq!(offset_of!(U, pixel_radius_max), 88);
        assert_eq!(offset_of!(U, strength), 92);
        assert_eq!(offset_of!(U, sample_count), 96);
        // Uniform buffers are allocated in 256-byte steps.
        assert_eq!(size_of::<U>(), 256);
    }

    #[test]
    fn full_roughness_is_ambient_occlusion() {
        // R4: roughness 1 must give SO = AO.
        for ao in [0.1, 0.5, 0.9] {
            for n_dot_v in [0.0, 0.5, 1.0] {
                let so = specular_occlusion(n_dot_v, ao, 1.0);
                assert!((so - ao).abs() < 1e-3, "n.v {n_dot_v}, ao {ao}: {so}");
            }
        }
    }

    #[test]
    fn a_smooth_surface_occludes_more_at_grazing_angles() {
        // The check R4 asked for, with the sign the formula actually has:
        // less than AO facing the viewer, more at grazing. research.md.
        let ao = 0.5;
        assert!(specular_occlusion(1.0, ao, 0.0) > ao, "facing");
        assert!(specular_occlusion(0.0, ao, 0.0) < ao, "grazing");
    }

    #[test]
    fn nothing_occluded_stays_unoccluded() {
        // AO 1 must leave the specular lobe alone at every angle, or the
        // unoccluded transparent path would darken highlights.
        for roughness in [0.0, 0.5, 1.0] {
            for n_dot_v in [0.0, 0.5, 1.0] {
                assert_eq!(specular_occlusion(n_dot_v, 1.0, roughness), 1.0);
            }
        }
    }
}
