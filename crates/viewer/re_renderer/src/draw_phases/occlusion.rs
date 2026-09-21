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
//! 2. An estimate pass reads both and writes raw occlusion. [`OcclusionMethod`]
//!    picks how: disk sampling, or a horizon search that also yields a bent
//!    normal.
//! 3. A depth-aware blur removes the per-pixel sampling pattern and applies
//!    the strength exponent.
//!
//! The results are bound in the global bind group: occlusion at binding 4 and
//! the bent normal at binding 5. The mesh shader applies occlusion to the
//! diffuse lobe and derives specular occlusion from it, so each lobe is masked
//! on its own.
//!
//! Neither fullscreen pass binds the global bind group. The blur writes the
//! very textures that group carries, and a texture cannot be an attachment and
//! a binding in one pass. The camera terms they need travel in their own
//! uniform instead.
//!
//! [`DrawPhase::OcclusionPrepass`]: crate::DrawPhase::OcclusionPrepass

use smallvec::{SmallVec, smallvec};

use crate::allocator::create_and_fill_uniform_buffer;
use crate::renderer::screen_triangle_vertex_shader;
use crate::view_builder::ViewBuilder;
use crate::wgpu_resources::{
    BindGroupDesc, BindGroupEntry, BindGroupLayoutDesc, GpuBindGroup, GpuRenderPipelineHandle,
    GpuRenderPipelinePoolAccessor, GpuTexture, PipelineLayoutDesc, PoolError, RenderPipelineDesc,
    TextureDesc,
};
use crate::{Label, RenderContext, include_shader_module};

/// How the estimate pass finds occlusion.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OcclusionMethod {
    /// Normal-oriented disk sampling (SAO). Specular occlusion then comes from
    /// the analytic formula of akatela SPEC-123 R4.
    Disk {
        /// Samples per pixel.
        sample_count: u32,
    },

    /// Horizon-based occlusion (GTAO), ported from Intel's `XeGTAO`. It also
    /// yields a bent normal, so specular occlusion comes from the overlap of the
    /// reflection cone with the open cone around it (SPEC-123 D3a).
    Horizon {
        /// Directions searched per pixel.
        slice_count: u32,

        /// Samples along each direction, on each side of the pixel.
        steps_per_slice: u32,
    },
}

/// Which occlusion term the mesh shader shows instead of the shading.
///
/// A debug view (akatela SPEC-123). It costs no pass: the choice travels in
/// the frame uniform, and the mesh shader returns the term as grey.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum OcclusionDebugView {
    /// Shade as usual.
    #[default]
    Off = 0,

    /// The ambient occlusion the fragment reads, which darkens the diffuse
    /// lobe.
    Ambient = 1,

    /// The reflection occlusion the specular lobe is masked by. A surface
    /// with no specular lobe shows white: it has nothing to occlude.
    Specular = 2,
}

/// How a view estimates ambient occlusion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OcclusionConfig {
    pub method: OcclusionMethod,

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
/// After Lagarde and de Rousiers, "Moving Frostbite to PBR" (2014). This is the
/// reference for `specular_occlusion` in `instanced_mesh_common.wgsl`, which a
/// shader cannot unit-test. Keep the two identical.
///
/// Roughness 1 gives `ao`. A smooth surface occludes less than `ao` facing the
/// viewer and more at grazing angles, where its reflection reaches the horizon
/// that neighbouring geometry blocks.
#[cfg(test)]
fn specular_occlusion(n_dot_v: f32, ao: f32, roughness: f32) -> f32 {
    ((n_dot_v + ao).powf((-16.0 * roughness - 1.0).exp2()) - 1.0 + ao).clamp(0.0, 1.0)
}

/// The solid angle where two spherical caps overlap, from the cosines of their
/// half-angles and of the angle between their axes.
///
/// After Oat and Sander (2007), as Unity's SRP core implements it. The reference
/// for `spherical_cap_intersection` in `instanced_mesh_common.wgsl`.
#[cfg(test)]
fn spherical_cap_intersection(cos_c1: f32, cos_c2: f32, cos_b: f32) -> f32 {
    use std::f32::consts::TAU;
    let r1 = cos_c1.clamp(-1.0, 1.0).acos();
    let r2 = cos_c2.clamp(-1.0, 1.0).acos();
    let rd = cos_b.clamp(-1.0, 1.0).acos();
    let smaller_cap = TAU - TAU * cos_c1.max(cos_c2);
    if rd <= r1.max(r2) - r1.min(r2) {
        smaller_cap
    } else if rd >= r1 + r2 {
        0.0
    } else {
        let diff = (r1 - r2).abs();
        let den = r1 + r2 - diff;
        let x = 1.0 - ((rd - diff) / den.max(0.0001)).clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x) * smaller_cap
    }
}

/// Specular occlusion from a bent cone (akatela SPEC-123 D3a).
///
/// After Jimenez et al., "Practical Realtime Strategies for Accurate Indirect
/// Occlusion" (SIGGRAPH 2016), slide 129: the share of the reflection cone that
/// falls inside the visible cone around the bent normal. `cos_between` is the
/// cosine between the reflection direction and the bent normal. The reference
/// for `specular_occlusion_cone` in `instanced_mesh_common.wgsl`.
#[cfg(test)]
fn specular_occlusion_cone(cos_between: f32, ao: f32, roughness: f32) -> f32 {
    let cos_visible = (1.0 - ao).max(0.0).sqrt();
    let roughness = roughness.max(0.01);
    let cos_reflection = (-std::f32::consts::LOG2_10 * roughness * roughness).exp2();
    (spherical_cap_intersection(cos_visible, cos_reflection, cos_between)
        / (std::f32::consts::TAU * (1.0 - cos_reflection)))
        .clamp(0.0, 1.0)
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

        /// Disk samples, or horizon slices.
        pub sample_count: u32,

        /// Horizon method only: samples along each slice direction.
        pub steps_per_slice: u32,

        pub _padding: [u32; 2],

        pub end_padding: [wgpu_buffer_types::PaddingRow; 16 - 7],
    }
}

pub struct OcclusionProcessor {
    label: Label,
    prepass_depth: GpuTexture,
    prepass_normal: GpuTexture,
    raw_occlusion: GpuTexture,
    occlusion: GpuTexture,

    /// Raw and blurred bent normals, for the horizon method only.
    bent_normals: Option<[GpuTexture; 2]>,

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

    /// Format of the bent-normal targets: the view-space bent normal mapped to
    /// `[0, 1]` in `.rgb`, and 1 in `.a` where it was computed.
    ///
    /// The alpha is how the mesh shader tells a bent normal from none: a view
    /// without one binds an all-zero texture.
    pub const BENT_NORMAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    pub fn new(
        ctx: &RenderContext,
        config: &OcclusionConfig,
        view_name: &Label,
        resolution_in_pixel: [u32; 2],
        projection_from_view: glam::Mat4,
    ) -> Self {
        re_tracing::profile_function!();
        let label: Label = format!("{view_name} - OcclusionProcessor").into();
        let (sample_count, steps_per_slice, horizon) = match config.method {
            OcclusionMethod::Disk { sample_count } => (sample_count, 0, false),
            OcclusionMethod::Horizon {
                slice_count,
                steps_per_slice,
            } => (slice_count, steps_per_slice.max(1), true),
        };

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
        // Copyable, so a test can read the results back.
        let result_usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC;
        let occlusion_desc = TextureDesc {
            label: format!("{label}::raw_occlusion").into(),
            format: Self::OCCLUSION_FORMAT,
            usage: result_usage,
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
        let bent_normals = if horizon {
            let bent_desc = TextureDesc {
                label: format!("{label}::raw_bent_normal").into(),
                format: Self::BENT_NORMAL_FORMAT,
                usage: result_usage,
                ..normal_desc
            };
            Some([
                texture_pool.alloc(&ctx.device, &bent_desc),
                texture_pool.alloc(
                    &ctx.device,
                    &TextureDesc {
                        label: format!("{label}::bent_normal").into(),
                        ..bent_desc
                    },
                ),
            ])
        } else {
            None
        };

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
            sample_count: sample_count.max(1),
            steps_per_slice,
            _padding: [0; 2],
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
        // The prepass normal at 3, so the blur can reject a neighbour across a
        // convex edge -- see `blur.wgsl`, where the depth weight alone could
        // not see one. The horizon-only bent normal comes last: the bind-group
        // pool numbers entries by position, so an optional binding anywhere
        // else would leave a gap the disk method's bind group cannot match.
        let mut blur_entries = vec![
            float_entry(0),
            depth_entry(1),
            uniform_entry(2),
            float_entry(3),
        ];
        if horizon {
            blur_entries.push(float_entry(4));
        }
        let layout_blur = ctx.gpu_resources.bind_group_layouts.get_or_create(
            &ctx.device,
            &BindGroupLayoutDesc {
                label: if horizon {
                    "OcclusionProcessor::layout_blur_bent"
                } else {
                    "OcclusionProcessor::layout_blur"
                }
                .into(),
                entries: blur_entries,
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
        let mut blur_bindings: SmallVec<[BindGroupEntry; 4]> = smallvec![
            BindGroupEntry::DefaultTextureView(raw_occlusion.handle),
            BindGroupEntry::DefaultTextureView(prepass_depth.handle),
            uniform,
            BindGroupEntry::DefaultTextureView(prepass_normal.handle),
        ];
        if let Some([raw_bent_normal, _]) = &bent_normals {
            blur_bindings.push(BindGroupEntry::DefaultTextureView(raw_bent_normal.handle));
        }
        let bind_group_blur = ctx.gpu_resources.bind_groups.alloc(
            &ctx.device,
            &ctx.gpu_resources,
            &BindGroupDesc {
                label: format!("{label}::blur").into(),
                entries: blur_bindings,
                layout: layout_blur,
            },
        );

        // ------------- Render Pipelines -------------
        let mut targets: SmallVec<[Option<wgpu::ColorTargetState>; 4]> =
            smallvec![Some(Self::OCCLUSION_FORMAT.into())];
        if horizon {
            targets.push(Some(Self::BENT_NORMAL_FORMAT.into()));
        }
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
                &if horizon {
                    include_shader_module!("../../shader/occlusion/horizon.wgsl")
                } else {
                    include_shader_module!("../../shader/occlusion/estimate.wgsl")
                },
            ),
            vertex_buffers: smallvec![],
            render_targets: targets,
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
                fragment_entrypoint: if horizon {
                    "main_with_bent_normal"
                } else {
                    "main"
                }
                .into(),
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
            bent_normals,
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
    /// [`Self::occlusion_texture`] and, for the horizon method,
    /// [`Self::bent_normal_texture`].
    pub fn compute_occlusion(
        &self,
        pipelines: &GpuRenderPipelinePoolAccessor<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<(), PoolError> {
        let bent = self.bent_normals.as_ref();
        let passes = [
            (
                "estimate",
                self.render_pipeline_estimate,
                &self.bind_group_estimate,
                &self.raw_occlusion,
                bent.map(|[raw, _]| raw),
            ),
            (
                "blur",
                self.render_pipeline_blur,
                &self.bind_group_blur,
                &self.occlusion,
                bent.map(|[_, blurred]| blurred),
            ),
        ];
        for (name, pipeline, bind_group, occlusion_target, bent_target) in passes {
            let mut attachments: SmallVec<[Option<wgpu::RenderPassColorAttachment<'_>>; 2]> =
                smallvec![Some(wgpu::RenderPassColorAttachment {
                    view: &occlusion_target.default_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })];
            if let Some(bent_target) = bent_target {
                attachments.push(Some(wgpu::RenderPassColorAttachment {
                    view: &bent_target.default_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        // Alpha 0 reads as "no bent normal".
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }));
            }
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Label::from(format!("{} - {name}", self.label)).wgpu_label(),
                color_attachments: &attachments,
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

    /// The finished bent normal, for the horizon method. See
    /// [`Self::BENT_NORMAL_FORMAT`] for the encoding.
    pub fn bent_normal_texture(&self) -> Option<&GpuTexture> {
        self.bent_normals.as_ref().map(|[_, blurred]| blurred)
    }
}

#[cfg(test)]
mod tests {
    use super::{specular_occlusion, specular_occlusion_cone, spherical_cap_intersection};

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
        assert_eq!(offset_of!(U, steps_per_slice), 100);
        // Uniform buffers are allocated in 256-byte steps.
        assert_eq!(size_of::<U>(), 256);
    }

    /// Both methods must build a valid blur bind group.
    ///
    /// The bind-group pool numbers entries by position, so the layout's
    /// binding numbers must run 0..n with no gap. A gap left by the
    /// horizon-only bent normal once invalidated the disk method's blur and
    /// blacked out every view.
    #[test]
    fn both_methods_build_valid_bind_groups() {
        use super::{OcclusionConfig, OcclusionMethod, OcclusionProcessor};
        use crate::RenderContext;

        for method in [
            OcclusionMethod::Disk { sample_count: 8 },
            OcclusionMethod::Horizon {
                slice_count: 2,
                steps_per_slice: 4,
            },
        ] {
            let ctx = RenderContext::new_test();
            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            let _processor = OcclusionProcessor::new(
                &ctx,
                &OcclusionConfig {
                    method,
                    world_radius: 1.0,
                    pixel_radius_range: [2.0, 64.0],
                    strength: 1.0,
                },
                &"test".into(),
                [64, 64],
                glam::Mat4::IDENTITY,
            );
            let error = pollster::block_on(scope.pop());
            assert!(error.is_none(), "{method:?}: {error:?}");
        }
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

    #[test]
    fn a_cap_inside_another_overlaps_by_its_own_area() {
        // A narrow cap centred in a wide one: the overlap is the narrow cap.
        let narrow = 0.9_f32;
        let area = spherical_cap_intersection(narrow, 0.0, 1.0);
        let expected = std::f32::consts::TAU * (1.0 - narrow);
        assert!((area - expected).abs() < 1e-4, "{area} vs {expected}");
    }

    #[test]
    fn caps_facing_apart_do_not_overlap() {
        assert_eq!(spherical_cap_intersection(0.9, 0.9, -1.0), 0.0);
    }

    #[test]
    fn an_open_sky_leaves_the_reflection_unoccluded() {
        // AO 1 opens the whole hemisphere around a bent normal that points
        // along the reflection: nothing is occluded.
        let so = specular_occlusion_cone(1.0, 1.0, 0.5);
        assert!((so - 1.0).abs() < 1e-3, "{so}");
    }

    #[test]
    fn a_closed_sky_occludes_the_reflection() {
        assert!(specular_occlusion_cone(1.0, 0.0, 0.5) < 1e-3);
    }

    #[test]
    fn a_reflection_away_from_the_opening_is_occluded() {
        // Half the sky open, but the reflection points the other way.
        assert!(specular_occlusion_cone(-1.0, 0.5, 0.5) < 1e-3);
    }
}
