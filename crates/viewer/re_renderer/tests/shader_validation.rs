//! Validates every `.wgsl` shader in the workspace at compile-test time.
//!
//! Our shaders are compiled lazily at runtime (the first time a pipeline that uses them is
//! created), so a broken shader could easily slip through code review and CI and only blow up
//! when a user happens to trigger that particular code path.
//!
//! This test resolves the `#import` directives the exact same way the renderer does at runtime
//! (via [`re_renderer::FileResolver`]) and then runs the resolved source through `naga`.

use std::{
    borrow::Cow,
    path::{Path, PathBuf},
};

use re_renderer::{FileResolver, FileSystem, SearchPath};
use wgpu::naga;

/// A [`FileSystem`] backed by `std::fs`.
///
/// We can't reuse `re_renderer`'s own `OsFileSystem` because it's only compiled in for native
/// debug builds with shader hot-reloading enabled (`cfg(load_shaders_from_disk)`), which is not
/// guaranteed to be the case when running tests in CI.
struct DiskFileSystem;

impl FileSystem for DiskFileSystem {
    fn read_to_string(&self, path: impl AsRef<Path>) -> anyhow::Result<Cow<'static, str>> {
        let path = path.as_ref();
        std::fs::read_to_string(path)
            .map(Into::into)
            .map_err(|err| anyhow::anyhow!("failed to read {path:?}: {err}"))
    }

    fn canonicalize(&self, path: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
        let path = path.as_ref();
        std::fs::canonicalize(path)
            .map_err(|err| anyhow::anyhow!("failed to canonicalize {path:?}: {err}"))
    }

    fn exists(&self, path: impl AsRef<Path>) -> bool {
        path.as_ref().exists()
    }
}

/// Recursively collects every `.wgsl` file under `root`.
fn collect_wgsl_files(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_wgsl_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "wgsl") {
            out.push(path);
        }
    }
}

#[test]
fn all_wgsl_shaders_are_valid() {
    // `CARGO_MANIFEST_DIR` points at the `re_renderer` crate directory.
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shader_dir = crate_dir.join("shader");
    // …/rerun (the workspace root that holds both `crates/` and `examples/`).
    let workspace_root = crate_dir
        .ancestors()
        .nth(3)
        .expect("re_renderer should live at crates/viewer/re_renderer");
    let examples_dir = workspace_root.join("examples");

    // Imports such as `#import <types.wgsl>` resolve against the search path; relative imports
    // (`#import <./utils/foo.wgsl>`) resolve against the importing file. The renderer's shaders
    // and the example shaders all import from `re_renderer/shader`, so that's the search root.
    let mut search_path = SearchPath::default();
    search_path.push(&shader_dir);
    let resolver = FileResolver::with_search_path(DiskFileSystem, search_path);

    let mut shaders = Vec::new();
    collect_wgsl_files(&shader_dir, &mut shaders);
    collect_wgsl_files(&examples_dir, &mut shaders);
    shaders.sort();
    assert!(
        !shaders.is_empty(),
        "found no .wgsl files to validate under {shader_dir:?} or {examples_dir:?}"
    );

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );

    let mut failures = Vec::new();
    let mut checked = 0;
    let mut skipped = 0;

    for shader in &shaders {
        let rel = shader.strip_prefix(workspace_root).unwrap_or(shader);

        // Resolve all `#import`s, producing a self-contained module.
        let interpolated = match resolver.populate(shader) {
            Ok(interpolated) => interpolated,
            Err(err) => {
                failures.push(format!(
                    "{}\n  failed to resolve imports: {err}",
                    rel.display()
                ));
                continue;
            }
        };

        // FORK ADDITION: a fragment that carries the entry points but expects its
        // *includer* to supply bindings. `instanced_mesh_common.wgsl` holds the
        // shared body of the mesh shader; `instanced_mesh.wgsl` (storage buffer)
        // and `instanced_mesh_limited.wgsl` (WebGL2 uniform buffer) each declare
        // `selected_ids` and `fn is_selected` before importing it. It is validated
        // transitively through both of those, so skipping it here loses no coverage.
        if rel.ends_with("shader/instanced_mesh_common.wgsl") {
            skipped += 1;
            continue;
        }

        // Files without an entry point are include-only fragments (type/util libraries). They
        // can't be validated standalone — they get validated transitively wherever they're
        // imported into an entry-point shader.
        let has_entry_point = ["@vertex", "@fragment", "@compute"]
            .iter()
            .any(|kw| interpolated.contents.contains(kw));
        if !has_entry_point {
            skipped += 1;
            continue;
        }

        let source = &interpolated.contents;

        let module = match naga::front::wgsl::parse_str(source) {
            Ok(module) => module,
            Err(err) => {
                failures.push(format!(
                    "{}\n{}",
                    rel.display(),
                    indent(&err.emit_to_string(source))
                ));
                continue;
            }
        };

        if let Err(err) = validator.validate(&module) {
            failures.push(format!(
                "{}\n{}",
                rel.display(),
                indent(&err.emit_to_string(source))
            ));
            continue;
        }

        checked += 1;
    }

    eprintln!(
        "Validated {checked} entry-point shader(s), skipped {skipped} include-only fragment(s)."
    );

    assert!(
        failures.is_empty(),
        "{} shader(s) failed validation:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

fn indent(s: &str) -> String {
    s.lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// The material uniform's WGSL layout must match `gpu_data::MaterialUniformBuffer`.
///
/// This pins a bug class that is silent by construction. The Rust struct spells
/// its scalars `wgpu_buffer_types::U32RowPadded` -- a `u32` plus three words of
/// padding, so each occupies a full 16-byte row -- while WGSL gives a bare
/// `u32` an alignment of 4. Writing the WGSL fields back to back therefore put
/// `use_matcap` at offset 20 where Rust wrote it at 32, so the shader read
/// `texture_format`'s first padding word. That word is always zero, so
/// `material.use_matcap != 0u` was always false and matcap shading could never
/// appear -- with no validation error, no warning, and nothing to see on the
/// CPU side, where the flag really was `true`.
///
/// `texture_format` masked the problem: offset 16 is correct for both spellings,
/// so textured meshes rendered fine and only the field AFTER it was displaced.
///
/// Read through `instanced_mesh.wgsl` rather than `instanced_mesh_common.wgsl`:
/// the latter is an include-only fragment that expects its includer to supply
/// bindings, so it does not parse standalone.
#[test]
fn material_uniform_layout_matches_the_rust_struct() {
    // Offsets of `gpu_data::MaterialUniformBuffer` in `mesh.rs`. That struct is
    // `pub(crate)`, so an integration test cannot `offset_of!` it directly --
    // the Rust half of this pair is asserted by
    // `mesh::tests::material_uniform_offsets_are_row_padded`. Both halves must
    // name the same numbers, so drift on either side fails one of them.
    const EXPECTED: &[(&str, u32)] = &[
        ("albedo_factor", 0),
        ("texture_format", 16),
        ("use_matcap", 32),
        ("specular_roughness", 48),
    ];

    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let shader_dir = crate_dir.join("shader");

    let mut search_path = SearchPath::default();
    search_path.push(&shader_dir);
    let resolver = FileResolver::with_search_path(DiskFileSystem, search_path);

    let shader = shader_dir.join("instanced_mesh.wgsl");
    let interpolated = resolver
        .populate(&shader)
        .expect("instanced_mesh.wgsl should resolve its imports");
    let source = &interpolated.contents;
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|err| panic!("{}", err.emit_to_string(source)));

    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("MaterialUniformBuffer"))
        .expect("instanced_mesh.wgsl should declare MaterialUniformBuffer");

    let naga::TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("MaterialUniformBuffer should be a struct");
    };

    for (name, expected_offset) in EXPECTED {
        let member = members
            .iter()
            .find(|m| m.name.as_deref() == Some(*name))
            .unwrap_or_else(|| panic!("MaterialUniformBuffer should have a `{name}` member"));
        assert_eq!(
            member.offset, *expected_offset,
            "`{name}` sits at WGSL offset {} but Rust writes it at {expected_offset}; \
             the uniform's padding has drifted out of sync with \
             `gpu_data::MaterialUniformBuffer`",
            member.offset,
        );
    }
}

/// The occlusion uniform's WGSL layout must match `OcclusionUniformBuffer`
/// (akatela SPEC-123).
///
/// The Rust half is `draw_phases::occlusion::tests::uniform_offsets_match_the_shader`.
/// Both halves carry the same numbers, so neither side can drift alone.
#[test]
fn occlusion_uniform_layout_matches_the_rust_struct() {
    const EXPECTED: &[(&str, u32)] = &[
        ("view_from_projection", 0),
        ("framebuffer_resolution", 64),
        ("pixels_per_world_unit", 72),
        ("perspective", 76),
        ("world_radius", 80),
        ("pixel_radius_min", 84),
        ("pixel_radius_max", 88),
        ("strength", 92),
        ("sample_count", 96),
        ("steps_per_slice", 100),
    ];
    let source = include_str!("../shader/occlusion/common.wgsl");
    let module = naga::front::wgsl::parse_str(source)
        .unwrap_or_else(|err| panic!("occlusion/common.wgsl should parse: {err}"));
    let (_, ty) = module
        .types
        .iter()
        .find(|(_, ty)| ty.name.as_deref() == Some("OcclusionUniformBuffer"))
        .expect("occlusion/common.wgsl should declare OcclusionUniformBuffer");
    let naga::TypeInner::Struct { members, .. } = &ty.inner else {
        panic!("OcclusionUniformBuffer should be a struct");
    };
    for (name, expected_offset) in EXPECTED {
        let member = members
            .iter()
            .find(|member| member.name.as_deref() == Some(*name))
            .unwrap_or_else(|| panic!("OcclusionUniformBuffer should have `{name}`"));
        assert_eq!(
            member.offset, *expected_offset,
            "`{name}` sits at WGSL offset {} but Rust writes it at {expected_offset}",
            member.offset,
        );
    }
}
