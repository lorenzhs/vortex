// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generate a self-contained HTML visualization of a Vortex file's byte layout.
//!
//! Reads only the file footer (no column data), walks the layout tree to attribute every
//! segment to a column and row range, and emits a single HTML file with three linked views:
//! a file-wide byte-range map, a column-by-row-range matrix, and per-column byte tracks.

use std::io::Read;
use std::io::Seek;
use std::io::SeekFrom;
use std::path::Path;
use std::path::PathBuf;

use flatbuffers::root;
use serde::Serialize;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::EOF_SIZE;
use vortex::file::MAGIC_BYTES;
use vortex::file::MAX_POSTSCRIPT_SIZE;
use vortex::file::OpenOptionsSessionExt;
use vortex::flatbuffers::footer as fb;
use vortex::session::VortexSession;

use crate::segment_tree::collect_segment_tree;

/// HTML template with a `/*__VORTEX_DATA__*/` placeholder for the embedded JSON model.
const TEMPLATE: &str = include_str!("template.html");

/// The placeholder string in [`TEMPLATE`] that is replaced with the serialized model.
const DATA_PLACEHOLDER: &str = "/*__VORTEX_DATA__*/";

/// Command-line arguments for the visualize command.
#[derive(Debug, clap::Parser)]
pub struct VisualizeArgs {
    /// Path to the Vortex file.
    pub file: PathBuf,
    /// Output HTML path. Defaults to `<file-stem>.layout.html` in the current directory.
    #[clap(short, long)]
    pub output: Option<PathBuf>,
    /// Open the generated HTML in the default browser.
    #[clap(long)]
    pub open: bool,
}

/// The full data model embedded into the HTML output.
#[derive(Serialize)]
struct VizModel {
    /// Display name of the file (file name component).
    file: String,
    /// Total size of the file in bytes.
    file_size: u64,
    /// Total number of rows.
    row_count: u64,
    /// Human-readable schema (the root dtype).
    schema: String,
    /// Canonical row-range partition boundaries (`[start, end)` per split).
    splits: Vec<[u64; 2]>,
    /// Interned segment-name table; each column's `n` array indexes into this.
    seg_names: Vec<String>,
    /// Columns in display order.
    columns: Vec<VizColumn>,
    /// Metadata byte ranges (footer/postscript regions at the end of the file).
    meta: Vec<VizMeta>,
}

/// A labeled metadata byte range (e.g. the footer, postscript, or EOF marker).
#[derive(Serialize)]
struct VizMeta {
    /// Human-readable label (e.g. `"dtype"`, `"layout"`, `"postscript"`, `"EOF"`).
    name: String,
    /// Byte offset in the file.
    byte_offset: u64,
    /// Length in bytes.
    byte_length: u32,
}

/// Per-column segment data in columnar (parallel-array) form.
///
/// Storing one array per field instead of one object per segment eliminates the
/// per-segment JSON key overhead (~6 MB on large files) while keeping the data
/// identical. All arrays have the same length; element `i` describes segment `i`.
#[derive(Serialize)]
struct VizColumn {
    /// Field name (may be dotted for nested fields).
    name: String,
    /// Index into `VizModel::seg_names` for each segment.
    n: Vec<u32>,
    /// First row covered by each segment.
    ro: Vec<u64>,
    /// Number of rows covered by each segment.
    rc: Vec<u64>,
    /// Byte offset in the file for each segment.
    o: Vec<u64>,
    /// Byte length of each segment.
    l: Vec<u32>,
}

/// Generate an HTML visualization of a Vortex file's byte layout.
///
/// # Errors
///
/// Returns an error if the file cannot be opened, read, or if the output cannot be written.
pub async fn exec_visualize(session: &VortexSession, args: VisualizeArgs) -> VortexResult<()> {
    let vxf = session.open_options().open_path(&args.file).await?;
    let footer = vxf.footer();

    let mut segment_tree = collect_segment_tree(footer.layout().as_ref(), footer.segment_map());

    // Build per-column segments in columnar form.  One parallel array per field avoids
    // repeating JSON key strings for every segment (saves ~6 MB on large files).
    // Segment names are interned into a global table to avoid repeating path strings.
    let mut name_to_idx: std::collections::HashMap<String, u32> = std::collections::HashMap::new();
    let mut seg_names: Vec<String> = Vec::new();

    let columns: Vec<VizColumn> = segment_tree
        .segment_ordering
        .iter()
        .filter_map(|col_name| {
            let mut segments = segment_tree.segments.remove(col_name)?;
            segments.sort_by(|a, b| a.spec.offset.cmp(&b.spec.offset));

            let count = segments.len();
            let mut n = Vec::with_capacity(count);
            let mut ro = Vec::with_capacity(count);
            let mut rc = Vec::with_capacity(count);
            let mut o = Vec::with_capacity(count);
            let mut l = Vec::with_capacity(count);

            for seg in segments {
                let seg_name = seg.name.to_string();
                let name_idx = if let Some(&idx) = name_to_idx.get(&seg_name) {
                    idx
                } else {
                    let idx = u32::try_from(seg_names.len()).unwrap_or(u32::MAX);
                    name_to_idx.insert(seg_name.clone(), idx);
                    seg_names.push(seg_name);
                    idx
                };
                n.push(name_idx);
                ro.push(seg.row_offset);
                rc.push(seg.row_count);
                o.push(seg.spec.offset);
                l.push(seg.spec.length);
            }

            Some(VizColumn { name: col_name.to_string(), n, ro, rc, o, l })
        })
        .collect();

    // Total file size: prefer the filesystem, fall back to the furthest segment end so that
    // non-local paths still produce a sensible visualization.
    let max_segment_end = footer
        .segment_map()
        .iter()
        .map(|s| s.offset + u64::from(s.length))
        .max()
        .unwrap_or(0);
    let file_size = std::fs::metadata(&args.file)
        .map(|m| m.len())
        .unwrap_or(max_segment_end)
        .max(max_segment_end);

    let meta = read_meta_segments(&args.file, file_size);

    let model = VizModel {
        file: args
            .file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| args.file.to_string_lossy().into_owned()),
        file_size,
        row_count: vxf.row_count(),
        schema: format!("{}", vxf.dtype()),
        splits: vxf
            .splits()?
            .into_iter()
            .map(|r| [r.start, r.end])
            .collect(),
        seg_names,
        columns,
        meta,
    };

    let json = serde_json::to_string(&model)
        .map_err(|e| vortex_err!("Failed to serialize visualization model: {e}"))?;
    let html = TEMPLATE.replace(DATA_PLACEHOLDER, &json);

    let output = args.output.unwrap_or_else(|| {
        let stem = args
            .file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "vortex".to_string());
        PathBuf::from(format!("{stem}.layout.html"))
    });

    std::fs::write(&output, html)
        .map_err(|e| vortex_err!("Failed to write {}: {e}", output.display()))?;
    println!("Wrote visualization to {}", output.display());

    if args.open {
        open_in_browser(&output);
    }

    Ok(())
}

/// Read the file tail and return labeled byte ranges for the footer/postscript metadata region.
///
/// Returns an empty vec if the file cannot be read or the postscript cannot be parsed (e.g.
/// for non-local paths already covered by the `file_size` fallback).
fn read_meta_segments(file: &Path, file_size: u64) -> Vec<VizMeta> {
    let read_size = (u64::from(MAX_POSTSCRIPT_SIZE) + EOF_SIZE as u64).min(file_size);
    let read_offset = file_size - read_size;

    let Ok(read_size_usize) = usize::try_from(read_size) else {
        return Vec::new();
    };
    let mut buf = vec![0u8; read_size_usize];
    {
        let Ok(mut f) = std::fs::File::open(file) else {
            return Vec::new();
        };
        if f.seek(SeekFrom::Start(read_offset)).is_err() {
            return Vec::new();
        }
        if f.read_exact(&mut buf).is_err() {
            return Vec::new();
        }
    }

    let eof_loc = buf.len() - EOF_SIZE;
    if buf[eof_loc + 4..eof_loc + 8] != MAGIC_BYTES {
        return Vec::new();
    }
    let ps_size_u16 = u16::from_le_bytes([buf[eof_loc + 2], buf[eof_loc + 3]]);
    let ps_size = usize::from(ps_size_u16);
    if ps_size + EOF_SIZE > buf.len() {
        return Vec::new();
    }
    let ps_bytes = &buf[eof_loc - ps_size..eof_loc];
    let Ok(ps) = root::<fb::Postscript>(ps_bytes) else {
        return Vec::new();
    };

    let ps_byte_offset = file_size - EOF_SIZE as u64 - ps_size as u64;
    let eof_byte_offset = file_size - EOF_SIZE as u64;

    let mut meta = Vec::new();
    if let Some(s) = ps.dtype() {
        meta.push(VizMeta {
            name: "dtype".to_string(),
            byte_offset: s.offset(),
            byte_length: s.length(),
        });
    }
    if let Some(s) = ps.layout() {
        meta.push(VizMeta {
            name: "layout".to_string(),
            byte_offset: s.offset(),
            byte_length: s.length(),
        });
    }
    if let Some(s) = ps.statistics() {
        meta.push(VizMeta {
            name: "statistics".to_string(),
            byte_offset: s.offset(),
            byte_length: s.length(),
        });
    }
    if let Some(s) = ps.footer() {
        meta.push(VizMeta {
            name: "footer".to_string(),
            byte_offset: s.offset(),
            byte_length: s.length(),
        });
    }
    meta.push(VizMeta {
        name: "postscript".to_string(),
        byte_offset: ps_byte_offset,
        byte_length: u32::from(ps_size_u16),
    });
    meta.push(VizMeta {
        name: "EOF".to_string(),
        byte_offset: eof_byte_offset,
        // EOF_SIZE = 8, always fits in u32
        byte_length: u32::try_from(EOF_SIZE).unwrap_or(8),
    });
    meta
}

/// Best-effort: open `path` in the system default browser. Failures are reported but ignored.
fn open_in_browser(path: &Path) {
    #[cfg(target_os = "macos")]
    let opener = "open";
    #[cfg(target_os = "windows")]
    let opener = "explorer";
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let opener = "xdg-open";

    if let Err(e) = std::process::Command::new(opener).arg(path).spawn() {
        eprintln!("Could not open browser with `{opener}`: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::DATA_PLACEHOLDER;
    use super::TEMPLATE;

    /// The HTML template must contain the data placeholder exactly once; otherwise
    /// `exec_visualize` would emit an HTML file with no embedded data (or with the
    /// JSON injected in the wrong place).
    #[test]
    fn template_has_single_placeholder() {
        assert_eq!(
            TEMPLATE.matches(DATA_PLACEHOLDER).count(),
            1,
            "template must embed `{DATA_PLACEHOLDER}` exactly once"
        );
    }
}
