// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! LZ4-backed compression encodings for variable-width Vortex arrays.
//!
//! [`Lz4Array`] stores UTF-8, binary, or primitive values as one or more LZ4 blocks ("frames").
//! Frame metadata lets slices decompress only the frames that can contribute values to the
//! requested row range.
//!
//! Unlike [`vortex-zstd`], the LZ4 encoding does not support trained dictionaries: the `lz4_flex`
//! codec is a pure-Rust block compressor with no dictionary trainer. Each frame is compressed
//! independently.
//!
//! This crate exposes array encodings only. Compression scheme selection is wired through
//! `vortex-btrblocks` and file writing. To deserialize arrays manually, register the encoding in
//! the array session:
//!
//! ```rust
//! use vortex_array::session::ArraySessionExt;
//!
//! let session = vortex_array::array_session();
//! session.arrays().register(vortex_lz4::Lz4);
//! ```
//!
//! [`vortex-zstd`]: https://docs.rs/vortex-zstd

pub use array::*;
use vortex_array::session::ArraySessionExt;
use vortex_session::VortexSession;

mod array;
mod compute;
mod rules;
mod slice;

#[cfg(test)]
mod test;

/// Initialize the LZ4 encoding in the given session.
pub fn initialize(session: &VortexSession) {
    session.arrays().register(Lz4);
}

#[derive(Clone, prost::Message)]
/// Metadata for one LZ4 frame.
pub struct Lz4FrameMetadata {
    /// Uncompressed byte size of this frame.
    #[prost(uint64, tag = "1")]
    pub uncompressed_size: u64,
    /// Number of valid values stored in this frame.
    #[prost(uint64, tag = "2")]
    pub n_values: u64,
}

#[derive(Clone, prost::Message)]
/// Serialized metadata for a [`Lz4Array`].
pub struct Lz4Metadata {
    /// Metadata for each compressed frame.
    #[prost(message, repeated, tag = "1")]
    pub frames: Vec<Lz4FrameMetadata>,
}
