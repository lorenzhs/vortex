// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! LZ4 compression for binary arrays.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_error::VortexResult;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// LZ4 compression for binary arrays.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Lz4Scheme;

impl Scheme for Lz4Scheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.binary.lz4"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_binary()
    }

    fn expected_compression_ratio(
        &self,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        CompressionEstimate::Deferred(DeferredEstimate::Sample)
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let compacted = data
            .array_as_varbinview()
            .into_owned()
            .compact_buffers(exec_ctx)?;
        Ok(vortex_lz4::Lz4::from_var_bin_view(&compacted, 8192, exec_ctx)?.into_array())
    }
}
