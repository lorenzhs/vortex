// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;

use itertools::Itertools as _;
use lz4_flex::block::compress_into;
use lz4_flex::block::decompress_into;
use lz4_flex::block::get_maximum_output_size;
use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::EqMode;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::varbinview::build_views::BinaryView;
use vortex_array::arrays::varbinview::build_views::MAX_BUFFER_LEN;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::smallvec::smallvec;
use vortex_array::validity::Validity;
use vortex_array::vtable::OperationsVTable;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::child_to_validity;
use vortex_array::vtable::validity_to_child;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Lz4FrameMetadata;
use crate::Lz4Metadata;

type ViewLen = u32;

// Overall approach here:
// LZ4 can be used on the whole array (values_per_frame = 0), resulting in a single LZ4
// block, or it can be split into multiple independent frames (values_per_frame < # values).
// This latter case is helpful if you want somewhat faster access to slices or individual
// rows, allowing us to only decompress the necessary frames. Unlike zstd, LZ4 has no
// dictionary trainer, so frames never share a dictionary.

// Visually, during decompression, we have an interval of frames we're
// decompressing and a tighter interval of the slice we actually care about.
// |=============values (all valid elements)==============|
// |<-skipped_uncompressed->|----decompressed-------------|
//                              |------slice-------|
//                              ^                  ^
// |<-slice_uncompressed_start->|                  |
// |<------------slice_uncompressed_stop---------->|
// We then insert these values to the correct position using a primitive array
// constructor.

/// A [`Lz4`]-encoded Vortex array.
pub type Lz4Array = Array<Lz4>;

impl ArrayHash for Lz4Data {
    fn array_hash<H: Hasher>(&self, state: &mut H, accuracy: EqMode) {
        for frame in &self.frames {
            frame.array_hash(state, accuracy);
        }
        self.unsliced_n_rows.hash(state);
        self.slice_start.hash(state);
        self.slice_stop.hash(state);
    }
}

impl ArrayEq for Lz4Data {
    fn array_eq(&self, other: &Self, accuracy: EqMode) -> bool {
        if self.frames.len() != other.frames.len() {
            return false;
        }
        for (a, b) in self.frames.iter().zip(&other.frames) {
            if !a.array_eq(b, accuracy) {
                return false;
            }
        }
        self.unsliced_n_rows == other.unsliced_n_rows
            && self.slice_start == other.slice_start
            && self.slice_stop == other.slice_stop
    }
}

impl VTable for Lz4 {
    type TypedArrayData = Lz4Data;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.lz4");
        *ID
    }

    fn validate(
        &self,
        data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let validity = child_to_validity(slots[0].as_ref(), dtype.nullability());
        data.validate(dtype, len, &validity)
    }

    fn nbuffers(array: ArrayView<'_, Self>) -> usize {
        array.frames.len()
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        BufferHandle::new_host(array.frames[idx].clone())
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        Some(format!("frame_{idx}"))
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        let mut data = array.data().clone();
        data.frames = buffers
            .iter()
            .map(|buffer| buffer.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;
        Ok(
            ArrayParts::new(self.clone(), array.dtype().clone(), array.len(), data)
                .with_slots(array.slots().iter().cloned().collect()),
        )
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(array.metadata.clone().encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = Lz4Metadata::decode(metadata)?;
        let validity = if children.is_empty() {
            Validity::from(dtype.nullability())
        } else if children.len() == 1 {
            let validity = children.get(0, &Validity::DTYPE, len)?;
            Validity::Array(validity)
        } else {
            vortex_bail!("Lz4Array expected 0 or 1 child, got {}", children.len());
        };

        let frames = buffers
            .iter()
            .map(|b| b.clone().try_to_host_sync())
            .collect::<VortexResult<Vec<_>>>()?;

        let slots = smallvec![validity_to_child(&validity, len)];
        let data = Lz4Data::new(frames, metadata, len);
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let unsliced_validity = child_to_validity(
            array.as_ref().slots()[0].as_ref(),
            array.dtype().nullability(),
        );
        array
            .data()
            .decompress(array.dtype(), &unsliced_validity, ctx)?
            .execute::<ArrayRef>(ctx)
            .map(ExecutionResult::done)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        crate::rules::RULES.evaluate(array, parent, child_idx)
    }
}

#[derive(Clone, Debug)]
/// LZ4 array encoding marker.
pub struct Lz4;

impl Lz4 {
    /// Construct a [`Lz4Array`] from validated compressed data and validity.
    pub fn try_new(dtype: DType, data: Lz4Data, validity: Validity) -> VortexResult<Lz4Array> {
        let len = data.len();
        data.validate(&dtype, len, &validity)?;
        let slots = smallvec![validity_to_child(&validity, data.unsliced_n_rows())];
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(Lz4, dtype, len, data).with_slots(slots))
        })
    }

    /// Compress a [`VarBinViewArray`] using LZ4.
    pub fn from_var_bin_view(
        vbv: &VarBinViewArray,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Lz4Array> {
        let validity = vbv.validity()?;
        Self::try_new(
            vbv.dtype().clone(),
            Lz4Data::from_var_bin_view(vbv, values_per_frame, ctx)?,
            validity,
        )
    }

    /// Compress a [`PrimitiveArray`] using LZ4.
    pub fn from_primitive(
        parray: &PrimitiveArray,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Lz4Array> {
        let validity = parray.validity()?;
        Self::try_new(
            parray.dtype().clone(),
            Lz4Data::from_primitive(parray, values_per_frame, ctx)?,
            validity,
        )
    }

    /// Decompress a [`Lz4Array`] into its canonical Vortex representation.
    pub fn decompress(array: &Lz4Array, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        let unsliced_validity = child_to_validity(
            array.as_ref().slots()[0].as_ref(),
            array.dtype().nullability(),
        );
        array
            .data()
            .decompress(array.dtype(), &unsliced_validity, ctx)
    }

    /// Decompress a `Utf8`/`Binary` [`Lz4Array`]'s values straight into a caller-owned buffer,
    /// bypassing canonicalization. See [`Lz4Data::decompress_var_bin_into`].
    pub fn decompress_var_bin_into(
        array: &Lz4Array,
        ctx: &mut ExecutionCtx,
        out: &mut Vec<u8>,
    ) -> VortexResult<VarBinDecompressed> {
        let unsliced_validity = child_to_validity(
            array.as_ref().slots()[0].as_ref(),
            array.dtype().nullability(),
        );
        array
            .data()
            .decompress_var_bin_into(&unsliced_validity, ctx, out)
    }
}

/// The validity bitmap indicating which elements are non-null.
pub(super) const NUM_SLOTS: usize = 1;
pub(super) const SLOT_NAMES: [&str; NUM_SLOTS] = ["validity"];

#[derive(Clone, Debug)]
/// Encoding-specific data for a [`Lz4Array`].
pub struct Lz4Data {
    pub(crate) frames: Vec<ByteBuffer>,
    pub(crate) metadata: Lz4Metadata,
    unsliced_n_rows: usize,
    slice_start: usize,
    slice_stop: usize,
}

impl Display for Lz4Data {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "nrows: {}, slice: {}..{}",
            self.unsliced_n_rows, self.slice_start, self.slice_stop
        )
    }
}

/// Movable parts of a [`Lz4Data`] value plus its validity.
pub struct Lz4DataParts {
    /// Compressed LZ4 frames.
    pub frames: Vec<ByteBuffer>,
    /// Serialized frame metadata.
    pub metadata: Lz4Metadata,
    /// Unsliced validity for the array.
    pub validity: Validity,
    /// Unsliced row count.
    pub n_rows: usize,
    /// Start of this logical slice in unsliced row coordinates.
    pub slice_start: usize,
    /// End of this logical slice in unsliced row coordinates.
    pub slice_stop: usize,
}

/// Compressed LZ4 frames and their metadata.
#[derive(Debug)]
struct Frames {
    frames: Vec<ByteBuffer>,
    frame_metas: Vec<Lz4FrameMetadata>,
}

/// The frames overlapping a slice, selected by [`Lz4Data::plan_frames`] without decompressing.
struct FramePlan<'a> {
    /// `(compressed frame, its uncompressed size)` for each frame overlapping the slice, in order.
    frames: Vec<(&'a ByteBuffer, usize)>,
    /// Sum of the selected frames' uncompressed sizes.
    total_uncompressed: usize,
    /// Values in frames that precede the first selected frame (skipped entirely).
    n_skipped_values: usize,
    /// Value index (non-null value count) at the start of the slice.
    slice_value_idx_start: usize,
    /// Value index (non-null value count) at the end of the slice.
    slice_value_idx_stop: usize,
}

/// Value-index window produced by [`Lz4Data::decompress_var_bin_into`]: within the decompressed
/// `[ViewLen][bytes]` stream written to the caller's buffer, the slice's values are the `n_values`
/// values that follow the first `skip_values` values.
pub struct VarBinDecompressed {
    /// Leading values in the buffer (from the head frame) that precede the slice.
    pub skip_values: usize,
    /// Values belonging to the slice.
    pub n_values: usize,
}

fn collect_valid_primitive(
    parray: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<PrimitiveArray> {
    let mask = parray
        .as_ref()
        .validity()?
        .execute_mask(parray.as_ref().len(), ctx)?;
    let result = parray.filter(mask)?.execute::<PrimitiveArray>(ctx)?;
    Ok(result)
}

fn collect_valid_vbv(
    vbv: &VarBinViewArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ByteBuffer, Vec<usize>)> {
    let mask = vbv
        .as_ref()
        .validity()?
        .execute_mask(vbv.as_ref().len(), ctx)?;
    let buffer_and_value_byte_indices = match mask.bit_buffer() {
        AllOr::None => (Buffer::empty(), Vec::new()),
        _ => {
            let mut buffer = BufferMut::with_capacity(
                usize::try_from(vbv.nbytes()).vortex_expect("must fit into buffer")
                    + mask.true_count() * size_of::<ViewLen>(),
            );
            let mut value_byte_indices = Vec::new();
            let views = vbv.views();
            let buffers = vbv
                .data_buffers()
                .iter()
                .map(|b| b.as_host())
                .collect::<Vec<_>>();
            // skip nulls, writing only valid values
            for (i, view) in views.iter().enumerate() {
                if !mask.value(i) {
                    continue;
                }
                let value = if view.is_inlined() {
                    view.as_inlined().value()
                } else {
                    let view_ref = view.as_view();
                    &buffers[view_ref.buffer_index as usize][view_ref.as_range()]
                };
                value_byte_indices.push(buffer.len());
                // here's where we write the string lengths
                buffer.extend_trusted(ViewLen::try_from(value.len())?.to_le_bytes().into_iter());
                buffer.extend_from_slice(value);
            }
            (buffer.freeze(), value_byte_indices)
        }
    };
    Ok(buffer_and_value_byte_indices)
}

/// Reconstruct BinaryView structs from length-prefixed byte data.
///
/// The buffer contains interleaved u32 lengths (little-endian) and string data.
/// When the cumulative data exceeds `max_buffer_len`, the buffer is split (zero-copy) into
/// multiple segments so that BinaryView's u32 offsets can address all data.
///
/// Pass [`MAX_BUFFER_LEN`] for `max_buffer_len` in production; a smaller value can be used in
/// tests to exercise the splitting path without allocating >2 GiB.
pub fn reconstruct_views(
    buffer: &ByteBuffer,
    max_buffer_len: usize,
) -> (Vec<ByteBuffer>, Buffer<BinaryView>) {
    let mut views = BufferMut::<BinaryView>::empty();
    let mut buffers = Vec::new();
    let mut segment_start: usize = 0;
    let mut offset = 0;

    while offset < buffer.len() {
        let str_len = ViewLen::from_le_bytes(
            buffer
                .get(offset..offset + size_of::<ViewLen>())
                .vortex_expect("corrupted lz4 length")
                .try_into()
                .ok()
                .vortex_expect("must fit ViewLen size"),
        ) as usize;

        let value_data_offset = offset + size_of::<ViewLen>();
        let local_offset = value_data_offset - segment_start;

        if local_offset + str_len > max_buffer_len && offset > segment_start {
            buffers.push(buffer.slice(segment_start..offset));
            segment_start = offset;
        }

        let local_offset = u32::try_from(value_data_offset - segment_start)
            .vortex_expect("local offset within segment must fit in u32");
        let buf_index = u32::try_from(buffers.len()).vortex_expect("buffer index must fit in u32");
        let value = &buffer[value_data_offset..value_data_offset + str_len];
        views.push(BinaryView::make_view(value, buf_index, local_offset));
        offset = value_data_offset + str_len;
    }

    if segment_start < buffer.len() {
        buffers.push(buffer.slice(segment_start..buffer.len()));
    }

    (buffers, views.freeze())
}

impl Lz4Data {
    /// Construct unsliced LZ4 data from raw frames and metadata.
    pub fn new(frames: Vec<ByteBuffer>, metadata: Lz4Metadata, n_rows: usize) -> Self {
        Self {
            frames,
            metadata,
            unsliced_n_rows: n_rows,
            slice_start: 0,
            slice_stop: n_rows,
        }
    }

    /// Validate dtype, slice, validity, and frame invariants.
    pub fn validate(&self, dtype: &DType, len: usize, validity: &Validity) -> VortexResult<()> {
        vortex_ensure!(
            matches!(
                dtype,
                DType::Primitive(..) | DType::Binary(_) | DType::Utf8(_)
            ),
            "Unsupported dtype for Lz4 array: {dtype}"
        );
        vortex_ensure!(
            self.slice_start <= self.slice_stop,
            "Invalid slice range {}..{}",
            self.slice_start,
            self.slice_stop
        );
        vortex_ensure!(
            self.slice_stop <= self.unsliced_n_rows,
            "Slice stop {} exceeds unsliced row count {}",
            self.slice_stop,
            self.unsliced_n_rows
        );
        vortex_ensure!(
            self.slice_stop - self.slice_start == len,
            "Slice length {} does not match array length {}",
            self.slice_stop - self.slice_start,
            len
        );
        if let Some(validity_len) = validity.maybe_len() {
            vortex_ensure!(
                validity_len == self.unsliced_n_rows,
                "Validity length {} does not match unsliced row count {}",
                validity_len,
                self.unsliced_n_rows
            );
        }

        vortex_ensure!(
            self.frames.len() == self.metadata.frames.len(),
            "Frame count {} does not match metadata frame count {}",
            self.frames.len(),
            self.metadata.frames.len()
        );

        Ok(())
    }

    pub(crate) fn with_slice(&self, start: usize, stop: usize) -> Self {
        let new_start = self.slice_start + start;
        let new_stop = self.slice_start + stop;

        assert!(
            new_start <= self.slice_stop,
            "new slice start {new_start} exceeds end {}",
            self.slice_stop
        );

        assert!(
            new_stop <= self.slice_stop,
            "new slice stop {new_stop} exceeds end {}",
            self.slice_stop
        );

        Self {
            slice_start: new_start,
            slice_stop: new_stop,
            ..self.clone()
        }
    }

    fn compress_values(
        value_bytes: &ByteBuffer,
        frame_byte_starts: &[usize],
        values_per_frame: usize,
        n_values: usize,
    ) -> VortexResult<Frames> {
        let n_frames = frame_byte_starts.len();

        let mut frame_metas = Vec::with_capacity(n_frames);
        let mut frames = Vec::with_capacity(n_frames);
        for i in 0..n_frames {
            let frame_byte_end = frame_byte_starts
                .get(i + 1)
                .copied()
                .unwrap_or(value_bytes.len());

            let uncompressed = value_bytes.slice(frame_byte_starts[i]..frame_byte_end);
            let mut compressed = vec![0u8; get_maximum_output_size(uncompressed.len())];
            let compressed_len = compress_into(uncompressed.as_slice(), &mut compressed)
                .map_err(|err| vortex_err!("while compressing with lz4: {err}"))?;
            compressed.truncate(compressed_len);

            frame_metas.push(Lz4FrameMetadata {
                uncompressed_size: uncompressed.len() as u64,
                n_values: values_per_frame.min(n_values - i * values_per_frame) as u64,
            });
            frames.push(ByteBuffer::from(compressed));
        }

        Ok(Frames {
            frames,
            frame_metas,
        })
    }

    /// Creates a Lz4Array from a primitive array.
    ///
    /// # Arguments
    /// * `parray` - The primitive array to compress
    /// * `values_per_frame` - Number of values per frame (0 = single frame)
    pub fn from_primitive(
        parray: &PrimitiveArray,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let byte_width = parray.ptype().byte_width();

        // We compress only the valid elements.
        let values = collect_valid_primitive(parray, ctx)?;
        let n_values = values.len();
        let values_per_frame = if values_per_frame > 0 {
            values_per_frame
        } else {
            n_values
        };

        let value_bytes = values.buffer_handle().try_to_host_sync()?;
        // Align frames to buffer alignment. This is necessary for overaligned buffers.
        let alignment = *value_bytes.alignment();
        let step_width = (values_per_frame * byte_width).div_ceil(alignment) * alignment;

        let frame_byte_starts = (0..n_values * byte_width)
            .step_by(step_width)
            .collect::<Vec<_>>();
        let Frames {
            frames,
            frame_metas,
        } = Self::compress_values(&value_bytes, &frame_byte_starts, values_per_frame, n_values)?;

        let metadata = Lz4Metadata {
            frames: frame_metas,
        };

        Ok(Lz4Data::new(frames, metadata, parray.len()))
    }

    /// Creates a Lz4Array from a VarBinView array.
    ///
    /// # Arguments
    /// * `vbv` - The VarBinView array to compress
    /// * `values_per_frame` - Number of values per frame (0 = single frame)
    pub fn from_var_bin_view(
        vbv: &VarBinViewArray,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        // Approach for strings: we prefix each string with its length as a u32.
        // This is the same as what Parquet does. In some cases it may be better
        // to separate the binary data and lengths as two separate streams, but
        // this approach is simpler and can be best in cases when there is
        // mutual information between strings and their lengths.
        // We compress only the valid elements.
        let (value_bytes, value_byte_indices) = collect_valid_vbv(vbv, ctx)?;
        let n_values = value_byte_indices.len();
        let values_per_frame = if values_per_frame > 0 {
            values_per_frame
        } else {
            n_values
        };

        let frame_byte_starts = (0..n_values)
            .step_by(values_per_frame)
            .map(|i| value_byte_indices[i])
            .collect::<Vec<_>>();
        let Frames {
            frames,
            frame_metas,
        } = Self::compress_values(&value_bytes, &frame_byte_starts, values_per_frame, n_values)?;

        let metadata = Lz4Metadata {
            frames: frame_metas,
        };
        Ok(Lz4Data::new(frames, metadata, vbv.len()))
    }

    /// Compress a supported canonical array into LZ4 data.
    ///
    /// Returns `Ok(None)` for canonical variants that this encoding does not support.
    pub fn from_canonical(
        canonical: &Canonical,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Self>> {
        match canonical {
            Canonical::Primitive(parray) => Ok(Some(Lz4Data::from_primitive(
                parray,
                values_per_frame,
                ctx,
            )?)),
            Canonical::VarBinView(vbv) => Ok(Some(Lz4Data::from_var_bin_view(
                vbv,
                values_per_frame,
                ctx,
            )?)),
            _ => Ok(None),
        }
    }

    /// Canonicalize and compress an array into LZ4 data.
    ///
    /// # Errors
    ///
    /// Returns an error if the array's canonical form is unsupported or compression fails.
    pub fn from_array(
        array: ArrayRef,
        values_per_frame: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Self> {
        let canonical = array.execute::<Canonical>(ctx)?;
        Self::from_canonical(&canonical, values_per_frame, ctx)?
            .ok_or_else(|| vortex_err!("Lz4 can only encode Primitive and VarBinView arrays"))
    }

    fn byte_width(dtype: &DType) -> usize {
        if dtype.is_primitive() {
            dtype.as_ptype().byte_width()
        } else {
            1
        }
    }

    fn decompress(
        &self,
        dtype: &DType,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        // To start, we figure out which frames we need to decompress, and with
        // what row offset into the first such frame.
        let byte_width = Self::byte_width(dtype);
        let slice_n_rows = self.slice_stop - self.slice_start;
        let plan = self.plan_frames(byte_width, unsliced_validity, ctx)?;
        let slice_value_idx_start = plan.slice_value_idx_start;
        let slice_value_idx_stop = plan.slice_value_idx_stop;
        let n_skipped_values = plan.n_skipped_values;
        let uncompressed_size_to_decompress = plan.total_uncompressed;
        let frames_to_decompress = plan.frames;

        // then we actually decompress those frames
        let mut decompressed = ByteBufferMut::with_capacity_aligned(
            uncompressed_size_to_decompress,
            Alignment::new(byte_width),
        );
        unsafe {
            // safety: we immediately fill all bytes in the following loop,
            // assuming our metadata's uncompressed size is correct
            decompressed.set_len(uncompressed_size_to_decompress);
        }
        let mut uncompressed_start = 0;
        for (frame, frame_uncompressed_size) in frames_to_decompress {
            let uncompressed_end = uncompressed_start + frame_uncompressed_size;
            let written = decompress_into(
                frame.as_slice(),
                &mut decompressed[uncompressed_start..uncompressed_end],
            )
            .map_err(|err| vortex_err!("while decompressing with lz4: {err}"))?;
            if written != frame_uncompressed_size {
                vortex_panic!(
                    "Lz4 metadata or frames were corrupt; expected {} bytes but decompressed {}",
                    frame_uncompressed_size,
                    written
                );
            }
            uncompressed_start = uncompressed_end;
        }
        if uncompressed_start != uncompressed_size_to_decompress {
            vortex_panic!(
                "Lz4 metadata or frames were corrupt; expected {} bytes but decompressed {}",
                uncompressed_size_to_decompress,
                uncompressed_start
            );
        }

        let decompressed = decompressed.freeze();
        // Last, we slice the exact values requested out of the decompressed data.
        let mut slice_validity = unsliced_validity.slice(self.slice_start..self.slice_stop)?;

        // NOTE: this block handles setting the output type when the validity and DType disagree.
        //
        // LZ4 is a compact block compressor, meaning that null values are not stored inline in
        // the data frames. A LZ4 Array that was initialized must always hold onto its full
        // validity bitmap, even if sliced to only include non-null values.
        //
        // We ensure that the validity of the decompressed array ALWAYS matches the validity
        // implied by the DType.
        if !dtype.is_nullable() && !matches!(slice_validity, Validity::NonNullable) {
            assert!(
                matches!(slice_validity, Validity::AllValid),
                "LZ4 array expects to be non-nullable but there are nulls after decompression"
            );

            slice_validity = Validity::NonNullable;
        } else if dtype.is_nullable() && matches!(slice_validity, Validity::NonNullable) {
            slice_validity = Validity::AllValid;
        }
        // END OF IMPORTANT BLOCK
        //

        match dtype {
            DType::Primitive(..) => {
                let slice_values_buffer = decompressed.slice(
                    (slice_value_idx_start - n_skipped_values) * byte_width
                        ..(slice_value_idx_stop - n_skipped_values) * byte_width,
                );
                let primitive = PrimitiveArray::from_values_byte_buffer(
                    slice_values_buffer,
                    dtype.as_ptype(),
                    slice_validity,
                    slice_n_rows,
                    ctx,
                );

                Ok(primitive.into_array())
            }
            DType::Binary(_) | DType::Utf8(_) => {
                match slice_validity.execute_mask(slice_n_rows, ctx)?.indices() {
                    AllOr::All => {
                        let (buffers, all_views) = reconstruct_views(&decompressed, MAX_BUFFER_LEN);
                        let valid_views = all_views.slice(
                            slice_value_idx_start - n_skipped_values
                                ..slice_value_idx_stop - n_skipped_values,
                        );

                        // SAFETY: we properly construct the views inside `reconstruct_views`
                        Ok(unsafe {
                            VarBinViewArray::new_unchecked(
                                valid_views,
                                Arc::from(buffers),
                                dtype.clone(),
                                slice_validity,
                            )
                        }
                        .into_array())
                    }
                    AllOr::None => Ok(ConstantArray::new(
                        Scalar::null(dtype.clone()),
                        slice_n_rows,
                    )
                    .into_array()),
                    AllOr::Some(valid_indices) => {
                        let (buffers, all_views) = reconstruct_views(&decompressed, MAX_BUFFER_LEN);
                        let valid_views = all_views.slice(
                            slice_value_idx_start - n_skipped_values
                                ..slice_value_idx_stop - n_skipped_values,
                        );

                        let mut views = BufferMut::<BinaryView>::zeroed(slice_n_rows);
                        for (view, index) in valid_views.into_iter().zip_eq(valid_indices) {
                            views[*index] = view
                        }

                        // SAFETY: we properly construct the views inside `reconstruct_views`
                        Ok(unsafe {
                            VarBinViewArray::new_unchecked(
                                views.freeze(),
                                Arc::from(buffers),
                                dtype.clone(),
                                slice_validity,
                            )
                        }
                        .into_array())
                    }
                }
            }
            _ => vortex_panic!("Unsupported dtype for Lz4 array: {}", dtype),
        }
    }

    /// Select the frames overlapping this array's slice, without decompressing. Shared by
    /// [`Lz4Data::decompress`] (Arrow path) and [`Lz4Data::decompress_var_bin_into`] (native path).
    fn plan_frames(
        &self,
        byte_width: usize,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<FramePlan<'_>> {
        let slice_value_indices = unsliced_validity
            .execute_mask(self.unsliced_n_rows, ctx)?
            .valid_counts_for_indices(&[self.slice_start, self.slice_stop]);
        let slice_value_idx_start = slice_value_indices[0];
        let slice_value_idx_stop = slice_value_indices[1];

        let mut frames = vec![];
        let mut value_idx_start = 0;
        let mut total_uncompressed = 0;
        let mut n_skipped_values = 0;
        for (frame, frame_meta) in self.frames.iter().zip(&self.metadata.frames) {
            if value_idx_start >= slice_value_idx_stop {
                break;
            }

            let frame_uncompressed_size = usize::try_from(frame_meta.uncompressed_size)
                .vortex_expect("Uncompressed size must fit in usize");
            let frame_n_values = if frame_meta.n_values == 0 {
                frame_uncompressed_size / byte_width
            } else {
                usize::try_from(frame_meta.n_values).vortex_expect("frame size must fit usize")
            };

            let value_idx_stop = value_idx_start + frame_n_values;
            if value_idx_stop > slice_value_idx_start {
                // we need this frame
                frames.push((frame, frame_uncompressed_size));
                total_uncompressed += frame_uncompressed_size;
            } else {
                n_skipped_values += frame_n_values;
            }
            value_idx_start = value_idx_stop;
        }

        Ok(FramePlan {
            frames,
            total_uncompressed,
            n_skipped_values,
            slice_value_idx_start,
            slice_value_idx_stop,
        })
    }

    /// Decompress the frames overlapping this array's slice straight into `out` (a caller-owned
    /// buffer reused across chunks), producing the raw `[ViewLen u32 LE][value bytes]` stream that
    /// [`reconstruct_views`] reads. Only valid for `Utf8`/`Binary` arrays (byte width 1).
    ///
    /// Unlike [`Lz4Data::decompress`], this hands back no aliasing `VarBinView`, so `out` is free to
    /// be overwritten on the next chunk. Null values are not stored inline, so `out` holds only the
    /// non-null values in order; the returned [`VarBinDecompressed`] gives the value-index window
    /// `[skip_values, skip_values + n_values)` within `out` that belongs to this slice, letting the
    /// caller skip head-frame values that precede the slice and stop after the slice's values.
    pub fn decompress_var_bin_into(
        &self,
        unsliced_validity: &Validity,
        ctx: &mut ExecutionCtx,
        out: &mut Vec<u8>,
    ) -> VortexResult<VarBinDecompressed> {
        let plan = self.plan_frames(1, unsliced_validity, ctx)?;
        out.clear();
        out.reserve(plan.total_uncompressed);
        // SAFETY: the fill loop below writes exactly `total_uncompressed` bytes (checked), mirroring
        // the `set_len` + fill pattern in `decompress`.
        unsafe {
            out.set_len(plan.total_uncompressed);
        }
        let mut uncompressed_start = 0;
        for (frame, frame_uncompressed_size) in &plan.frames {
            let uncompressed_end = uncompressed_start + frame_uncompressed_size;
            let written = decompress_into(
                frame.as_slice(),
                &mut out[uncompressed_start..uncompressed_end],
            )
            .map_err(|err| vortex_err!("while decompressing with lz4: {err}"))?;
            if written != *frame_uncompressed_size {
                vortex_panic!(
                    "Lz4 metadata or frames were corrupt; expected {} bytes but decompressed {}",
                    frame_uncompressed_size,
                    written
                );
            }
            uncompressed_start = uncompressed_end;
        }
        if uncompressed_start != plan.total_uncompressed {
            vortex_panic!(
                "Lz4 metadata or frames were corrupt; expected {} bytes but decompressed {}",
                plan.total_uncompressed,
                uncompressed_start
            );
        }
        Ok(VarBinDecompressed {
            skip_values: plan.slice_value_idx_start - plan.n_skipped_values,
            n_values: plan.slice_value_idx_stop - plan.slice_value_idx_start,
        })
    }

    /// Returns the length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.slice_stop - self.slice_start
    }

    /// Returns whether the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.slice_stop == self.slice_start
    }

    /// Split this data into movable parts, attaching the supplied validity.
    pub fn into_parts(self, validity: Validity) -> Lz4DataParts {
        Lz4DataParts {
            frames: self.frames,
            metadata: self.metadata,
            validity,
            n_rows: self.unsliced_n_rows,
            slice_start: self.slice_start,
            slice_stop: self.slice_stop,
        }
    }

    pub(crate) fn slice_start(&self) -> usize {
        self.slice_start
    }

    pub(crate) fn slice_stop(&self) -> usize {
        self.slice_stop
    }

    pub(crate) fn unsliced_n_rows(&self) -> usize {
        self.unsliced_n_rows
    }
}

impl ValidityVTable<Lz4> for Lz4 {
    fn validity(array: ArrayView<'_, Lz4>) -> VortexResult<Validity> {
        let unsliced_validity =
            child_to_validity(array.slots()[0].as_ref(), array.dtype().nullability());
        unsliced_validity.slice(array.slice_start()..array.slice_stop())
    }
}

impl OperationsVTable<Lz4> for Lz4 {
    fn scalar_at(
        array: ArrayView<'_, Lz4>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let unsliced_validity =
            child_to_validity(array.slots()[0].as_ref(), array.dtype().nullability());
        let sliced = array.data().with_slice(index, index + 1);
        sliced
            .decompress(array.dtype(), &unsliced_validity, ctx)?
            .execute_scalar(0, ctx)
    }
}

#[cfg(test)]
#[expect(clippy::cast_possible_truncation)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use super::reconstruct_views;
    use crate::array::BinaryView;

    /// Build an LZ4-style interleaved buffer: [u32-LE length][string bytes] repeated.
    fn make_interleaved(strings: &[&[u8]]) -> ByteBuffer {
        let mut buf = Vec::new();
        for s in strings {
            let len = s.len() as u32;
            buf.extend_from_slice(&len.to_le_bytes());
            buf.extend_from_slice(s);
        }
        ByteBuffer::copy_from(buf.as_slice())
    }

    #[test]
    fn test_reconstruct_views_no_split() {
        let strings: &[&[u8]] = &[b"hello", b"world"];
        let buf = make_interleaved(strings);
        let (buffers, views) = reconstruct_views(&buf, 1024);

        assert_eq!(buffers.len(), 1);
        assert_eq!(views.len(), 2);
        // Each entry: [u32 len (4 bytes)][data], so offsets are 4 and 4+5+4=13
        assert_eq!(views[0], BinaryView::make_view(b"hello", 0, 4));
        assert_eq!(views[1], BinaryView::make_view(b"world", 0, 13));
    }

    #[test]
    fn test_reconstruct_views_split_across_segments() {
        // "aaaaaaaaaaaaa" (13 bytes) and "bbbbbbbbbbbbb" (13 bytes).
        // Each entry occupies 4 (length prefix) + 13 (data) = 17 bytes.
        // With max_buffer_len=20, the second entry's data (offset 4+13+4=21) exceeds the limit,
        // so it rolls into a second segment.
        let strings: &[&[u8]] = &[b"aaaaaaaaaaaaa", b"bbbbbbbbbbbbb"];
        let buf = make_interleaved(strings);
        let (buffers, views) = reconstruct_views(&buf, 20);

        assert_eq!(buffers.len(), 2);
        assert_eq!(views.len(), 2);
        assert_eq!(views[0], BinaryView::make_view(b"aaaaaaaaaaaaa", 0, 4));
        // Second entry starts a new segment at byte 17 (the length prefix), so local offset = 4.
        assert_eq!(views[1], BinaryView::make_view(b"bbbbbbbbbbbbb", 1, 4));
    }
}
