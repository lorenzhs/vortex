// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::array_session;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::assert_arrays_eq;
use vortex_array::assert_nth_scalar;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_mask::Mask;

use crate::Lz4;

#[test]
fn test_lz4_compress_decompress() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::from_iter(data.clone());

    let compressed = Lz4::from_primitive(&array, 0, &mut ctx).unwrap();

    // check full decompression works
    let decompressed = Lz4::decompress(&compressed, &mut ctx).unwrap();
    assert_arrays_eq!(decompressed, PrimitiveArray::from_iter(data), &mut ctx);

    // check slicing works
    let slice = compressed.slice(100..105).unwrap();
    for i in 0_i32..5 {
        assert_nth_scalar!(slice, i as usize, 100 + i, &mut ctx);
    }
    assert_arrays_eq!(
        slice,
        PrimitiveArray::from_iter([100, 101, 102, 103, 104]),
        &mut ctx
    );

    let slice = compressed.slice(200..200).unwrap();
    assert_arrays_eq!(
        slice,
        PrimitiveArray::from_iter(Vec::<i32>::new()),
        &mut ctx
    );
}

#[test]
fn test_lz4_empty() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<i32> = vec![];
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let compressed = Lz4::from_primitive(&array, 100, &mut ctx).unwrap();

    assert_arrays_eq!(compressed, PrimitiveArray::from_iter(data), &mut ctx);
}

#[test]
fn test_lz4_with_validity_and_multi_frame() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<i32> = (0..200).collect();
    let mut validity: Vec<bool> = vec![false; 200];
    validity[3] = true;
    validity[177] = true;
    let array = PrimitiveArray::new(
        Buffer::from(data),
        Validity::Array(BoolArray::from_iter(validity).into_array()),
    );

    let compressed = Lz4::from_primitive(&array, 30, &mut ctx).unwrap();
    assert_nth_scalar!(compressed, 0, None::<i32>, &mut ctx);
    assert_nth_scalar!(compressed, 3, 3, &mut ctx);
    assert_nth_scalar!(compressed, 10, None::<i32>, &mut ctx);
    assert_nth_scalar!(compressed, 177, 177, &mut ctx);

    let decompressed = Lz4::decompress(&compressed, &mut ctx)
        .unwrap()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    let decompressed_values = decompressed.as_slice::<i32>();
    assert_eq!(decompressed_values[3], 3);
    assert_eq!(decompressed_values[177], 177);
    assert!(
        decompressed
            .validity()
            .unwrap()
            .mask_eq(&array.validity().unwrap(), decompressed.len(), &mut ctx)
            .unwrap()
    );

    // check slicing works
    let slice = compressed.slice(176..179).unwrap();
    let primitive = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();
    assert_eq!(
        i32::try_from(&primitive.execute_scalar(1, &mut ctx).unwrap()).unwrap(),
        177
    );
    assert!(
        primitive
            .validity()
            .unwrap()
            .mask_eq(
                &Validity::Array(BoolArray::from_iter(vec![false, true, false]).into_array()),
                primitive.len(),
                &mut ctx
            )
            .unwrap()
    );
}

#[test]
fn test_lz4_multi_frame_roundtrip() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<i32> = (0..200).collect();
    let array = PrimitiveArray::new(
        data.iter().cloned().collect::<Buffer<_>>(),
        Validity::NonNullable,
    );

    let compressed = Lz4::from_primitive(&array, 16, &mut ctx).unwrap();
    assert!(compressed.metadata.frames.len() > 1);
    assert_nth_scalar!(compressed, 0, 0, &mut ctx);
    assert_nth_scalar!(compressed, 199, 199, &mut ctx);

    let decompressed = Lz4::decompress(&compressed, &mut ctx)
        .unwrap()
        .execute::<PrimitiveArray>(&mut ctx)
        .unwrap();
    assert_arrays_eq!(decompressed, PrimitiveArray::from_iter(data), &mut ctx);

    // check slicing works
    let slice = compressed.slice(176..179).unwrap();
    let primitive = slice.execute::<PrimitiveArray>(&mut ctx).unwrap();
    assert_arrays_eq!(
        primitive,
        PrimitiveArray::from_iter([176, 177, 178]),
        &mut ctx
    );
}

#[test]
fn test_validity_vtable() {
    let mut ctx = array_session().create_execution_ctx();
    let mask_bools = vec![false, true, true, false, true];
    let array = PrimitiveArray::new(
        (0..5).collect::<Buffer<_>>(),
        Validity::Array(BoolArray::from_iter(mask_bools.clone()).into_array()),
    );
    let compressed = Lz4::from_primitive(&array, 0, &mut ctx).unwrap();
    let arr = compressed.as_array();
    assert_eq!(
        arr.validity()
            .unwrap()
            .execute_mask(arr.len(), &mut array_session().create_execution_ctx())
            .unwrap(),
        Mask::from_iter(mask_bools)
    );
    let sliced = compressed.slice(1..4).unwrap();
    assert_eq!(
        sliced
            .validity()
            .unwrap()
            .execute_mask(sliced.len(), &mut array_session().create_execution_ctx())
            .unwrap(),
        Mask::from_iter(vec![true, true, false])
    );
}

#[test]
fn test_lz4_var_bin_view() {
    let mut ctx = array_session().create_execution_ctx();
    let data: [Option<&'static [u8]>; 5] = [
        Some(b"foo"),
        Some(b"bar"),
        None,
        Some(b"Lorem ipsum dolor sit amet"),
        Some(b"baz"),
    ];
    let array = VarBinViewArray::from_iter(data, DType::Utf8(Nullability::Nullable));

    let compressed = Lz4::from_var_bin_view(&array, 3, &mut ctx).unwrap();
    assert_nth_scalar!(compressed, 0, "foo", &mut ctx);
    assert_nth_scalar!(compressed, 1, "bar", &mut ctx);
    assert_nth_scalar!(compressed, 2, None::<String>, &mut ctx);
    assert_nth_scalar!(compressed, 3, "Lorem ipsum dolor sit amet", &mut ctx);
    assert_nth_scalar!(compressed, 4, "baz", &mut ctx);

    let sliced = compressed.slice(1..4).unwrap();
    assert_nth_scalar!(sliced, 0, "bar", &mut ctx);
    assert_nth_scalar!(sliced, 1, None::<String>, &mut ctx);
    assert_nth_scalar!(sliced, 2, "Lorem ipsum dolor sit amet", &mut ctx);
}

#[test]
fn test_lz4_decompress_var_bin_view() {
    let mut ctx = array_session().create_execution_ctx();
    let data: [Option<&'static [u8]>; 5] = [
        Some(b"foo"),
        Some(b"bar"),
        None,
        Some(b"Lorem ipsum dolor sit amet"),
        Some(b"baz"),
    ];
    let array = VarBinViewArray::from_iter(data, DType::Utf8(Nullability::Nullable));

    let compressed = Lz4::from_var_bin_view(&array, 3, &mut ctx).unwrap();

    let decompressed = Lz4::decompress(&compressed, &mut ctx)
        .unwrap()
        .execute::<VarBinViewArray>(&mut ctx)
        .unwrap();
    assert_nth_scalar!(decompressed, 0, "foo", &mut ctx);
    assert_nth_scalar!(decompressed, 1, "bar", &mut ctx);
    assert_nth_scalar!(decompressed, 2, None::<String>, &mut ctx);
    assert_nth_scalar!(decompressed, 3, "Lorem ipsum dolor sit amet", &mut ctx);
    assert_nth_scalar!(decompressed, 4, "baz", &mut ctx);
}

#[test]
fn test_sliced_array_children() {
    let mut ctx = array_session().create_execution_ctx();
    let data: Vec<Option<i32>> = (0..10).map(|v| (v != 5).then_some(v)).collect();
    let compressed =
        Lz4::from_primitive(&PrimitiveArray::from_option_iter(data), 100, &mut ctx).unwrap();
    let sliced = compressed.slice(0..4).unwrap();
    sliced.children();
}

/// Tests that each beginning of a frame in LZ4 matches
/// the buffer alignment when compressing primitive arrays.
#[test]
fn test_lz4_frame_start_buffer_alignment() {
    let mut ctx = array_session().create_execution_ctx();
    let data = vec![0u8; 2];
    let aligned_buffer = Buffer::copy_from_aligned(&data, Alignment::new(8));
    // u8 array now has a 8-byte alignment.
    let array = PrimitiveArray::new(aligned_buffer, Validity::NonNullable);
    let compressed = Lz4::from_primitive(&array, 1, &mut ctx);

    assert!(compressed.is_ok());
}
