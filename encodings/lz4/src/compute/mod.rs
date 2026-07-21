// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod cast;

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::consistency::test_array_consistency;
    use vortex_buffer::buffer;

    use crate::Lz4;
    use crate::Lz4Array;

    fn lz4_i32() -> Lz4Array {
        let values = PrimitiveArray::from_iter([100i32, 200, 300, 400, 500]);
        Lz4::from_primitive(&values, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn lz4_f64() -> Lz4Array {
        let values = PrimitiveArray::from_iter([1.1f64, 2.2, 3.3, 4.4, 5.5]);
        Lz4::from_primitive(&values, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn lz4_u32() -> Lz4Array {
        let values = PrimitiveArray::from_iter([10u32, 20, 30, 40, 50]);
        Lz4::from_primitive(&values, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn lz4_nullable_i64() -> Lz4Array {
        let values =
            PrimitiveArray::from_option_iter([Some(1000i64), None, Some(3000), Some(4000), None]);
        Lz4::from_primitive(&values, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn lz4_single() -> Lz4Array {
        let values = PrimitiveArray::new(
            buffer![42i64],
            vortex_array::validity::Validity::NonNullable,
        );
        Lz4::from_primitive(&values, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn lz4_large() -> Lz4Array {
        let values = PrimitiveArray::new(
            buffer![0u32..1000],
            vortex_array::validity::Validity::NonNullable,
        );
        Lz4::from_primitive(&values, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn lz4_all_same() -> Lz4Array {
        let values = PrimitiveArray::new(
            buffer![42i32; 100],
            vortex_array::validity::Validity::NonNullable,
        );
        Lz4::from_primitive(&values, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    fn lz4_negative() -> Lz4Array {
        let values = PrimitiveArray::from_iter([-100i32, -50, 0, 50, 100]);
        Lz4::from_primitive(&values, 0, &mut array_session().create_execution_ctx()).unwrap()
    }

    #[rstest]
    #[case::i32(lz4_i32())]
    #[case::f64(lz4_f64())]
    #[case::u32(lz4_u32())]
    #[case::nullable_i64(lz4_nullable_i64())]
    #[case::single(lz4_single())]
    #[case::large(lz4_large())]
    #[case::all_same(lz4_all_same())]
    #[case::negative(lz4_negative())]
    fn test_lz4_consistency(#[case] array: Lz4Array) {
        test_array_consistency(
            &array.into_array(),
            &mut array_session().create_execution_ctx(),
        );
    }
}
