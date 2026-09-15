    for (; y < nr / 4; y++) {

        const BnBlockQ8Kx4 * a_ptr = a_ptr_start + (y * nb);

        // Take group of eight block_q4_kx8 structures at each pass of the loop and perform dot product operation
        for (int64_t x = group_start * 2; x < group_end * 2; x += 2) {

            const BnBlockQ4Kx8 * b_ptr_0 = b_ptr_start + ((x) * b_nb);
            const BnBlockQ4Kx8 * b_ptr_1 = b_ptr_start + ((x + 1) * b_nb);

            // Master FP accumulators
            __m512 acc_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_rows[i] = _mm512_setzero_ps();
            }

            __m512 acc_min_rows[4];
            for (int i = 0; i < 4; i++) {
                acc_min_rows[i] = _mm512_setzero_ps();
            }

            // For super block
            for (int64_t b = 0; b < nb; b++) {
                // Scale values - Load the sixteen scale values from two block_q4_kx8 structures
                const __m512 col_scale_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].d, b_ptr_1[b].d);

                // dmin values - Load the sixteen dmin values from two block_q4_kx8 structures
                const __m512 col_dmin_f32 = GGML_F32Cx8x2_LOAD(b_ptr_0[b].dmin, b_ptr_1[b].dmin);

                // Loop to iterate over the eight sub blocks of a super block - two sub blocks are processed per iteration
                for (int sb = 0; sb < BN_QK_K / 64; sb++) {

                    const __m256i rhs_raw_mat_0123_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_4567_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_0123_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_4567_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_0123_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_4567_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_0123_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_4567_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_0[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_89AB_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_0 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 32 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 64 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_1 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 96 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 128 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_2 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 160 + sb * 256));
                    const __m256i rhs_raw_mat_89AB_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 192 + sb * 256));
                    const __m256i rhs_raw_mat_CDEF_3 = _mm256_loadu_si256((const __m256i * )(b_ptr_1[b].qs + 224 + sb * 256));

                    const __m256i rhs_raw_mat_0145_0 = _mm256_blend_epi32(rhs_raw_mat_0123_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_0, requiredOrder), rhs_raw_mat_4567_0, 240);
                    const __m256i rhs_raw_mat_0145_1 = _mm256_blend_epi32(rhs_raw_mat_0123_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_1, requiredOrder), rhs_raw_mat_4567_1, 240);
                    const __m256i rhs_raw_mat_0145_2 = _mm256_blend_epi32(rhs_raw_mat_0123_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_2, requiredOrder), rhs_raw_mat_4567_2, 240);
                    const __m256i rhs_raw_mat_0145_3 = _mm256_blend_epi32(rhs_raw_mat_0123_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_4567_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_2367_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_0123_3, requiredOrder), rhs_raw_mat_4567_3, 240);

                    const __m256i rhs_raw_mat_89CD_0 = _mm256_blend_epi32(rhs_raw_mat_89AB_0, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_0, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_0 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_0, requiredOrder), rhs_raw_mat_CDEF_0, 240);
                    const __m256i rhs_raw_mat_89CD_1 = _mm256_blend_epi32(rhs_raw_mat_89AB_1, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_1, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_1 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_1, requiredOrder), rhs_raw_mat_CDEF_1, 240);
                    const __m256i rhs_raw_mat_89CD_2 = _mm256_blend_epi32(rhs_raw_mat_89AB_2, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_2, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_2 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_2, requiredOrder), rhs_raw_mat_CDEF_2, 240);
                    const __m256i rhs_raw_mat_89CD_3 = _mm256_blend_epi32(rhs_raw_mat_89AB_3, _mm256_permutevar8x32_epi32(rhs_raw_mat_CDEF_3, requiredOrder), 240);
                    const __m256i rhs_raw_mat_ABEF_3 = _mm256_blend_epi32(_mm256_permutevar8x32_epi32(rhs_raw_mat_89AB_3, requiredOrder), rhs_raw_mat_CDEF_3, 240);

                    const __m512i rhs_raw_mat_014589CD_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_0), rhs_raw_mat_89CD_0, 1);
                    const __m512i rhs_raw_mat_2367ABEF_0 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_0), rhs_raw_mat_ABEF_0, 1);
                    const __m512i rhs_raw_mat_014589CD_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_1), rhs_raw_mat_89CD_1, 1);
                    const __m512i rhs_raw_mat_2367ABEF_1 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_1), rhs_raw_mat_ABEF_1, 1);

                    const __m512i rhs_raw_mat_014589CD_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_2), rhs_raw_mat_89CD_2, 1);
                    const __m512i rhs_raw_mat_2367ABEF_2 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_2), rhs_raw_mat_ABEF_2, 1);
                    const __m512i rhs_raw_mat_014589CD_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_0145_3), rhs_raw_mat_89CD_3, 1);
                    const __m512i rhs_raw_mat_2367ABEF_3 = _mm512_inserti32x8(_mm512_castsi256_si512(rhs_raw_mat_2367_3), rhs_raw_mat_ABEF_3, 1);

                    //4-bit -> 8-bit
                    const __m512i rhs_mat_014589CD_00 = _mm512_and_si512(rhs_raw_mat_014589CD_0, m4bexpanded); //B00(0-7) B01(0-7) B04(0-7) B05(0-7) B08(0-7) B09(0-7) B0C(0-7) B0D(0-7)
                    const __m512i rhs_mat_2367ABEF_00 = _mm512_and_si512(rhs_raw_mat_2367ABEF_0, m4bexpanded); //B02(0-7) B03(0-7) B06(0-7) B07(0-7) B0A(0-7) B0B(0-7) B0E(0-7) B0F(0-7)
                    const __m512i rhs_mat_014589CD_01 = _mm512_and_si512(rhs_raw_mat_014589CD_1, m4bexpanded); //B00(8-15) B01(8-15) B04(8-15) B05(8-15) B08(8-15) B09(8-15) B0C(8-15) B0D(8-15)
                    const __m512i rhs_mat_2367ABEF_01 = _mm512_and_si512(rhs_raw_mat_2367ABEF_1, m4bexpanded); //B02(8-15) B03(8-15) B06(8-15) B07(8-15) B0A(8-15) B0B(8-15) B0E(8-15) B0F(8-15)

                    const __m512i rhs_mat_014589CD_02 = _mm512_and_si512(rhs_raw_mat_014589CD_2, m4bexpanded); //B00(16-23) B01(16-23) B04(16-23) B05(16-23) B08(16-23) B09(16-23) B0C(16-23) B0D(16-23)
                    const __m512i rhs_mat_2367ABEF_02 = _mm512_and_si512(rhs_raw_mat_2367ABEF_2, m4bexpanded); //B02(16-23) B03(16-23) B06(16-23) B07(16-23) B0A(16-23) B0B(16-23) B0E(16-23) B0F(16-23)
                    const __m512i rhs_mat_014589CD_03 = _mm512_and_si512(rhs_raw_mat_014589CD_3, m4bexpanded); //B00(24-31) B01(24-31) B04(24-31) B05(24-31) B08(24-31) B09(24-31) B0C(24-31) B0D(24-31)
                    const __m512i rhs_mat_2367ABEF_03 = _mm512_and_si512(rhs_raw_mat_2367ABEF_3, m4bexpanded); //B02(24-31) B03(24-31) B06(24-31) B07(24-31) B0A(24-31) B0B(24-31) B0E(24-31) B0F(24-31)

                    const __m512i rhs_mat_014589CD_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_0, 4), m4bexpanded); //B10(0-7) B11(0-7) B14(0-7) B15(0-7) B18(0-7) B19(0-7) B1C(0-7) B1D(0-7)
                    const __m512i rhs_mat_2367ABEF_10 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_0, 4), m4bexpanded); //B12(0-7) B13(0-7) B16(0-7) B17(0-7) B1A(0-7) B1B(0-7) B1E(0-7) B1F(0-7)
                    const __m512i rhs_mat_014589CD_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_1, 4), m4bexpanded); //B10(8-15) B11(8-15) B14(8-15) B15(8-15) B18(8-15) B19(8-15) B1C(8-15) B1D(8-15)
                    const __m512i rhs_mat_2367ABEF_11 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_1, 4), m4bexpanded); //B12(8-15) B13(8-15) B16(8-15) B17(8-15) B1A(8-15) B1B(8-15) B1E(8-15) B1F(8-15)

                    const __m512i rhs_mat_014589CD_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_2, 4), m4bexpanded); //B10(16-23) B11(16-23) B14(16-23) B15(16-23) B18(16-23) B19(16-23) B1C(16-23) B1D(16-23)
                    const __m512i rhs_mat_2367ABEF_12 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_2, 4), m4bexpanded); //B12(16-23) B13(16-23) B16(16-23) B17(16-23) B1A(16-23) B1B(16-23) B1E(16-23) B1F(16-23)
                    const __m512i rhs_mat_014589CD_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_014589CD_3, 4), m4bexpanded); //B10(24-31) B11(24-31) B14(24-31) B15(24-31) B18(24-31) B19(24-31) B1C(24-31) B1D(24-31)
                    const __m512i rhs_mat_2367ABEF_13 = _mm512_and_si512(_mm512_srli_epi16(rhs_raw_mat_2367ABEF_3, 4), m4bexpanded); //B12(24-31) B13(24-31) B16(24-31) B17(24-31) B1A(24-31) B1B(24-31) B1E(24-31) B1F(24-31)

                    // Shuffle pattern one - right side input
                    const __m512i rhs_mat_014589CD_00_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, 136); //B00(0-3) B01(0-3) B00(0-3) B01(0-3) B04(0-3) B05(0-3) B04(0-3) B05(0-3) B08(0-3) B09(0-3) B08(0-3) B09(0-3) B0C(0-3) B0D(0-3) B0C(0-3) B0D(0-3)
                    const __m512i rhs_mat_2367ABEF_00_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, 136); //B02(0-3) B03(0-3) B02(0-3) B03(0-3) B06(0-3) B07(0-3) B06(0-3) B07(0-3) B0A(0-3) B0B(0-3) B0A(0-3) B0B(0-3) B0E(0-3) B0F(0-3) B0E(0-3) B0F(0-3)
                    const __m512i rhs_mat_014589CD_01_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, 136); //B00(8-11) B01(8-11) B00(8-11) B01(8-11) B04(8-11) B05(8-11) B04(8-11) B05(8-11) B08(8-11) B09(8-11) B08(8-11) B09(8-11) B0C(8-11) B0D(8-11) B0C(8-11) B0D(8-11)
                    const __m512i rhs_mat_2367ABEF_01_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, 136); //B02(8-11) B03(8-11) B02(8-11) B03(8-11) B06(8-11) B07(8-11) B06(8-11) B07(8-11) B0A(8-11) B0B(8-11) B0A(8-11) B0B(8-11) B0E(8-11) B0F(8-11) B0E(8-11) B0F(8-11)
                    const __m512i rhs_mat_014589CD_02_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, 136); //B00(16-19) B01(16-19) B00(16-19) B01(16-19) B04(16-19) B05(16-19) B04(16-19) B05(16-19) B08(16-19) B09(16-19) B08(16-19) B09(16-19) B0C(16-19) B0D(16-19) B0C(16-19) B0D(16-19)
                    const __m512i rhs_mat_2367ABEF_02_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, 136); //B02(16-19) B03(16-19) B02(16-19) B03(16-19) B06(16-19) B07(16-19) B06(16-19) B07(16-19) B0A(16-19) B0B(16-19) B0A(16-19) B0B(16-19) B0E(16-19) B0F(16-19) B0E(16-19) B0F(16-19)
                    const __m512i rhs_mat_014589CD_03_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, 136); //B00(24-27) B01(24-27) B00(24-27) B01(24-27) B04(24-27) B05(24-27) B04(24-27) B05(24-27) B08(24-27) B09(24-27) B08(24-27) B09(24-27) B0C(24-27) B0D(24-27) B0C(24-27) B0D(24-27)
                    const __m512i rhs_mat_2367ABEF_03_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, 136); //B02(24-27) B03(24-27) B02(24-27) B03(24-27) B06(24-27) B07(24-27) B06(24-27) B07(24-27) B0A(24-27) B0B(24-27) B0A(24-27) B0B(24-27) B0E(24-27) B0F(24-27) B0E(24-27) B0F(24-27)

                    const __m512i rhs_mat_014589CD_10_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, 136); //B10(0-3) B11(0-3) B10(0-3) B11(0-3) B14(0-3) B15(0-3) B14(0-3) B15(0-3) B18(0-3) B19(0-3) B18(0-3) B19(0-3) B1C(0-3) B1D(0-3) B1C(0-3) B1D(0-3)
                    const __m512i rhs_mat_2367ABEF_10_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, 136); //B12(0-3) B13(0-3) B12(0-3) B13(0-3) B16(0-3) B17(0-3) B16(0-3) B17(0-3) B1A(0-3) B1B(0-3) B1A(0-3) B1B(0-3) B1E(0-3) B1F(0-3) B1E(0-3) B1F(0-3)
                    const __m512i rhs_mat_014589CD_11_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, 136); //B10(8-11) B11(8-11) B10(8-11) B11(8-11) B14(8-11) B15(8-11) B14(8-11) B15(8-11) B18(8-11) B19(8-11) B18(8-11) B19(8-11) B1C(8-11) B1D(8-11) B1C(8-11) B1D(8-11)
                    const __m512i rhs_mat_2367ABEF_11_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, 136); //B12(8-11) B13(8-11) B12(8-11) B13(8-11) B16(8-11) B17(8-11) B16(8-11) B17(8-11) B1A(8-11) B1B(8-11) B1A(8-11) B1B(8-11) B1E(8-11) B1F(8-11) B1E(8-11) B1F(8-11)
                    const __m512i rhs_mat_014589CD_12_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, 136); //B10(16-19) B11(16-19) B10(16-19) B11(16-19) B14(16-19) B15(16-19) B14(16-19) B15(16-19) B18(16-19) B19(16-19) B18(16-19) B19(16-19) B1C(16-19) B1D(16-19) B1C(16-19) B1D(16-19)
                    const __m512i rhs_mat_2367ABEF_12_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, 136); //B12(16-19) B13(16-19) B12(16-19) B13(16-19) B16(16-19) B17(16-19) B16(16-19) B17(16-19) B1A(16-19) B1B(16-19) B1A(16-19) B1B(16-19) B1E(16-19) B1F(16-19) B1E(16-19) B1F(16-19)
                    const __m512i rhs_mat_014589CD_13_sp1 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, 136); //B10(24-27) B11(24-27) B10(24-27) B11(24-27) B14(24-27) B15(24-27) B14(24-27) B15(24-27) B18(24-27) B19(24-27) B18(24-27) B19(24-27) B1C(24-27) B1D(24-27) B1C(24-27) B1D(24-27)
                    const __m512i rhs_mat_2367ABEF_13_sp1 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, 136); //B12(24-27) B13(24-27) B12(24-27) B13(24-27) B16(24-27) B17(24-27) B16(24-27) B17(24-27) B1A(24-27) B1B(24-27) B1A(24-27) B1B(24-27) B1E(24-27) B1F(24-27) B1E(24-27) B1F(24-27)

                    // Shuffle pattern two - right side input
                    const __m512i rhs_mat_014589CD_00_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_00, 221); //B00(4-7) B01(4-7) B00(4-7) B01(4-7) B04(4-7) B05(4-7) B04(4-7) B05(4-7) B08(4-7) B09(4-7) B08(4-7) B09(4-7) B0C(4-7) B0D(4-7) B0C(4-7) B0D(4-7)
                    const __m512i rhs_mat_2367ABEF_00_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_00, 221); //B02(4-7) B03(4-7) B02(4-7) B03(4-7) B06(4-7) B07(4-7) B06(4-7) B07(4-7) B0A(4-7) B0B(4-7) B0A(4-7) B0B(4-7) B0E(4-7) B0F(4-7) B0E(4-7) B0F(4-7)
                    const __m512i rhs_mat_014589CD_01_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_01, 221); //B00(12-15) B01(12-15) B00(12-15) B01(12-15) B04(12-15) B05(12-15) B04(12-15) B05(12-15) B08(12-15) B09(12-15) B08(12-15) B09(12-15) B0C(12-15) B0D(12-15) B0C(12-15) B0D(12-15)
                    const __m512i rhs_mat_2367ABEF_01_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_01, 221); //B02(12-15) B03(12-15) B02(12-15) B03(12-15) B06(12-15) B07(12-15) B06(12-15) B07(12-15) B0A(12-15) B0B(12-15) B0A(12-15) B0B(12-15) B0E(12-15) B0F(12-15) B0E(12-15) B0F(12-15)
                    const __m512i rhs_mat_014589CD_02_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_02, 221); //B00(20-23) B01(20-23) B00(20-23) B01(20-23) B04(20-23) B05(20-23) B04(20-23) B05(20-23) B08(20-23) B09(20-23) B08(20-23) B09(20-23) B0C(20-23) B0D(20-23) B0C(20-23) B0D(20-23)
                    const __m512i rhs_mat_2367ABEF_02_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_02, 221); //B02(20-23) B03(20-23) B02(20-23) B03(20-23) B06(20-23) B07(20-23) B06(20-23) B07(20-23) B0A(20-23) B0B(20-23) B0A(20-23) B0B(20-23) B0E(20-23) B0F(20-23) B0E(20-23) B0F(20-23)
                    const __m512i rhs_mat_014589CD_03_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_03, 221); //B00(28-31) B01(28-31) B00(28-31) B01(28-31) B04(28-31) B05(28-31) B04(28-31) B05(28-31) B08(28-31) B09(28-31) B08(28-31) B09(28-31) B0C(28-31) B0D(28-31) B0C(28-31) 0BD(28-31)
                    const __m512i rhs_mat_2367ABEF_03_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_03, 221); //B02(28-31) B03(28-31) B02(28-31) B03(28-31) B06(28-31) B07(28-31) B06(28-31) B07(28-31) B0A(28-31) B0B(28-31) B0A(28-31) B0B(28-31) B0E(28-31) B0F(28-31) B0E(28-31) B0F(28-31)

                    const __m512i rhs_mat_014589CD_10_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_10, 221); //B10(4-7) B11(4-7) B10(4-7) B11(4-7) B14(4-7) B15(4-7) B14(4-7) B15(4-7) B18(4-7) B19(4-7) B18(4-7) B19(4-7) B1C(4-7) B1D(4-7) B1C(4-7) B1D(4-7)
                    const __m512i rhs_mat_2367ABEF_10_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_10, 221); //B12(4-7) B13(4-7) B12(4-7) B13(4-7) B16(4-7) B17(4-7) B16(4-7) B17(4-7) B1A(4-7) B1B(4-7) B1A(4-7) B1B(4-7) B1E(4-7) B1F(4-7) B1E(4-7) B1F(4-7)
                    const __m512i rhs_mat_014589CD_11_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_11, 221); //B10(12-15) B11(12-15) B10(12-15) B11(12-15) B14(12-15) B15(12-15) B14(12-15) B15(12-15) B18(12-15) B19(12-15) B18(12-15) B19(12-15) B1C(12-15) B1D(12-15) B1C(12-15) B1D(12-15)
                    const __m512i rhs_mat_2367ABEF_11_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_11, 221); //B12(12-15) B13(12-15) B12(12-15) B13(12-15) B16(12-15) B17(12-15) B16(12-15) B17(12-15) B1A(12-15) B1B(12-15) B1A(12-15) B1B(12-15) B1E(12-15) B1F(12-15) B1E(12-15) B1F(12-15)
                    const __m512i rhs_mat_014589CD_12_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_12, 221); //B10(20-23) B11(20-23) B10(20-23) B11(20-23) B14(20-23) B15(20-23) B14(20-23) B15(20-23) B18(20-23) B19(20-23) B18(20-23) B19(20-23) B1C(20-23) B1D(20-23) B1C(20-23) B1D(20-23)
                    const __m512i rhs_mat_2367ABEF_12_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_12, 221); //B12(20-23) B13(20-23) B12(20-23) B13(20-23) B16(20-23) B17(20-23) B16(20-23) B17(20-23) B1A(20-23) B1B(20-23) B1A(20-23) B1B(20-23) B1E(20-23) B1F(20-23) B1E(20-23) B1F(20-23)
                    const __m512i rhs_mat_014589CD_13_sp2 = _mm512_shuffle_epi32(rhs_mat_014589CD_13, 221); //B10(28-31) B11(28-31) B10(28-31) B11(28-31) B14(28-31) B15(28-31) B14(28-31) B15(28-31) B18(28-31) B19(28-31) B18(28-31) B19(28-31) B1C(28-31) B1D(28-31) B1C(28-31) B1D(28-31)
                    const __m512i rhs_mat_2367ABEF_13_sp2 = _mm512_shuffle_epi32(rhs_mat_2367ABEF_13, 221); //B12(28-31) B13(28-31) B12(28-31) B13(28-31) B16(28-31) B17(28-31) B16(28-31) B17(28-31) B1A(28-31) B1B(28-31) B1A(28-31) B1B(28-31) B1E(28-31) B1F(28-31) B1E(28-31) B1F(28-31)

                    uint32_t utmp_00[4], utmp_01[4], utmp_10[4], utmp_11[4];

                    // Scales and Mins of corresponding sub blocks from different Q4_K structures are stored together
                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_00, b_ptr_0[b].scales + 24 * sb, 12);
                    utmp_00[3] = ((utmp_00[2] >> 4) & kmask2) | (((utmp_00[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_00 = utmp_00[1] & kmask1;
                    utmp_00[1] = (utmp_00[2] & kmask2) | (((utmp_00[0] >> 6) & kmask3) << 4);
                    utmp_00[2] = uaux_00;
                    utmp_00[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_01, b_ptr_0[b].scales + 12 + sb * 24, 12);
                    utmp_01[3] = ((utmp_01[2] >> 4) & kmask2) | (((utmp_01[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_01 = utmp_01[1] & kmask1;
                    utmp_01[1] = (utmp_01[2] & kmask2) | (((utmp_01[0] >> 6) & kmask3) << 4);
                    utmp_01[2] = uaux_01;
                    utmp_01[0] &= kmask1;

                    // The below block is for eg to extract first sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_10, b_ptr_1[b].scales + sb * 24, 12);
                    utmp_10[3] = ((utmp_10[2] >> 4) & kmask2) | (((utmp_10[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_10 = utmp_10[1] & kmask1;
                    utmp_10[1] = (utmp_10[2] & kmask2) | (((utmp_10[0] >> 6) & kmask3) << 4);
                    utmp_10[2] = uaux_10;
                    utmp_10[0] &= kmask1;

                    // The below block is for eg to extract second sub block's scales and mins from different Q4_K structures for the sb loop
                    memcpy(utmp_11, b_ptr_1[b].scales + 12 + sb * 24, 12);
                    utmp_11[3] = ((utmp_11[2] >> 4) & kmask2) | (((utmp_11[1] >> 6) & kmask3) << 4);
                    const uint32_t uaux_11 = utmp_11[1] & kmask1;
                    utmp_11[1] = (utmp_11[2] & kmask2) | (((utmp_11[0] >> 6) & kmask3) << 4);
                    utmp_11[2] = uaux_11;
                    utmp_11[0] &= kmask1;

                    // Scales of first sub block in the sb loop
                    const __m256i mins_and_scales_0 = _mm256_set_epi32(utmp_10[3], utmp_10[2], utmp_10[1], utmp_10[0], utmp_00[3], utmp_00[2], utmp_00[1], utmp_00[0]);
                    const __m512i scales_0 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_0, mins_and_scales_0));

                    // Scales of second sub block in the sb loop
                    const __m256i mins_and_scales_1 = _mm256_set_epi32(utmp_11[3], utmp_11[2], utmp_11[1], utmp_11[0], utmp_01[3], utmp_01[2], utmp_01[1], utmp_01[0]);
                    const __m512i scales_1 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(mins_and_scales_1, mins_and_scales_1));

                    // Mins of first and second sub block of Q4_K block are arranged side by side
                    const __m512i mins_01 = _mm512_cvtepu8_epi16(_mm256_unpacklo_epi8(_mm256_shuffle_epi32(mins_and_scales_0, 78), _mm256_shuffle_epi32(mins_and_scales_1, 78)));

                    const __m512i scale_014589CD_0 = _mm512_shuffle_epi32(scales_0, 68);
                    const __m512i scale_2367ABEF_0 = _mm512_shuffle_epi32(scales_0, 238);

                    const __m512i scale_014589CD_1 = _mm512_shuffle_epi32(scales_1, 68);
                    const __m512i scale_2367ABEF_1 = _mm512_shuffle_epi32(scales_1, 238);

                    // Load the four block_q8_k quantized values interleaved with each other in chunks of eight bytes - A0,A1,A2,A3
                    // Loaded as set of 128 bit vectors and repeated into a 256 bit vector
                    __m256i lhs_mat_ymm_0123_00 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 256 * sb)));
                    __m256i lhs_mat_ymm_01_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 0);
                    __m256i lhs_mat_ymm_23_00 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_00, lhs_mat_ymm_0123_00, 17);
                    __m256i lhs_mat_ymm_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 32 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 0);
                    __m256i lhs_mat_ymm_23_01 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_01, lhs_mat_ymm_0123_01, 17);
                    __m256i lhs_mat_ymm_0123_02 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 64 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 0);
                    __m256i lhs_mat_ymm_23_02 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_02, lhs_mat_ymm_0123_02, 17);
                    __m256i lhs_mat_ymm_0123_03 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 96 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 0);
                    __m256i lhs_mat_ymm_23_03 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_03, lhs_mat_ymm_0123_03, 17);
                    __m256i lhs_mat_ymm_0123_10 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 128 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 0);
                    __m256i lhs_mat_ymm_23_10 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_10, lhs_mat_ymm_0123_10, 17);
                    __m256i lhs_mat_ymm_0123_11 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 160 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 0);
                    __m256i lhs_mat_ymm_23_11 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_11, lhs_mat_ymm_0123_11, 17);
                    __m256i lhs_mat_ymm_0123_12 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 192 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 0);
                    __m256i lhs_mat_ymm_23_12 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_12, lhs_mat_ymm_0123_12, 17);
                    __m256i lhs_mat_ymm_0123_13 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].qs + 224 + 256 * sb)));
                    __m256i lhs_mat_ymm_01_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 0);
                    __m256i lhs_mat_ymm_23_13 = _mm256_permute2f128_si256(lhs_mat_ymm_0123_13, lhs_mat_ymm_0123_13, 17);

                    //Loaded as set of 128 bit vectors and repeated and stored into a 256 bit vector before again repeating into a 512 bit vector
                    __m512i lhs_mat_01_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_00), lhs_mat_ymm_01_00, 1);
                    __m512i lhs_mat_23_00 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_00), lhs_mat_ymm_23_00, 1);
                    __m512i lhs_mat_01_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_01), lhs_mat_ymm_01_01, 1);
                    __m512i lhs_mat_23_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_01), lhs_mat_ymm_23_01, 1);
                    __m512i lhs_mat_01_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_02), lhs_mat_ymm_01_02, 1);
                    __m512i lhs_mat_23_02 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_02), lhs_mat_ymm_23_02, 1);
                    __m512i lhs_mat_01_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_03), lhs_mat_ymm_01_03, 1);
                    __m512i lhs_mat_23_03 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_03), lhs_mat_ymm_23_03, 1);

                    __m512i lhs_mat_01_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_10), lhs_mat_ymm_01_10, 1);
                    __m512i lhs_mat_23_10 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_10), lhs_mat_ymm_23_10, 1);
                    __m512i lhs_mat_01_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_11), lhs_mat_ymm_01_11, 1);
                    __m512i lhs_mat_23_11 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_11), lhs_mat_ymm_23_11, 1);
                    __m512i lhs_mat_01_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_12), lhs_mat_ymm_01_12, 1);
                    __m512i lhs_mat_23_12 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_12), lhs_mat_ymm_23_12, 1);
                    __m512i lhs_mat_01_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_01_13), lhs_mat_ymm_01_13, 1);
                    __m512i lhs_mat_23_13 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_mat_ymm_23_13), lhs_mat_ymm_23_13, 1);

                    // Bsums are loaded - four bsums are loaded (for two sub blocks) for the different Q8_K blocks
                    __m256i lhs_bsums_0123_01 = _mm256_loadu_si256((const __m256i * )((a_ptr[b].bsums + 16 * sb)));
                    __m256i lhs_bsums_hsum_ymm_0123_01 = _mm256_castsi128_si256(_mm_hadd_epi16(_mm256_castsi256_si128(lhs_bsums_0123_01), _mm256_extractf128_si256(lhs_bsums_0123_01, 1)));
                    lhs_bsums_hsum_ymm_0123_01 = _mm256_permute2x128_si256(lhs_bsums_hsum_ymm_0123_01, lhs_bsums_hsum_ymm_0123_01, 0);
                    __m512i lhs_bsums_hsum_0123_01 = _mm512_inserti32x8(_mm512_castsi256_si512(lhs_bsums_hsum_ymm_0123_01), lhs_bsums_hsum_ymm_0123_01, 1);

                    // Shuffle pattern one - left side input
                    const __m512i lhs_mat_01_00_sp1 = _mm512_shuffle_epi32(lhs_mat_01_00, 160); //A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3) A00(0-3) A00(0-3) A01(0-3) A01(0-3)
                    const __m512i lhs_mat_23_00_sp1 = _mm512_shuffle_epi32(lhs_mat_23_00, 160); //A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3) A02(0-3) A02(0-3) A03(0-3) A03(0-3)
                    const __m512i lhs_mat_01_01_sp1 = _mm512_shuffle_epi32(lhs_mat_01_01, 160); //A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11) A00(8-11) A00(8-11) A01(8-11) A01(8-11)
                    const __m512i lhs_mat_23_01_sp1 = _mm512_shuffle_epi32(lhs_mat_23_01, 160); //A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11) A02(8-11) A02(8-11) A03(8-11) A03(8-11)
                    const __m512i lhs_mat_01_02_sp1 = _mm512_shuffle_epi32(lhs_mat_01_02, 160); //A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19) A00(16-19) A00(16-19) A01(16-19) A01(16-19)
                    const __m512i lhs_mat_23_02_sp1 = _mm512_shuffle_epi32(lhs_mat_23_02, 160); //A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19) A02(16-19) A02(16-19) A03(16-19) A03(16-19)
                    const __m512i lhs_mat_01_03_sp1 = _mm512_shuffle_epi32(lhs_mat_01_03, 160); //A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27) A00(24-27) A00(24-27) A01(24-27) A01(24-27)
                    const __m512i lhs_mat_23_03_sp1 = _mm512_shuffle_epi32(lhs_mat_23_03, 160); //A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27) A02(24-27) A02(24-27) A03(24-27) A03(24-27)

                    const __m512i lhs_mat_01_10_sp1 = _mm512_shuffle_epi32(lhs_mat_01_10, 160); //A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3) A10(0-3) A10(0-3) A11(0-3) A11(0-3)
                    const __m512i lhs_mat_23_10_sp1 = _mm512_shuffle_epi32(lhs_mat_23_10, 160); //A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3) A12(0-3) A12(0-3) A13(0-3) A13(0-3)
                    const __m512i lhs_mat_01_11_sp1 = _mm512_shuffle_epi32(lhs_mat_01_11, 160); //A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11) A10(8-11) A10(8-11) A11(8-11) A11(8-11)
                    const __m512i lhs_mat_23_11_sp1 = _mm512_shuffle_epi32(lhs_mat_23_11, 160); //A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11) A12(8-11) A12(8-11) A13(8-11) A13(8-11)
                    const __m512i lhs_mat_01_12_sp1 = _mm512_shuffle_epi32(lhs_mat_01_12, 160); //A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19) A10(16-19) A10(16-19) A11(16-19) A11(16-19)
                    const __m512i lhs_mat_23_12_sp1 = _mm512_shuffle_epi32(lhs_mat_23_12, 160); //A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19) A12(16-19) A12(16-19) A13(16-19) A13(16-19)
                    const __m512i lhs_mat_01_13_sp1 = _mm512_shuffle_epi32(lhs_mat_01_13, 160); //A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27) A10(24-27) A10(24-27) A11(24-27) A11(24-27)
                    const __m512i lhs_mat_23_13_sp1 = _mm512_shuffle_epi32(lhs_mat_23_13, 160); //A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27) A12(24-27) A12(24-27) A13(24-27) A13(24-27)

                    const __m512i lhs_mat_01_00_sp2 = _mm512_shuffle_epi32(lhs_mat_01_00, 245); //A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7) A00(4-7) A00(4-7) A01(4-7) A01(4-7)
                    const __m512i lhs_mat_23_00_sp2 = _mm512_shuffle_epi32(lhs_mat_23_00, 245); //A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7) A02(4-7) A02(4-7) A03(4-7) A03(4-7)
                    const __m512i lhs_mat_01_01_sp2 = _mm512_shuffle_epi32(lhs_mat_01_01, 245); //A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15) A00(12-15) A00(12-15) A01(12-15) A01(12-15)
                    const __m512i lhs_mat_23_01_sp2 = _mm512_shuffle_epi32(lhs_mat_23_01, 245); //A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15) A02(12-15) A02(12-15) A03(12-15) A03(12-15)
                    const __m512i lhs_mat_01_02_sp2 = _mm512_shuffle_epi32(lhs_mat_01_02, 245); //A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23) A00(20-23) A00(20-23) A01(20-23) A01(20-23)
                    const __m512i lhs_mat_23_02_sp2 = _mm512_shuffle_epi32(lhs_mat_23_02, 245); //A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23) A02(20-23) A02(20-23) A03(20-23) A03(20-23)
                    const __m512i lhs_mat_01_03_sp2 = _mm512_shuffle_epi32(lhs_mat_01_03, 245); //A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31) A00(28-31) A00(28-31) A01(28-31) A01(28-31)
                    const __m512i lhs_mat_23_03_sp2 = _mm512_shuffle_epi32(lhs_mat_23_03, 245); //A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31) A02(28-31) A02(28-31) A03(28-31) A03(28-31)

                    const __m512i lhs_mat_01_10_sp2 = _mm512_shuffle_epi32(lhs_mat_01_10, 245); //A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7) A10(4-7) A10(4-7) A11(4-7) A11(4-7)
                    const __m512i lhs_mat_23_10_sp2 = _mm512_shuffle_epi32(lhs_mat_23_10, 245); //A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7) A12(4-7) A12(4-7) A13(4-7) A13(4-7)
                    const __m512i lhs_mat_01_11_sp2 = _mm512_shuffle_epi32(lhs_mat_01_11, 245); //A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15) A10(12-15) A10(12-15) A11(12-15) A11(12-15)
                    const __m512i lhs_mat_23_11_sp2 = _mm512_shuffle_epi32(lhs_mat_23_11, 245); //A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15) A12(12-15) A12(12-15) A13(12-15) A13(12-15)
                    const __m512i lhs_mat_01_12_sp2 = _mm512_shuffle_epi32(lhs_mat_01_12, 245); //A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23) A10(20-23) A10(20-23) A11(20-23) A11(20-23)
                    const __m512i lhs_mat_23_12_sp2 = _mm512_shuffle_epi32(lhs_mat_23_12, 245); //A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23) A12(20-23) A12(20-23) A13(20-23) A13(20-23)
                    const __m512i lhs_mat_01_13_sp2 = _mm512_shuffle_epi32(lhs_mat_01_13, 245); //A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31) A10(28-31) A10(28-31) A11(28-31) A11(28-31)
                    const __m512i lhs_mat_23_13_sp2 = _mm512_shuffle_epi32(lhs_mat_23_13, 245); //A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31) A12(28-31) A12(28-31) A13(28-31) A13(28-31)

                    // The values arranged in shuffle patterns are operated with dot product operation within 32 bit lane i.e corresponding bytes and multiplied and added into 32 bit integers within 32 bit lane
                    __m512i iacc_mat_00_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_01_00_sp1));
                    __m512i iacc_mat_01_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_01_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_01_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_01_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_01_00_sp1));
                    __m512i iacc_mat_10_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp1, lhs_mat_23_00_sp1));
                    __m512i iacc_mat_11_0_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp1, lhs_mat_23_03_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp1, lhs_mat_23_02_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp1, lhs_mat_23_01_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp1, lhs_mat_23_00_sp1));
                    __m512i iacc_mat_00_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_01_10_sp1));
                    __m512i iacc_mat_01_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_01_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_01_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_01_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_01_10_sp1));
                    __m512i iacc_mat_10_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp1, lhs_mat_23_10_sp1));
                    __m512i iacc_mat_11_1_sp1 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp1, lhs_mat_23_13_sp1), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp1, lhs_mat_23_12_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp1, lhs_mat_23_11_sp1)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp1, lhs_mat_23_10_sp1));

                    __m512i iacc_mat_00_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_01_00_sp2));
                    __m512i iacc_mat_01_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_01_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_01_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_01_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_01_00_sp2));
                    __m512i iacc_mat_10_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_00_sp2, lhs_mat_23_00_sp2));
                    __m512i iacc_mat_11_0_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_03_sp2, lhs_mat_23_03_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_02_sp2, lhs_mat_23_02_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_01_sp2, lhs_mat_23_01_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_00_sp2, lhs_mat_23_00_sp2));
                    __m512i iacc_mat_00_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_01_10_sp2));
                    __m512i iacc_mat_01_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_01_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_01_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_01_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_01_10_sp2));
                    __m512i iacc_mat_10_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_014589CD_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_014589CD_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_014589CD_10_sp2, lhs_mat_23_10_sp2));
                    __m512i iacc_mat_11_1_sp2 = _mm512_add_epi16(_mm512_add_epi16(_mm512_add_epi16(_mm512_maddubs_epi16(rhs_mat_2367ABEF_13_sp2, lhs_mat_23_13_sp2), _mm512_maddubs_epi16(rhs_mat_2367ABEF_12_sp2, lhs_mat_23_12_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_11_sp2, lhs_mat_23_11_sp2)), _mm512_maddubs_epi16(rhs_mat_2367ABEF_10_sp2, lhs_mat_23_10_sp2));

                    // Output of both shuffle patterns are added in order to sum dot product outputs of all 32 values in block
                    __m512i iacc_mat_00_0 = _mm512_add_epi16(iacc_mat_00_0_sp1, iacc_mat_00_0_sp2);
                    __m512i iacc_mat_01_0 = _mm512_add_epi16(iacc_mat_01_0_sp1, iacc_mat_01_0_sp2);
                    __m512i iacc_mat_10_0 = _mm512_add_epi16(iacc_mat_10_0_sp1, iacc_mat_10_0_sp2);
                    __m512i iacc_mat_11_0 = _mm512_add_epi16(iacc_mat_11_0_sp1, iacc_mat_11_0_sp2);

                    __m512i iacc_mat_00_1 = _mm512_add_epi16(iacc_mat_00_1_sp1, iacc_mat_00_1_sp2);
                    __m512i iacc_mat_01_1 = _mm512_add_epi16(iacc_mat_01_1_sp1, iacc_mat_01_1_sp2);
                    __m512i iacc_mat_10_1 = _mm512_add_epi16(iacc_mat_10_1_sp1, iacc_mat_10_1_sp2);
                    __m512i iacc_mat_11_1 = _mm512_add_epi16(iacc_mat_11_1_sp1, iacc_mat_11_1_sp2);

                    iacc_mat_00_0 = _mm512_madd_epi16(iacc_mat_00_0, scale_014589CD_0);
                    iacc_mat_01_0 = _mm512_madd_epi16(iacc_mat_01_0, scale_2367ABEF_0);
                    iacc_mat_10_0 = _mm512_madd_epi16(iacc_mat_10_0, scale_014589CD_0);
                    iacc_mat_11_0 = _mm512_madd_epi16(iacc_mat_11_0, scale_2367ABEF_0);

                    iacc_mat_00_1 = _mm512_madd_epi16(iacc_mat_00_1, scale_014589CD_1);
                    iacc_mat_01_1 = _mm512_madd_epi16(iacc_mat_01_1, scale_2367ABEF_1);
                    iacc_mat_10_1 = _mm512_madd_epi16(iacc_mat_10_1, scale_014589CD_1);
                    iacc_mat_11_1 = _mm512_madd_epi16(iacc_mat_11_1, scale_2367ABEF_1);

                    // Straighten out to make 4 row vectors (4 for each sub block which are accumulated together in the next step)
                    __m512i iacc_row_0_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_0, _mm512_shuffle_epi32(iacc_mat_01_0, 78));
                    __m512i iacc_row_1_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_0, 78), iacc_mat_01_0);
                    __m512i iacc_row_2_0 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_0, _mm512_shuffle_epi32(iacc_mat_11_0, 78));
                    __m512i iacc_row_3_0 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_10_0, 78), iacc_mat_11_0);
                    __m512i iacc_row_0_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_00_1, _mm512_shuffle_epi32(iacc_mat_01_1, 78));
                    __m512i iacc_row_1_1 = _mm512_mask_blend_epi32(0xCCCC, _mm512_shuffle_epi32(iacc_mat_00_1, 78), iacc_mat_01_1);
                    __m512i iacc_row_2_1 = _mm512_mask_blend_epi32(0xCCCC, iacc_mat_10_1, _mm512_shuffle_epi32(iacc_mat_11_1, 78));
                    __m512i iacc_row_3_1 = _mm512_mask_blend_epi32(0xCCCC,_mm512_shuffle_epi32(iacc_mat_10_1, 78), iacc_mat_11_1);

                    __m512i iacc_row_0 = _mm512_add_epi32(iacc_row_0_0, iacc_row_0_1);
                    __m512i iacc_row_1 = _mm512_add_epi32(iacc_row_1_0, iacc_row_1_1);
                    __m512i iacc_row_2 = _mm512_add_epi32(iacc_row_2_0, iacc_row_2_1);
                    __m512i iacc_row_3 = _mm512_add_epi32(iacc_row_3_0, iacc_row_3_1);

                    // Load the scale(d) values for all the 4 Q8_k blocks and repeat it across lanes
                    const __m128 row_scale_f32_sse = _mm_load_ps(a_ptr[b].d);
                    const __m256 row_scale_f32_ymm = _mm256_set_m128(row_scale_f32_sse, row_scale_f32_sse);
                    const __m512 row_scale_f32 = _mm512_insertf32x8(_mm512_castps256_ps512(row_scale_f32_ymm), row_scale_f32_ymm, 1);

                    // Multiply with appropriate scales and accumulate (for both d and dmin) below
                    acc_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_0), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_rows[0]);
                    acc_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_1), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_rows[1]);
                    acc_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_2), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_rows[2]);
                    acc_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_3), _mm512_mul_ps(col_scale_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_rows[3]);

                    __m512i iacc_row_min_0 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, 0), mins_01);
                    __m512i iacc_row_min_1 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, 85), mins_01);
                    __m512i iacc_row_min_2 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, 170), mins_01);
                    __m512i iacc_row_min_3 = _mm512_madd_epi16(_mm512_shuffle_epi32(lhs_bsums_hsum_0123_01, 255), mins_01);

                    acc_min_rows[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_0), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 0)), acc_min_rows[0]);
                    acc_min_rows[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_1), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 85)), acc_min_rows[1]);
                    acc_min_rows[2] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_2), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 170)), acc_min_rows[2]);
                    acc_min_rows[3] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(iacc_row_min_3), _mm512_mul_ps(col_dmin_f32, _mm512_shuffle_ps(row_scale_f32, row_scale_f32, 255)), acc_min_rows[3]);
                }
            }
            // Store accumulated values
            for (int i = 0; i < 4; i++) {
                _mm512_storeu_ps((float * )(s + ((y * 4 + i) * bs + x * 8)), _mm512_sub_ps(acc_rows[i], acc_min_rows[i]));
            }
        }
    }
