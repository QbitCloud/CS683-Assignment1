// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp.
// Combine everything you have learned across the whole assignment  loop
// reordering, register blocking and unrolling (Task 1 / Stage 1 here), cache
// tiling and software prefetch (Stage 2)  and TUNE it to be as fast as you can.
// Your speedup over matmul_naive determines your score (see the tier table the
// harness prints), and this same function will power a real LLM inference via
// `make llama-demo`.

#include <immintrin.h>
#include <algorithm>
#include "matmul.h"

#define PREFETCH_DISTANCE 128
#define PREFETCH_DEGREE 1
#define PREFETCH_HINT _MM_HINT_T0

inline float simd_reg_to_val(__m256 reg)
{
  __m128 lo = _mm256_castps256_ps128(reg);
  __m128 hi = _mm256_extractf128_ps(reg, 1);

  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);

  return _mm_cvtss_f32(sum);
}

inline float product_1xcol(const float *a, const float *b, int K)
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  int p = 0;
  for (; p + 31 < K; p += 32)
  {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 8), _mm256_loadu_ps(b + p + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 16), _mm256_loadu_ps(b + p + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 24), _mm256_loadu_ps(b + p + 24), acc3);
  }

  for (; p + 7 < K; p += 8)
  {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
  }

  float acc = simd_reg_to_val(
      _mm256_add_ps(
          _mm256_add_ps(acc0, acc1),
          _mm256_add_ps(acc2, acc3)));

  for (; p < K; p++)
  {
    acc += a[p] * b[p];
  }

  return acc;
}

void matmul_optimized(const float *A, const float *B, float *C, int M, int N, int K,
                      int lda, int ldb, int ldc)
{
  constexpr int TILE_HEIGHT = 128;
  constexpr int TILE_WIDTH = 128;

  for (int ii = 0; ii < M; ii += TILE_HEIGHT)
  {
    int i_end = std::min(ii + TILE_HEIGHT, M);

    for (int jj = 0; jj < N; jj += TILE_WIDTH)
    {
      int j_end = std::min(jj + TILE_WIDTH, N);

      for (int i = ii; i < i_end;)
      {
        if (i + 2 < i_end)
        {
          // Unrolling 3 rows
          const float *a0 = A + static_cast<long>(i + 0) * lda;
          const float *a1 = A + static_cast<long>(i + 1) * lda;
          const float *a2 = A + static_cast<long>(i + 2) * lda;

          int j = jj;
          for (; j + 3 < j_end; j += 4)
          {
            const float *b0 = B + static_cast<long>(j + 0) * ldb;
            const float *b1 = B + static_cast<long>(j + 1) * ldb;
            const float *b2 = B + static_cast<long>(j + 2) * ldb;
            const float *b3 = B + static_cast<long>(j + 3) * ldb;

            // 12 Accumulators
            __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps(), acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();
            __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps(), acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();
            __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps(), acc22 = _mm256_setzero_ps(), acc23 = _mm256_setzero_ps();

            int p = 0;
            // Using 16/16 simd registers
            for (; p + 15 < K; p += 16)
            {

              for (int d = 0; d < PREFETCH_DEGREE; d++)
              {
                const int off = PREFETCH_DISTANCE + d * 16;
                _mm_prefetch((const char *)(a0 + p + off), PREFETCH_HINT);
                _mm_prefetch((const char *)(b0 + p + off), PREFETCH_HINT);
                _mm_prefetch((const char *)(a1 + p + off), PREFETCH_HINT);
                _mm_prefetch((const char *)(b1 + p + off), PREFETCH_HINT);
                _mm_prefetch((const char *)(a2 + p + off), PREFETCH_HINT);
                _mm_prefetch((const char *)(b2 + p + off), PREFETCH_HINT);
                _mm_prefetch((const char *)(b3 + p + off), PREFETCH_HINT);
              }

              // First 8 floats
              __m256 va0 = _mm256_loadu_ps(a0 + p);
              __m256 va1 = _mm256_loadu_ps(a1 + p);
              __m256 va2 = _mm256_loadu_ps(a2 + p);

              __m256 vb = _mm256_loadu_ps(b0 + p);
              acc00 = _mm256_fmadd_ps(va0, vb, acc00);
              acc10 = _mm256_fmadd_ps(va1, vb, acc10);
              acc20 = _mm256_fmadd_ps(va2, vb, acc20);

              vb = _mm256_loadu_ps(b1 + p);
              acc01 = _mm256_fmadd_ps(va0, vb, acc01);
              acc11 = _mm256_fmadd_ps(va1, vb, acc11);
              acc21 = _mm256_fmadd_ps(va2, vb, acc21);

              vb = _mm256_loadu_ps(b2 + p);
              acc02 = _mm256_fmadd_ps(va0, vb, acc02);
              acc12 = _mm256_fmadd_ps(va1, vb, acc12);
              acc22 = _mm256_fmadd_ps(va2, vb, acc22);

              vb = _mm256_loadu_ps(b3 + p);
              acc03 = _mm256_fmadd_ps(va0, vb, acc03);
              acc13 = _mm256_fmadd_ps(va1, vb, acc13);
              acc23 = _mm256_fmadd_ps(va2, vb, acc23);

              // Next 8 floats
              va0 = _mm256_loadu_ps(a0 + p + 8);
              va1 = _mm256_loadu_ps(a1 + p + 8);
              va2 = _mm256_loadu_ps(a2 + p + 8);

              vb = _mm256_loadu_ps(b0 + p + 8);
              acc00 = _mm256_fmadd_ps(va0, vb, acc00);
              acc10 = _mm256_fmadd_ps(va1, vb, acc10);
              acc20 = _mm256_fmadd_ps(va2, vb, acc20);

              vb = _mm256_loadu_ps(b1 + p + 8);
              acc01 = _mm256_fmadd_ps(va0, vb, acc01);
              acc11 = _mm256_fmadd_ps(va1, vb, acc11);
              acc21 = _mm256_fmadd_ps(va2, vb, acc21);

              vb = _mm256_loadu_ps(b2 + p + 8);
              acc02 = _mm256_fmadd_ps(va0, vb, acc02);
              acc12 = _mm256_fmadd_ps(va1, vb, acc12);
              acc22 = _mm256_fmadd_ps(va2, vb, acc22);

              vb = _mm256_loadu_ps(b3 + p + 8);
              acc03 = _mm256_fmadd_ps(va0, vb, acc03);
              acc13 = _mm256_fmadd_ps(va1, vb, acc13);
              acc23 = _mm256_fmadd_ps(va2, vb, acc23);
            }

            for (; p + 7 < K; p += 8)
            {
              const __m256 va0 = _mm256_loadu_ps(a0 + p);
              const __m256 va1 = _mm256_loadu_ps(a1 + p);
              const __m256 va2 = _mm256_loadu_ps(a2 + p);

              acc00 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + p), acc00);
              acc01 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b1 + p), acc01);
              acc02 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b2 + p), acc02);
              acc03 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b3 + p), acc03);

              acc10 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b0 + p), acc10);
              acc11 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + p), acc11);
              acc12 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b2 + p), acc12);
              acc13 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b3 + p), acc13);

              acc20 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b0 + p), acc20);
              acc21 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b1 + p), acc21);
              acc22 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b2 + p), acc22);
              acc23 = _mm256_fmadd_ps(va2, _mm256_loadu_ps(b3 + p), acc23);
            }

            float sum00 = simd_reg_to_val(acc00), sum01 = simd_reg_to_val(acc01), sum02 = simd_reg_to_val(acc02), sum03 = simd_reg_to_val(acc03);
            float sum10 = simd_reg_to_val(acc10), sum11 = simd_reg_to_val(acc11), sum12 = simd_reg_to_val(acc12), sum13 = simd_reg_to_val(acc13);
            float sum20 = simd_reg_to_val(acc20), sum21 = simd_reg_to_val(acc21), sum22 = simd_reg_to_val(acc22), sum23 = simd_reg_to_val(acc23);

            for (int pp = p; pp < K; pp++)
            {
              const float v_a0 = a0[pp], v_a1 = a1[pp], v_a2 = a2[pp];
              sum00 += v_a0 * b0[pp];
              sum01 += v_a0 * b1[pp];
              sum02 += v_a0 * b2[pp];
              sum03 += v_a0 * b3[pp];
              sum10 += v_a1 * b0[pp];
              sum11 += v_a1 * b1[pp];
              sum12 += v_a1 * b2[pp];
              sum13 += v_a1 * b3[pp];
              sum20 += v_a2 * b0[pp];
              sum21 += v_a2 * b1[pp];
              sum22 += v_a2 * b2[pp];
              sum23 += v_a2 * b3[pp];
            }

            C[static_cast<long>(i + 0) * ldc + j + 0] = sum00;
            C[static_cast<long>(i + 0) * ldc + j + 1] = sum01;
            C[static_cast<long>(i + 0) * ldc + j + 2] = sum02;
            C[static_cast<long>(i + 0) * ldc + j + 3] = sum03;

            C[static_cast<long>(i + 1) * ldc + j + 0] = sum10;
            C[static_cast<long>(i + 1) * ldc + j + 1] = sum11;
            C[static_cast<long>(i + 1) * ldc + j + 2] = sum12;
            C[static_cast<long>(i + 1) * ldc + j + 3] = sum13;

            C[static_cast<long>(i + 2) * ldc + j + 0] = sum20;
            C[static_cast<long>(i + 2) * ldc + j + 1] = sum21;
            C[static_cast<long>(i + 2) * ldc + j + 2] = sum22;
            C[static_cast<long>(i + 2) * ldc + j + 3] = sum23;
          }

          for (; j < j_end; j++)
          {
            const float *b = B + static_cast<long>(j) * ldb;
            C[static_cast<long>(i + 0) * ldc + j] = product_1xcol(a0, b, K);
            C[static_cast<long>(i + 1) * ldc + j] = product_1xcol(a1, b, K);
            C[static_cast<long>(i + 2) * ldc + j] = product_1xcol(a2, b, K);
          }
          i += 3;
        }
        else
        {
          const float *a = A + static_cast<long>(i) * lda;
          int j = jj;
          for (; j < j_end; j++)
          {
            const float *b = B + static_cast<long>(j) * ldb;
            C[static_cast<long>(i) * ldc + j] = product_1xcol(a, b, K);
          }
          i += 1;
        }
      }
    }
  }
}
