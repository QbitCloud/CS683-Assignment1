// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

// void matmul_simd(const float *A, const float *B, float *C, int M, int N, int
// K,
//                  int lda, int ldb, int ldc) {
//   for (int i = 0; i < M; ++i) {
//     const float *a = A + static_cast<long>(i) * lda;
//     for (int j = 0; j < N; ++j) {
//       const float *b = B + static_cast<long>(j) * ldb;
//       __m256 vec_acc = _mm256_setzero_ps();
//
//       int p = 0;
//
//       for (; p + 7 < K; p += 8) {
//         const __m256 va = _mm256_loadu_ps(a + p);
//         const __m256 vb = _mm256_loadu_ps(b + p);
//         vec_acc = _mm256_fmadd_ps(va, vb, vec_acc);
//       }
//
//       alignas(32) float partial_sums[8];
//       _mm256_store_ps(partial_sums, vec_acc);
//
//       float acc = 0.0f;
//
//       for (int lane = 0; lane < 8; ++lane) {
//         acc += partial_sums[lane];
//       }
//
//       for (; p < K; ++p) {
//         acc += a[p] * b[p];
//       }
//
//       C[static_cast<long>(i) * ldc + j] = acc;
//     }
//   }
// }

// Another version of SIMD implementation with multiple vector registers

void matmul_simd(const float *A, const float *B, float *C, int M, int N, int K,
                 int lda, int ldb, int ldc) {
  for (int i = 0; i < M; ++i) {
    const float *a = A + static_cast<long>(i) * lda;

    for (int j = 0; j < N; ++j) {
      const float *b = B + static_cast<long>(j) * ldb;
      __m256 vec_acc0 = _mm256_setzero_ps();
      __m256 vec_acc1 = _mm256_setzero_ps();
      __m256 vec_acc2 = _mm256_setzero_ps();
      __m256 vec_acc3 = _mm256_setzero_ps();

      int p = 0;
      for (; p + 31 < K; p += 32) {
        vec_acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p),
                                   _mm256_loadu_ps(b + p), vec_acc0);
        vec_acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 8),
                                   _mm256_loadu_ps(b + p + 8), vec_acc1);
        vec_acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 16),
                                   _mm256_loadu_ps(b + p + 16), vec_acc2);
        vec_acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 24),
                                   _mm256_loadu_ps(b + p + 24), vec_acc3);
      }

      // Finish any remaining complete 8-float vectors with one register.
      for (; p + 7 < K; p += 8) {
        vec_acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p),
                                   _mm256_loadu_ps(b + p), vec_acc0);
      }

      alignas(32) float partial_sums[8];
      const __m256 vec_sum01 = _mm256_add_ps(vec_acc0, vec_acc1);
      const __m256 vec_sum23 = _mm256_add_ps(vec_acc2, vec_acc3);
      _mm256_store_ps(partial_sums, _mm256_add_ps(vec_sum01, vec_sum23));

      float acc = 0.0f;
      for (int lane = 0; lane < 8; ++lane) {
        acc += partial_sums[lane];
      }
      for (; p < K; ++p) {
        acc += a[p] * b[p];
      }

      C[static_cast<long>(i) * ldc + j] = acc;
    }
  }
}
