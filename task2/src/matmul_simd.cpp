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

// void matmul_simd(const float *A, const float *B, float *C, int M, int N, int
// K,
//                  int lda, int ldb, int ldc) {
//   for (int i = 0; i < M; ++i) {
//     const float *a = A + static_cast<long>(i) * lda;
//
//     for (int j = 0; j < N; ++j) {
//       const float *b = B + static_cast<long>(j) * ldb;
//       __m256 vec_acc0 = _mm256_setzero_ps();
//       __m256 vec_acc1 = _mm256_setzero_ps();
//       __m256 vec_acc2 = _mm256_setzero_ps();
//       __m256 vec_acc3 = _mm256_setzero_ps();
//
//       int p = 0;
//       for (; p + 31 < K; p += 32) {
//         vec_acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p),
//                                    _mm256_loadu_ps(b + p), vec_acc0);
//         vec_acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 8),
//                                    _mm256_loadu_ps(b + p + 8), vec_acc1);
//         vec_acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 16),
//                                    _mm256_loadu_ps(b + p + 16), vec_acc2);
//         vec_acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 24),
//                                    _mm256_loadu_ps(b + p + 24), vec_acc3);
//       }
//
//       // Finish any remaining complete 8-float vectors with one register.
//       for (; p + 7 < K; p += 8) {
//         vec_acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p),
//                                    _mm256_loadu_ps(b + p), vec_acc0);
//       }
//
//       alignas(32) float partial_sums[8];
//       const __m256 vec_sum01 = _mm256_add_ps(vec_acc0, vec_acc1);
//       const __m256 vec_sum23 = _mm256_add_ps(vec_acc2, vec_acc3);
//       _mm256_store_ps(partial_sums, _mm256_add_ps(vec_sum01, vec_sum23));
//
//       float acc = 0.0f;
//       for (int lane = 0; lane < 8; ++lane) {
//         acc += partial_sums[lane];
//       }
//       for (; p < K; ++p) {
//         acc += a[p] * b[p];
//       }
//
//       C[static_cast<long>(i) * ldc + j] = acc;
//     }
//   }
// }
//

//
//
//
//
//
//
//
// More optimized version of SIMD implementation.

inline float dot_single(const float *a, const float *b, int K) {
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  int p = 0;
  for (; p + 31 < K; p += 32) {
    acc0 =
        _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 8),
                           _mm256_loadu_ps(b + p + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 16),
                           _mm256_loadu_ps(b + p + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 24),
                           _mm256_loadu_ps(b + p + 24), acc3);
  }

  for (; p + 7 < K; p += 8) {
    acc0 =
        _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
  }

  alignas(32) float partial[8];

  _mm256_store_ps(partial, _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                         _mm256_add_ps(acc2, acc3)));

  float acc = 0.0f;

  for (int lane = 0; lane < 8; lane++) {
    acc += partial[lane];
  }

  for (; p < K; p++) {
    acc += a[p] * b[p];
  }

  return acc;
}

void matmul_simd(const float *A, const float *B, float *C, int M, int N, int K,
                 int lda, int ldb, int ldc) {
  for (int i = 0; i < M; i++) {
    const float *a = A + static_cast<long>(i) * lda;
    int j = 0;

    for (; j + 3 < N; j += 4) {
      const float *b0 = B + static_cast<long>(j + 0) * ldb;
      const float *b1 = B + static_cast<long>(j + 1) * ldb;
      const float *b2 = B + static_cast<long>(j + 2) * ldb;
      const float *b3 = B + static_cast<long>(j + 3) * ldb;

      __m256 acc00 = _mm256_setzero_ps();
      __m256 acc01 = _mm256_setzero_ps();
      __m256 acc10 = _mm256_setzero_ps();
      __m256 acc11 = _mm256_setzero_ps();
      __m256 acc20 = _mm256_setzero_ps();
      __m256 acc21 = _mm256_setzero_ps();
      __m256 acc30 = _mm256_setzero_ps();
      __m256 acc31 = _mm256_setzero_ps();

      int p = 0;

      for (; p + 15 < K; p += 16) {
        const __m256 va0 = _mm256_loadu_ps(a + p);
        const __m256 va1 = _mm256_loadu_ps(a + p + 8);

        acc00 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + p), acc00);
        acc01 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b0 + p + 8), acc01);

        acc10 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b1 + p), acc10);
        acc11 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b1 + p + 8), acc11);

        acc20 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b2 + p), acc20);
        acc21 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b2 + p + 8), acc21);

        acc30 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b3 + p), acc30);
        acc31 = _mm256_fmadd_ps(va1, _mm256_loadu_ps(b3 + p + 8), acc31);
      }

      for (; p + 7 < K; p += 8) {
        const __m256 va0 = _mm256_loadu_ps(a + p);
        acc00 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + p), acc00);
        acc10 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b1 + p), acc10);
        acc20 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b2 + p), acc20);
        acc30 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b3 + p), acc30);
      }

      alignas(32) float partial[8];
      float sum[4];

      _mm256_store_ps(partial, _mm256_add_ps(acc00, acc01));
      sum[0] = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] +
               partial[5] + partial[6] + partial[7];
      _mm256_store_ps(partial, _mm256_add_ps(acc10, acc11));
      sum[1] = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] +
               partial[5] + partial[6] + partial[7];
      _mm256_store_ps(partial, _mm256_add_ps(acc20, acc21));
      sum[2] = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] +
               partial[5] + partial[6] + partial[7];
      _mm256_store_ps(partial, _mm256_add_ps(acc30, acc31));
      sum[3] = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] +
               partial[5] + partial[6] + partial[7];

      for (int pp = p; pp < K; pp++) {
        const float av = a[pp];
        sum[0] += av * b0[pp];
        sum[1] += av * b1[pp];
        sum[2] += av * b2[pp];
        sum[3] += av * b3[pp];
      }

      C[static_cast<long>(i) * ldc + j + 0] = sum[0];
      C[static_cast<long>(i) * ldc + j + 1] = sum[1];
      C[static_cast<long>(i) * ldc + j + 2] = sum[2];
      C[static_cast<long>(i) * ldc + j + 3] = sum[3];

      for (; j < N; j++) {
        const float *b = B + static_cast<long>(j) * ldb;
        C[static_cast<long>(i) * ldc + j] = dot_single(a, b, K);
      }
    }
  }
}
