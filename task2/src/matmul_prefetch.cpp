// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

static inline void prefetch_matmul_data(const float *a, const float *b, int p,
                                        int K) {
  int prefetch_distance = 32;
  int prefetch_degree = 1;
  // Distance is expressed in SIMD iterations (8 floats each).
  const int pf = p + prefetch_distance * 8;

  if (pf < K) {
    // Each degree advances by one 64-byte cache line = 16 floats.
    for (int d = 1; d <= prefetch_degree; ++d) {
      const int offset = pf + d * 16;

      if (offset < K) {
        _mm_prefetch(reinterpret_cast<const char *>(a + offset), _MM_HINT_T1);
        _mm_prefetch(reinterpret_cast<const char *>(b + offset), _MM_HINT_T1);
      }
    }
  }
}

void matmul_prefetch(const float *A, const float *B, float *C, int M, int N,
                     int K, int lda, int ldb, int ldc) {
  // TODO(student): replace this placeholder with your cache-blocked SIMD +
  // prefetch implementation.
  //

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
        // Prefetch future data before processing the current block.
        prefetch_matmul_data(a, b, p, K);

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
        prefetch_matmul_data(a, b, p, K);

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
