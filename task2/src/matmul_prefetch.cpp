// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include "matmul.h"
#include <immintrin.h>

// Defining prefetch parameters
#define PREFETCH_DISTANCE 32
#define PREFETCH_DEGREE 1
#define PREFETCH_HINT _MM_HINT_T0

inline float product_1xcol(const float *a, const float *b, int K)
{
  __m256 acc0 = _mm256_setzero_ps();
  __m256 acc1 = _mm256_setzero_ps();
  __m256 acc2 = _mm256_setzero_ps();
  __m256 acc3 = _mm256_setzero_ps();

  int p = 0;

  for (; p + 31 < K; p += 32)
  {
    for (int d = 0; d < PREFETCH_DEGREE; d++)
    {
      const int off = PREFETCH_DISTANCE + d * 16;
      _mm_prefetch((const char *)(a + p + off), PREFETCH_HINT);
      _mm_prefetch((const char *)(b + p + off), PREFETCH_HINT);
    }

    acc0 =
        _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 8),
                           _mm256_loadu_ps(b + p + 8), acc1);
    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 16),
                           _mm256_loadu_ps(b + p + 16), acc2);
    acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(a + p + 24),
                           _mm256_loadu_ps(b + p + 24), acc3);
  }

  for (; p + 7 < K; p += 8)
  {
    acc0 =
        _mm256_fmadd_ps(_mm256_loadu_ps(a + p), _mm256_loadu_ps(b + p), acc0);
  }

  alignas(32) float partial[8];

  _mm256_store_ps(partial, _mm256_add_ps(_mm256_add_ps(acc0, acc1),
                                         _mm256_add_ps(acc2, acc3)));

  float acc = 0.0f;

  for (int lane = 0; lane < 8; lane++)
  {
    acc += partial[lane];
  }

  for (; p < K; p++)
  {
    acc += a[p] * b[p];
  }
  return acc;
}

inline void product_4xcol(const float *a, const float *b0, const float *b1,
                          const float *b2, const float *b3, int K, float *out4)
{
  __m256 acc00 = _mm256_setzero_ps();
  __m256 acc01 = _mm256_setzero_ps();
  __m256 acc10 = _mm256_setzero_ps();
  __m256 acc11 = _mm256_setzero_ps();
  __m256 acc20 = _mm256_setzero_ps();
  __m256 acc21 = _mm256_setzero_ps();
  __m256 acc30 = _mm256_setzero_ps();
  __m256 acc31 = _mm256_setzero_ps();

  int p = 0;
  for (; p + 15 < K; p += 16)
  {
    for (int d = 0; d < PREFETCH_DEGREE; d++)
    {
      const int off = PREFETCH_DISTANCE + d * 16;
      _mm_prefetch((const char *)(a + p + off), PREFETCH_HINT);
      _mm_prefetch((const char *)(b0 + p + off), PREFETCH_HINT);
      _mm_prefetch((const char *)(b1 + p + off), PREFETCH_HINT);
      _mm_prefetch((const char *)(b2 + p + off), PREFETCH_HINT);
      _mm_prefetch((const char *)(b3 + p + off), PREFETCH_HINT);
    }

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

  for (; p + 7 < K; p += 8)
  {
    const __m256 va0 = _mm256_loadu_ps(a + p);
    acc00 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b0 + p), acc00);
    acc10 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b1 + p), acc10);
    acc20 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b2 + p), acc20);
    acc30 = _mm256_fmadd_ps(va0, _mm256_loadu_ps(b3 + p), acc30);
  }

  alignas(32) float partial[8];
  _mm256_store_ps(partial, _mm256_add_ps(acc00, acc01));
  out4[0] = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] +
            partial[5] + partial[6] + partial[7];
  _mm256_store_ps(partial, _mm256_add_ps(acc10, acc11));
  out4[1] = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] +
            partial[5] + partial[6] + partial[7];
  _mm256_store_ps(partial, _mm256_add_ps(acc20, acc21));
  out4[2] = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] +
            partial[5] + partial[6] + partial[7];
  _mm256_store_ps(partial, _mm256_add_ps(acc30, acc31));
  out4[3] = partial[0] + partial[1] + partial[2] + partial[3] + partial[4] +
            partial[5] + partial[6] + partial[7];

  for (int pp = p; pp < K; pp++)
  {
    const float av = a[pp];
    out4[0] += av * b0[pp];
    out4[1] += av * b1[pp];
    out4[2] += av * b2[pp];
    out4[3] += av * b3[pp];
  }
}

void matmul_prefetch(const float *A, const float *B, float *C, int M, int N,
                     int K, int lda, int ldb, int ldc)
{
  // Tiling parameter, applied on width
  constexpr int TILE_WIDTH = 128;

  for (int jj = 0; jj < N; jj += TILE_WIDTH)
  {
    const int j_end = (jj + TILE_WIDTH < N) ? jj + TILE_WIDTH : N;

    for (int i = 0; i < M; i++)
    {
      const float *a = A + static_cast<long>(i) * lda;

      int j = jj;

      for (; j + 3 < j_end; j += 4)
      {
        const float *b0 = B + static_cast<long>(j + 0) * ldb;
        const float *b1 = B + static_cast<long>(j + 1) * ldb;
        const float *b2 = B + static_cast<long>(j + 2) * ldb;
        const float *b3 = B + static_cast<long>(j + 3) * ldb;

        float out4[4] = {};
        product_4xcol(a, b0, b1, b2, b3, K, out4);

        C[static_cast<long>(i) * ldc + j + 0] = out4[0];
        C[static_cast<long>(i) * ldc + j + 1] = out4[1];
        C[static_cast<long>(i) * ldc + j + 2] = out4[2];
        C[static_cast<long>(i) * ldc + j + 3] = out4[3];
      }

      for (; j < j_end; j++)
      {
        const float *b = B + static_cast<long>(j) * ldb;
        C[static_cast<long>(i) * ldc + j] = product_1xcol(a, b, K);
      }
    }
  }
}
