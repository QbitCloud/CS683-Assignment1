// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

static inline float horizontal_sum(__m256 v) { //returns v[0]+v[1]+v[2]+v[3]....v[7]
    __m128 low = _mm256_castps256_ps128(v);
    __m128 high = _mm256_extractf128_ps(v, 1);

    __m128 sum = _mm_add_ps(low, high); // sum = lower128+upper128 of 256 avx reg
    sum = _mm_hadd_ps(sum, sum);//sum[0&2]=sum[0]+sum[1],sum[1&3]=sum[2]+sum[3], here lower sum = upper sum
    sum = _mm_hadd_ps(sum, sum);

    return _mm_cvtss_f32(sum); //converts sum[0] to float and returns
}

void matmul_simd(const float *A,const float *B,float *C,int M,int N,int K,int lda,int ldb,int ldc) {
    for (int i = 0; i < M; ++i) {
        const float *a = A + static_cast<long>(i) * lda;
        float *c = C + static_cast<long>(i) * ldc;

        int j = 0;
        for (; j + 3 < N; j += 4) {
            const float *b0 = B + static_cast<long>(j + 0) * ldb;
            const float *b1 = B + static_cast<long>(j + 1) * ldb;
            const float *b2 = B + static_cast<long>(j + 2) * ldb;
            const float *b3 = B + static_cast<long>(j + 3) * ldb;

            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();

            int p = 0;

            for (; p + 7 < K; p += 8) { // 8 values of A are fmadded with 8*4 (4 logical col/program me rows of B) in 1 iteration
                __m256 va = _mm256_loadu_ps(a + p); 

                __m256 vb0 = _mm256_loadu_ps(b0 + p);
                __m256 vb1 = _mm256_loadu_ps(b1 + p);
                __m256 vb2 = _mm256_loadu_ps(b2 + p);
                __m256 vb3 = _mm256_loadu_ps(b3 + p);

                acc0 = _mm256_fmadd_ps(va, vb0, acc0);
                acc1 = _mm256_fmadd_ps(va, vb1, acc1);
                acc2 = _mm256_fmadd_ps(va, vb2, acc2);
                acc3 = _mm256_fmadd_ps(va, vb3, acc3);
            }

            float sum0 = horizontal_sum(acc0);
            float sum1 = horizontal_sum(acc1);
            float sum2 = horizontal_sum(acc2);
            float sum3 = horizontal_sum(acc3);

            for (; p < K; ++p) {  //for k(row length) not divisible by 8
                sum0 += a[p] * b0[p];
                sum1 += a[p] * b1[p];
                sum2 += a[p] * b2[p];
                sum3 += a[p] * b3[p];
            }

            c[j + 0] = sum0;
            c[j + 1] = sum1;
            c[j + 2] = sum2;
            c[j + 3] = sum3;
        }

        /*for (; j < N; ++j) {  //handling values not divisible by 8 in the j iterated loop
            const float *b = B + static_cast<long>(j) * ldb;

            __m256 acc = _mm256_setzero_ps();

            int p = 0;

            for (; p + 7 < K; p += 8) {
                __m256 va = _mm256_loadu_ps(a + p);
                __m256 vb = _mm256_loadu_ps(b + p);

                acc = _mm256_fmadd_ps(va, vb, acc);
            }

            float sum = horizontal_sum(acc);

            for (; p < K; ++p) {
                sum += a[p] * b[p];
            }

            c[j] = sum;
        }*/
    }
}