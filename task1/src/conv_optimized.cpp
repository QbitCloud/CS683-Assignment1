// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <algorithm>
#include <immintrin.h>
#include "convolution.h"

// TILE + UNROLL + SIND version, setting TILE to W to disable tile
void conv_optimized(const float *in, float *out, const float *ker,
                    int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    // TILE PARAMETER
    const int TILE = W;

    for (int by = 0; by < H; by += TILE)
    {
        int y_end = std::min(by + TILE, H);

        for (int bx = 0; bx < W; bx += TILE)
        {
            int x_end = std::min(bx + TILE, W);

            int oy;

            // Expanding 3 rows
            for (oy = by; oy + 2 < y_end; oy += 3)
            {
                int ox;

                // Expanding 32 cols, to utilise every simd register
                for (ox = bx; ox + 31 < x_end; ox += 32)
                {
                    __m256 acc01 = _mm256_setzero_ps();
                    __m256 acc02 = _mm256_setzero_ps();
                    __m256 acc03 = _mm256_setzero_ps();
                    __m256 acc04 = _mm256_setzero_ps();

                    __m256 acc11 = _mm256_setzero_ps();
                    __m256 acc12 = _mm256_setzero_ps();
                    __m256 acc13 = _mm256_setzero_ps();
                    __m256 acc14 = _mm256_setzero_ps();

                    __m256 acc21 = _mm256_setzero_ps();
                    __m256 acc22 = _mm256_setzero_ps();
                    __m256 acc23 = _mm256_setzero_ps();
                    __m256 acc24 = _mm256_setzero_ps();

                    for (int ky = 0; ky < K; ++ky)
                    {
                        for (int kx = 0; kx < K; ++kx)
                        {
                            __m256 kernel = _mm256_broadcast_ss(&ker[ky * K + kx]);

                            // Row 0
                            acc01 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + 0 + kx]),
                                kernel, acc01);

                            acc02 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + 8 + kx]),
                                kernel, acc02);

                            acc03 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + 16 + kx]),
                                kernel, acc03);

                            acc04 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + ky) * in_stride + ox + 24 + kx]),
                                kernel, acc04);

                            // Row 1
                            acc11 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + 1 + ky) * in_stride + ox + 0 + kx]),
                                kernel, acc11);

                            acc12 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + 1 + ky) * in_stride + ox + 8 + kx]),
                                kernel, acc12);

                            acc13 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + 1 + ky) * in_stride + ox + 16 + kx]),
                                kernel, acc13);

                            acc14 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + 1 + ky) * in_stride + ox + 24 + kx]),
                                kernel, acc14);

                            // Row 2
                            acc21 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + 2 + ky) * in_stride + ox + 0 + kx]),
                                kernel, acc21);

                            acc22 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + 2 + ky) * in_stride + ox + 8 + kx]),
                                kernel, acc22);

                            acc23 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + 2 + ky) * in_stride + ox + 16 + kx]),
                                kernel, acc23);

                            acc24 = _mm256_fmadd_ps(
                                _mm256_loadu_ps(&in[(oy + 2 + ky) * in_stride + ox + 24 + kx]),
                                kernel, acc24);
                        }
                    }

                    _mm256_storeu_ps(&out[oy * W + ox + 0], acc01);
                    _mm256_storeu_ps(&out[oy * W + ox + 8], acc02);
                    _mm256_storeu_ps(&out[oy * W + ox + 16], acc03);
                    _mm256_storeu_ps(&out[oy * W + ox + 24], acc04);

                    _mm256_storeu_ps(&out[(oy + 1) * W + ox + 0], acc11);
                    _mm256_storeu_ps(&out[(oy + 1) * W + ox + 8], acc12);
                    _mm256_storeu_ps(&out[(oy + 1) * W + ox + 16], acc13);
                    _mm256_storeu_ps(&out[(oy + 1) * W + ox + 24], acc14);

                    _mm256_storeu_ps(&out[(oy + 2) * W + ox + 0], acc21);
                    _mm256_storeu_ps(&out[(oy + 2) * W + ox + 8], acc22);
                    _mm256_storeu_ps(&out[(oy + 2) * W + ox + 16], acc23);
                    _mm256_storeu_ps(&out[(oy + 2) * W + ox + 24], acc24);
                }

                // Remaining columns of this tile
                for (; ox < x_end; ++ox)
                {
                    for (int r = 0; r < 3; ++r)
                    {
                        float sum = 0.0f;

                        for (int ky = 0; ky < K; ++ky)
                            for (int kx = 0; kx < K; ++kx)
                                sum += in[(oy + r + ky) * in_stride + ox + kx] *
                                       ker[ky * K + kx];

                        out[(oy + r) * W + ox] = sum;
                    }
                }
            }

            // Remaining rows of this tile
            for (; oy < y_end; ++oy)
            {
                for (int ox = bx; ox < x_end; ++ox)
                {
                    float sum = 0.0f;

                    for (int ky = 0; ky < K; ++ky)
                        for (int kx = 0; kx < K; ++kx)
                            sum += in[(oy + ky) * in_stride + ox + kx] *
                                   ker[ky * K + kx];

                    out[oy * W + ox] = sum;
                }
            }
        }
    }
}

// Reordering + Unrolling
// void conv_optimized(const float *in, float *out, const float *ker, int H, int W,
//                     int K)
// {
//     const int p = K / 2;
//     const int in_stride = W + 2 * p;

//     std::memset(out, 0, sizeof(float) * H * W);

//     for (int ky = 0; ky < K; ++ky)
//     {
//         for (int kx = 0; kx < K; ++kx)
//         {
//             int ox = 0;
//             const float kval = ker[ky * K + kx];
//             for (int oy = 0; oy < H; ++oy)
//             {
//                 const float *in_row = in + (oy + ky) * in_stride + kx;
//                 float *out_row = out + oy * W;
//                 for (ox = 0; ox + 7 < W; ox += 8)
//                 {
//                     out_row[ox] += in_row[ox] * kval;
//                     out_row[ox + 1] += in_row[ox + 1] * kval;
//                     out_row[ox + 2] += in_row[ox + 2] * kval;
//                     out_row[ox + 3] += in_row[ox + 3] * kval;
//                     out_row[ox + 4] += in_row[ox + 4] * kval;
//                     out_row[ox + 5] += in_row[ox + 5] * kval;
//                     out_row[ox + 6] += in_row[ox + 6] * kval;
//                     out_row[ox + 7] += in_row[ox + 7] * kval;
//                 }
//                 for (; ox < W; ox++)
//                 {
//                     out_row[ox] += in_row[ox] * kval;
//                 }
//             }
//         }
//     }
// }

// Tiling + Unrolling
// void conv_optimized(const float *in, float *out, const float *ker,
//                     int H, int W, int K)

// {
//     const int p = K / 2;
//     const int in_stride = W + 2 * p;

//     const int TILE_H = 512;
//     const int TILE_W = 512;

//     for (int oy_tile = 0; oy_tile < H; oy_tile += TILE_H)
//     {
//         const int oy_end = std::min(oy_tile + TILE_H, H);

//         for (int ox_tile = 0; ox_tile < W; ox_tile += TILE_W)
//         {
//             const int ox_end = std::min(ox_tile + TILE_W, W);

//             for (int oy = oy_tile; oy < oy_end; ++oy)
//             {
//                 float *out_row = out + oy * W;
//                 int ox = ox_tile;

//                 for (int ox = ox_tile; ox < ox_end; ox += 8)
//                 {
//                     float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
//                     float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;

//                     for (int ky = 0; ky < K; ++ky)
//                     {
//                         const float *in_row = in + (oy + ky) * in_stride + ox;
//                         const float *krow = ker + ky * K;

//                         for (int kx = 0; kx < K; ++kx)
//                         {
//                             const float kval = krow[kx];

//                             acc0 += in_row[kx + 0] * kval;
//                             acc1 += in_row[kx + 1] * kval;
//                             acc2 += in_row[kx + 2] * kval;
//                             acc3 += in_row[kx + 3] * kval;
//                             acc4 += in_row[kx + 4] * kval;
//                             acc5 += in_row[kx + 5] * kval;
//                             acc6 += in_row[kx + 6] * kval;
//                             acc7 += in_row[kx + 7] * kval;
//                         }
//                     }

//                     out_row[ox + 0] = acc0;
//                     out_row[ox + 1] = acc1;
//                     out_row[ox + 2] = acc2;
//                     out_row[ox + 3] = acc3;
//                     out_row[ox + 4] = acc4;
//                     out_row[ox + 5] = acc5;
//                     out_row[ox + 6] = acc6;
//                     out_row[ox + 7] = acc7;
//                 }
//             }
//         }
//     }
// }
