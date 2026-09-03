// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>
#include <algoritham>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {

                        const int p = K / 2;
    const int in_stride = W + 2 * p; // padded row stride
    const int TILE = 128;
    // TODO(student): replace this placeholder with your best combined implementation.
    for (int i = 0; i < H * W; ++i)
{
    out[i] = 0.0f;
}

for (int ky = 0; ky < K; ++ky)
{
    for (int kx = 0; kx < K; ++kx)
    {
        for (int ty = 0; ty < H; ty += TILE)
        {
            for (int tx = 0; tx < W; tx += TILE)
            {
                for (int oy = ty; oy < std::min(ty + TILE, H); ++oy)
                {
                    for (int ox = tx; ox < std::min(tx + TILE, W); ox += 32)
                    {
                        out[oy * W + ox + 0] += in[(oy + ky) * in_stride + (ox + 0 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 1] += in[(oy + ky) * in_stride + (ox + 1 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 2] += in[(oy + ky) * in_stride + (ox + 2 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 3] += in[(oy + ky) * in_stride + (ox + 3 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 4] += in[(oy + ky) * in_stride + (ox + 4 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 5] += in[(oy + ky) * in_stride + (ox + 5 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 6] += in[(oy + ky) * in_stride + (ox + 6 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 7] += in[(oy + ky) * in_stride + (ox + 7 + kx)] * ker[ky * K + kx];

                        out[oy * W + ox + 8] += in[(oy + ky) * in_stride + (ox + 8 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 9] += in[(oy + ky) * in_stride + (ox + 9 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 10] += in[(oy + ky) * in_stride + (ox + 10 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 11] += in[(oy + ky) * in_stride + (ox + 11 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 12] += in[(oy + ky) * in_stride + (ox + 12 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 13] += in[(oy + ky) * in_stride + (ox + 13 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 14] += in[(oy + ky) * in_stride + (ox + 14 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 15] += in[(oy + ky) * in_stride + (ox + 15 + kx)] * ker[ky * K + kx];

                        out[oy * W + ox + 16] += in[(oy + ky) * in_stride + (ox + 16 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 17] += in[(oy + ky) * in_stride + (ox + 17 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 18] += in[(oy + ky) * in_stride + (ox + 18 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 19] += in[(oy + ky) * in_stride + (ox + 19 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 20] += in[(oy + ky) * in_stride + (ox + 20 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 21] += in[(oy + ky) * in_stride + (ox + 21 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 22] += in[(oy + ky) * in_stride + (ox + 22 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 23] += in[(oy + ky) * in_stride + (ox + 23 + kx)] * ker[ky * K + kx];

                        out[oy * W + ox + 24] += in[(oy + ky) * in_stride + (ox + 24 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 25] += in[(oy + ky) * in_stride + (ox + 25 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 26] += in[(oy + ky) * in_stride + (ox + 26 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 27] += in[(oy + ky) * in_stride + (ox + 27 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 28] += in[(oy + ky) * in_stride + (ox + 28 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 29] += in[(oy + ky) * in_stride + (ox + 29 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 30] += in[(oy + ky) * in_stride + (ox + 30 + kx)] * ker[ky * K + kx];
                        out[oy * W + ox + 31] += in[(oy + ky) * in_stride + (ox + 31 + kx)] * ker[ky * K + kx];
                    }
                }
            }
        }
    }
}
}
