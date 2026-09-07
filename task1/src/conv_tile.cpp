// conv_tile.cpp  STAGE 3: CACHE TILING

#include <algorithm>
#include "convolution.h"

void conv_tile(const float *in, float *out, const float *ker,
               int H, int W, int K)
{
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    const int TILE_H = 128;
    const int TILE_W = 128;

    for (int oy_tile = 0; oy_tile < H; oy_tile += TILE_H)
    {
        const int oy_end = std::min(oy_tile + TILE_H, H);

        for (int ox_tile = 0; ox_tile < W; ox_tile += TILE_W)
        {
            const int ox_end = std::min(ox_tile + TILE_W, W);

            for (int oy = oy_tile; oy < oy_end; ++oy)
            {
                for (int ox = ox_tile; ox < ox_end; ++ox)
                {
                    float acc = 0.0f;
                    for (int ky = 0; ky < K; ++ky)
                    {
                        for (int kx = 0; kx < K; ++kx)
                        {
                            acc += in[(oy + ky) * in_stride + (ox + kx)] *
                                   ker[ky * K + kx];
                        }
                    }
                    out[oy * W + ox] = acc;
                }
            }
        }
    }
}