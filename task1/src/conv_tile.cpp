// conv_tile.cpp  STAGE 3: CACHE TILING
#include <algorithm>
#include <cstring>

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int CONV_TILE_H=3;
    const int CONV_TILE_W=512;
    std::memset(out, 0, sizeof(float) * static_cast<std::size_t>(H) * W);


    for (int oy_tile = 0; oy_tile < H; oy_tile += CONV_TILE_H) {
        const int oy_end = std::min(oy_tile + CONV_TILE_H, H);

        for (int ox_tile = 0; ox_tile < W; ox_tile += CONV_TILE_W) {
            const int ox_end = std::min(ox_tile + CONV_TILE_W, W);

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float kval = ker[ky * K + kx];  

                    for (int oy = oy_tile; oy < oy_end; ++oy) {
                        const float* in_row = in + (oy + ky) * in_stride + kx;
                        float* out_row = out + oy * W;

                        for (int ox = ox_tile; ox < ox_end; ++ox) {
                            out_row[ox] += in_row[ox] * kval; 
                        }
                    }
                }
            }
        }
    }
}






}
