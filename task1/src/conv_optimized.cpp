// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include "convolution.h"

void conv_optimized(const float *in, float *out, const float *ker,
                    int H, int W, int K)
{
    conv_naive(in, out, ker, H, W, K);
}
