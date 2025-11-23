#ifndef KERNEL_VECRED_NOBR
#define KERNEL_VECRED_NOBR

#include "kinds.h"
#include "../vector_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_vecred_nobr(VectorView<AtomT> a, AtomT res) {
    const int BLOCK_SIZE = 16;
    __shared__ AtomT sh[BLOCK_SIZE];
    int t = blockIdx.x;

    for (size_t i = 0; i <= (a.size() + BLOCK_SIZE - 1) / BLOCK_SIZE; i++) {
        sh[t] = a[i] + a[i+1];
        __syncthreads();
    }

    res += sh;
    __syncthreads();
}

#endif