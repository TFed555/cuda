#ifndef KERNEL_VECRED_NOBR
#define KERNEL_VECRED_NOBR

#include "kinds.h"
#include "../vector_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_vecred_nobr(VectorView<AtomT> a, AtomT res) {
    const int BLOCK_SIZE = 16;
    __shared__ AtomT sh[BLOCK_SIZE];

    
}

#endif