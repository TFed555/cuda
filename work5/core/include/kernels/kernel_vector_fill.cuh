#ifndef KERNEL_VECTOR_FILL
#define KERNEL_VECTOR_FILL

#include "kinds.h"
#include "../vector_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_vector_fill(VectorView<AtomT> view, AtomT val) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < view.size()) {
        view[j] = val;
    }
}

#endif