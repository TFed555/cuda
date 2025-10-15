#ifndef KERNEL_MATRIX_FILL
#define KERNEL_MATRIX_FILL

#include "../kinds.h"
#include "../matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matrix_fill(MatrixView<AtomT> view, AtomT val) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < view.nrows() && j < view.ncols()) {
        view(i, j) = val;
    }
}

#endif