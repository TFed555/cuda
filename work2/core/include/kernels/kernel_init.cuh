#ifndef KERNEL_MATRIX_INIT
#define KERNEL_MATRIX_INIT

#include "../kinds.h"
#include "../matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matrix_init(MatrixView<AtomT> view, AtomT val) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < view.nrows() && j < view.ncols()) {
        view(i, j) = val;
    }
}

#endif