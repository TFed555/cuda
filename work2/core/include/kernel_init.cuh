#ifndef KERNEL_MATRIX_INIT
#define KERNEL_MATRIX_INIT

#include "matrix.cuh"

template <AtomKind AtomT>
__global__ void kernel_matrix_init(MatrixView<AtomT> view, AtomT val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < view.nrows() && j < view.ncols()) {
        view(i, j) = i * val + j;
    }
}

#endif