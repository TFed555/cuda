#ifndef KERNEL_MATRIX_ADD
#define KERNEL_MATRIX_ADD

#include "matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matrix_sum(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res, std::size_t rows, std::size_t cols) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows && j < cols) {
    res[i * cols + j] = a[i * cols + j] + b[i * cols + j];
  }
}


#endif