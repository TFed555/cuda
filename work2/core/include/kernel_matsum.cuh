#ifndef KERNEL_MATRIX_ADD
#define KERNEL_MATRIX_ADD

#include "matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matrix_sum(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  std::size_t rows = a.nrows();
  std::size_t cols = a.ncols();

  if (i < rows && j < cols) {
    res(i, j) = a(i, j) + b(i, j);
  }
}


#endif