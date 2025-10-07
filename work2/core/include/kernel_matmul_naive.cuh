#ifndef KERNEL_MATRIX_MUL
#define KERNEL_MATRIX_MUL

#include "matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matmul_naive(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < res.nrows() && j < res.ncols()) {
    AtomT sum = 0;
    for (int k = 0; k < a.ncols(); k++) {
      sum += a(i, k) * b(k, j);
    }
    res(i, j) = sum;
  }
}


#endif