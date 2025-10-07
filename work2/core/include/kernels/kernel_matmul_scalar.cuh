#ifndef KERNEL_MATRIX_MUL_SCALAR
#define KERNEL_MATRIX_MUL_SCALAR

#include "../matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matmul_scalar(MatrixView<AtomT> a, AtomT val, 
                                MatrixView<AtomT> res) {
  int i = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < res.nrows() && j < res.ncols()) {
    res(i, j) = a(i, j) * val;
  }
}


#endif