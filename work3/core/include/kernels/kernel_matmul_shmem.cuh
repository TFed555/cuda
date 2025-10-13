#ifndef KERNEL_MATRIX_MUL_SHMEM
#define KERNEL_MATRIX_MUL_SHMEM

#include "../matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matmul_shmem(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;
  //MatrixView<AtomT> resSub.get_submatrix(res, blockRow, blockCol);
  Matrix<AtomT> res_sub(blockDim.y, blockDim.x, res.stride());

  if (i < res.nrows() && j < res.ncols()) {
    AtomT sum = 0;
    for (int k = 0; k < a.ncols(); k++) {
      sum += a(i, k) * b(k, j);
    }
    res(i, j) = sum;
  }
}


#endif