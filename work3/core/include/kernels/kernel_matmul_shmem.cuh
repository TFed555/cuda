#ifndef KERNEL_MATRIX_MUL_SHMEM
#define KERNEL_MATRIX_MUL_SHMEM

#include "../matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matmul_shmem(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res) {
  int blockRow = blockIdx.y;
  int blockCol = blockIdx.x;

  const int BLOCK_SIZE = 16;

  AtomT sum = 0;
  int row = threadIdx.y;
  int col = threadIdx.x;

  __shared__ AtomT a_sub_sh[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ AtomT b_sub_sh[BLOCK_SIZE][BLOCK_SIZE];

  int sub_rowA = BLOCK_SIZE*blockRow + row;
  int sub_colB = BLOCK_SIZE*blockCol + col;

  for (int k = 0; k < (a.ncols() + BLOCK_SIZE - 1) / BLOCK_SIZE; k++) {
    int sub_colA = BLOCK_SIZE*k + col;
    int sub_rowB = BLOCK_SIZE*k + row;
    a_sub_sh[row][col] = (sub_rowA < a.nrows() && sub_colA < a.ncols()) ? 
                            a(sub_rowA, sub_colA) : 0;
    b_sub_sh[row][col] = (sub_rowB < b.nrows() && sub_colB < b.ncols()) ?
                            b(sub_rowB, sub_colB) : 0;
    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; i++) {
      sum += a_sub_sh[row][i] * b_sub_sh[i][col];
    }
    __syncthreads();
  }
  
  if (sub_rowA < res.nrows() && sub_colB < res.ncols())
    res(sub_rowA, sub_colB) = sum;
}


#endif