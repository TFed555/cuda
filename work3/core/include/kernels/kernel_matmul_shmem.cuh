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
  int ch = (a.ncols() + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (int k = 0; k < ch; k++) {
    //if (i < res.nrows() && j < res.ncols())
    a_sub_sh[row][col] = a(BLOCK_SIZE*blockRow + row, BLOCK_SIZE*k + col);
    b_sub_sh[row][col] = b(BLOCK_SIZE*k + row, BLOCK_SIZE*blockCol + col);
    __syncthreads();

    for (int i = 0; i < BLOCK_SIZE; i++) {
      sum += a_sub_sh[row][i] * b_sub_sh[i][col];
    }
    __syncthreads();
  }

  res(BLOCK_SIZE*blockRow + row, BLOCK_SIZE*blockCol + col) = sum;
}


#endif