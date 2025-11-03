#ifndef KERNEL_MATRIX_MUL_WMMA
#define KERNEL_MATRIX_MUL_WMMA

#include "../matrix_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_matmul_wmma(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res) {
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;

    int m = a.nrows();
    int n = b.ncols();
    int k = a.ncols();

    int lda = (k + WMMA_K-1)/WMMA_K*WMMA_K;
    int ldb = (k + WMMA_K-1)/WMMA_K*WMMA_K;
    int ldc = b.ncols();

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    wmma::fragment<wmma::matrix_a,WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    int rowA = WMMA_M * warpM;
    int colB = WMMA_N * warpN;

    for (int i = 0; i < k; i+=WMMA_K) {
      int colA = i;
      int rowB = i;
      if (rowA < m && colA < k && rowB < k && colB < n) {
        half* a_tile_ptr = a + rowA * lda + colA;
        wmma::load_matrix_sync(a_frag, a_tile_ptr, lda);
        half* b_tile_ptr = b + colB * ldb + rowB;
        wmma::load_matrix_sync(b_frag, b_tile_ptr, ldb);

        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
      }
    }

    int rowC = warpN * WMMA_N;
    int colC = warpM * WMMA_M;

    if (rowC < m && colC < n) {
       wmma::load_matrix_sync(c_frag, c + rowC*ldc + colC, ldc, wmma::mem_row_major);

       wmma::store_matrix_sync(res + colC + rowC * ldc, c_frag, ldc, wmma::mem_row_major);
    }
} 


#endif