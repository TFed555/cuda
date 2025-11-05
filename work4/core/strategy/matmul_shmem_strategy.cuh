#ifndef MATMUL_SHMEM_STRATEGY_CUH
#define MATMUL_SHMEM_STRATEGY_CUH

#include "matmul_strategy.h"
#include "../include/kernels/kernel_matmul_shmem.cuh"

template <AtomKind AtomA, AtomKind AtomC>
class MatmulShmemStrategy : public MatmulStrategy<AtomA, AtomC>  {
public:
    void multiply(MatrixView<AtomA> a, MatrixView<AtomA> b, 
                                MatrixView<AtomC> res) override {
        dim3 block_size(16, 16);
        dim3 grid_size = cuda_utils::make_grid_2d(res.nrows(), res.ncols(), block_size);

        kernel_matmul_shmem<<<grid_size, block_size>>>(a, b, res);
        cudaDeviceSynchronize();
    }
};

#endif