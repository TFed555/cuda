#ifndef MATMUL_SHMEM_STRATEGY_CUH
#define MATMUL_SHMEM_STRATEGY_CUH

#include "matmul_strategy.h"
#include "../include/kernels/kernel_matmul_shmem.cuh"

template <AtomKind AtomT>
class MatmulShmemStrategy : public IMatmulStrategy<AtomT> {
public:
    void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res) override {
        dim3 block_size(16, 16);
        dim3 grid_size = make_grid_2d(a.nrows(), a.ncols(), block_size);

        kernel_matmul_shmem<<<grid_size, block_size>>>(a, b, res);
        cudaDeviceSynchronize();
    }
};

#endif