#ifndef MATMUL_NAIVE_STRATEGY_CUH
#define MATMUL_NAIVE_STRATEGY_CUH

#include "matmul_strategy.h"
#include "../include/kernels/kernel_matmul_naive.cuh"

template <AtomKind AtomT>
class MatmulNaiveStrategy : public MatmulStrategy<AtomT> {
public:
    void multiply(MatrixView<AtomT> a, MatrixView<AtomT> b, 
                                MatrixView<AtomT> res) override {
        dim3 block_size(16, 16);
        dim3 grid_size = make_grid_2d(res.nrows(), res.ncols(), block_size);

        kernel_matmul_naive<<<grid_size, block_size>>>(a, b, res);
        cudaDeviceSynchronize();
    }
};

#endif