#ifndef MATMUL_WMMA_STRATEGY_CUH
#define MATMUL_WMMA_STRATEGY_CUH

#include "matmul_strategy.h"
#include "../include/kernels/kernel_matmul_wmma.cuh"

template <AtomKind AtomA, AtomKind AtomC>
class MatmulWmmaStrategy : public MatmulStrategy<AtomA, AtomC>  {
public:
    void multiply(MatrixView<AtomA> a, MatrixView<AtomA> b, 
                                MatrixView<AtomC> res) override {
        auto [grid_size, block_size] = cuda_utils::make_wmma_grid_block_2d(res.nrows(), res.ncols());

        kernel_matmul_wmma<<<grid_size, block_size>>>(a, b, res);
        cudaDeviceSynchronize();
    }
};

#endif