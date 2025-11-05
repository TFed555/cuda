#ifndef MATMUL_WMMA_STRATEGY_CUH
#define MATMUL_WMMA_STRATEGY_CUH

#include "matmul_strategy.h"
#include "../include/kernels/kernel_matmul_wmma.cuh"

template <AtomKind AtomA, AtomKind AtomC>
class MatmulWmmaStrategy : public MatmulStrategy<AtomA, AtomC>  {
public:
    void multiply(MatrixView<AtomA> a, MatrixView<AtomA> b, 
                                MatrixView<AtomC> res) override {
        dim3 block_size(128, 4);
        dim3 grid_size = make_wmma_grid_2d(res.nrows(), res.ncols(), block_size);

        kernel_matmul_wmma<<<grid_size, block_size>>>(a, b, res);
        cudaDeviceSynchronize();
    }
};

#endif