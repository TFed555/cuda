#ifndef VECSUM_BR_STRATEGY_CUH
#define VECSUM_BR_STRATEGY_CUH

#include "vecsum_strategy.h"
#include "../include/kernels/kernel_vecred_br.cuh"

template <AtomKind AtomT>
class VecsumBrStrategy : public VecsumStrategy<VecsumBrStrategy, AtomT> {
public:
    static void addImpl(std::size_t grid_size, std::size_t block_size, std::size_t shm, VectorView<AtomT> a, AtomT* res) {
        kernel_vecred_br<<<grid_size, block_size, shm>>>(a, res);
    }
};

#endif