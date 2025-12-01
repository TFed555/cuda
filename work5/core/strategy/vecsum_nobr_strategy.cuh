#ifndef VECSUM_NOBR_STRATEGY_CUH
#define VECSUM_NOBR_STRATEGY_CUH

#include "vecsum_strategy.h"
#include "../include/kernels/kernel_vecred_nobr.cuh"

template <AtomKind AtomT>
class VecsumNobrStrategy : public VecsumStrategy<VecsumNobrStrategy, AtomT> {
public:
   static void addImpl(std::size_t grid_size, std::size_t block_size, std::size_t shm, VectorView<AtomT> a, AtomT* res) {
        kernel_vecred_nobr<<<grid_size, block_size, shm>>>(a, res);
    }
};

#endif