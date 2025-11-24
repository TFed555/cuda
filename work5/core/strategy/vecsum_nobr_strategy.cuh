#ifndef VECSUM_NOBR_STRATEGY_CUH
#define VECSUM_NOBR_STRATEGY_CUH

#include "vecsum_strategy.h"
#include "../include/kernels/kernel_vecred_nobr.cuh"

template <AtomKind AtomT>
class VecsumNobrStrategy : public VecsumStrategy<VecsumNobrStrategy<AtomT>, AtomT> {
public:
    void addImpl(VectorView<AtomT> a, AtomT res) const {
        dim3 block_size(16, 16);
        dim3 grid_size = dim3((a.size() + block_size.x - 1) / block_size.x);

        kernel_vecred_nobr<<<grid_size, block_size>>>(a, res);
        
        cudaError_t errSync  = cudaGetLastError();
        if (errSync != cudaSuccess) {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        }
        cudaDeviceSynchronize();
    }
};

#endif