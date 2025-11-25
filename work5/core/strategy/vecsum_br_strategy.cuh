#ifndef VECSUM_BR_STRATEGY_CUH
#define VECSUM_BR_STRATEGY_CUH

#include "vecsum_strategy.h"
#include "../include/kernels/kernel_vecred_br.cuh"

template <AtomKind AtomT>
class VecsumBrStrategy : public VecsumStrategy<VecsumBrStrategy, AtomT> {
public:
    static void addImpl(VectorView<AtomT> a, AtomT* res) {
        constexpr std::size_t block_size = 256;
        std::size_t grid_size = (a.size() + block_size - 1) / block_size;
        std::size_t shm = block_size * sizeof(AtomT);

        kernel_vecred_br<<<grid_size, block_size, shm>>>(a, res);
        
        cudaError_t errSync  = cudaGetLastError();
        if (errSync != cudaSuccess) {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        }
        cudaDeviceSynchronize();
    }
};

#endif