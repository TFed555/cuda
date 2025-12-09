#ifndef VECSUM_BR_STRATEGY_CUH
#define VECSUM_BR_STRATEGY_CUH

#include "vecsum_strategy.h"
#include "../include/kernels/kernel_vecred_br.cuh"

template <AtomKind AtomT>
class VecsumBrStrategy : public VecsumStrategy<VecsumBrStrategy, AtomT> {
public:
   static void addImpl(VectorView<AtomT> a, 
                        AtomT* res) {
        std::size_t n = a.size();
        constexpr std::size_t block_size = 256;
        std::size_t numBlocks = (((n + block_size - 1) / block_size) < 128) ? ((n + block_size - 1) / block_size) : 128;
        std::size_t shm = block_size * sizeof(AtomT);

        Data<AtomT> partSum_device(numBlocks);
        //cudaMemset(partSum_device.data(), 0, blocks * sizeof(AtomT));

        kernel_vecred_br<<<numBlocks, block_size, ((block_size + 31) / 32) * sizeof(AtomT)>>>(a, res);

        cudaError_t errSync  = cudaGetLastError();
        if (errSync != cudaSuccess) {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        }
    }
};

#endif