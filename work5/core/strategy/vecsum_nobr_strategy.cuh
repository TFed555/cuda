#ifndef VECSUM_NOBR_STRATEGY_CUH
#define VECSUM_NOBR_STRATEGY_CUH

#include "vecsum_strategy.h"
#include "../include/kernels/kernel_vecred_nobr.cuh"

template <AtomKind AtomT>
class VecsumNobrStrategy : public VecsumStrategy<VecsumNobrStrategy, AtomT> {
public:
   static void addImpl(VectorView<AtomT> a, AtomT* res) {
        std::size_t n = a.size();
        constexpr std::size_t block_size = 256;
        std::size_t blocks = (((n + block_size - 1) / block_size) < 128) ? ((n + block_size - 1) / block_size) : 128;
        std::size_t shm = block_size * sizeof(AtomT);

        Data<AtomT> partSum_device(blocks);
        //cudaMemset(partSum_device.data(), 0, blocks * sizeof(AtomT));

        kernel_vecred_nobr<<<blocks, block_size, shm>>>(a, partSum_device.data());

        cudaError_t errSync  = cudaGetLastError();
        if (errSync != cudaSuccess) {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        }

        kernel_vecred_final_nobr<<<1, block_size, shm>>>(partSum_device.data(), res, blocks, n);

        cudaError_t err2 = cudaGetLastError();
        if (err2 != cudaSuccess) {
            printf("Kernel2 error: %s\n", cudaGetErrorString(err2));
        }
    }
};

#endif