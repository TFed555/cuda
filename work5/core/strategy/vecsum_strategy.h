#ifndef IVECSUM_STRATEGY_H
#define IVECSUM_STRATEGY_H

#include "../include/vector.cuh"

template <AtomKind AtomT>
class VectorView;

template <template<AtomKind> class Derived, AtomKind AtomT>
class VecsumStrategy {
public:
    static void add(VectorView<AtomT> a, AtomT* res) {
        std::size_t n = a.size();
        constexpr std::size_t block_size = 256;
        std::size_t blocks = (((n + block_size - 1) / block_size) < 128) ? ((n + block_size - 1) / block_size) : 128;
        std::size_t shm = block_size * sizeof(AtomT);

        Data<AtomT> partSum_device(blocks);
        cudaMemset(partSum_device.data(), 0, blocks * sizeof(AtomT));

        Derived<AtomT>::addImpl(blocks, block_size, shm, a, partSum_device.data());

        cudaError_t errSync  = cudaGetLastError();
        if (errSync != cudaSuccess) {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        }

        Derived<AtomT>::addFinalImpl(1, block_size, shm, partSum_device.data(), res, blocks, a.size());

        cudaError_t err2 = cudaGetLastError();
        if (err2 != cudaSuccess) {
            printf("Kernel2 error: %s\n", cudaGetErrorString(err2));
        }

        }
};

#endif