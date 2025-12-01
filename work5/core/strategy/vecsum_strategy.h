#ifndef IVECSUM_STRATEGY_H
#define IVECSUM_STRATEGY_H

#include "../include/vector.cuh"

template <AtomKind AtomT>
class VectorView;

template <template<AtomKind> class Derived, AtomKind AtomT>
class VecsumStrategy {
public:
    static void add(VectorView<AtomT> a, AtomT* res) {
        constexpr std::size_t block_size = 256;
        std::size_t grid_size = (a.size() + block_size - 1) / block_size;
        std::size_t shm = block_size * sizeof(AtomT);
        
        Derived<AtomT>::addImpl(grid_size, block_size, shm, a, res);

        cudaError_t errSync  = cudaGetLastError();
        if (errSync != cudaSuccess) {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
        }
    }
};

#endif