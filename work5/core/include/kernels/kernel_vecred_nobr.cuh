#ifndef KERNEL_VECRED_NOBR
#define KERNEL_VECRED_NOBR

#include "kinds.h"
#include "../vector_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_vecred_nobr(VectorView<AtomT> a, AtomT* res) {

    extern __shared__ AtomT sh[];

    std::size_t tid = threadIdx.x;
    std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    //printf("%llu %llu \n", tid, t);

    sh[tid] = (t < a.size()) ? a[t] : 0;

    printf("%llu sh %f \n", tid, sh[tid]);

    __syncthreads();

    for (std::size_t i = blockDim.x/2; i>0; i/=2) {
        if (tid < i) {
          sh[tid] += sh[tid + i];
          printf("%llu sh2 %f \n", tid, sh[tid]);
        }
        __syncthreads();
    }

    if (tid == 0){
        printf("%llu res %f \n", tid, sh[0]);
        atomicAdd(res, sh[0]);
    }
}

#endif