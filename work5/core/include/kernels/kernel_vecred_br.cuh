#ifndef KERNEL_VECRED_BR
#define KERNEL_VECRED_BR

#include "../vector_view.cuh"

template <AtomKind AtomT>
__device__ inline AtomT warp_reduce(AtomT value) {
  #pragma unroll
    for (int j = warpSize/2; j > 0; j /= 2) {
        value += __shfl_down_sync(0xffffffff, value, j, 32);
    }
    return value;
}

template <AtomKind AtomT>
__global__ void kernel_vecred_br(VectorView<AtomT> a, AtomT* res) {

    extern __shared__ AtomT sh[];

    std::size_t tid = threadIdx.x;
    std::size_t t = blockIdx.x * blockDim.x + tid;
    std::size_t laneId = tid % warpSize;
    std::size_t warpId = tid / warpSize;

    AtomT value = 0;
    if (t < a.size()) {
      value = a[t];
    }

    value = warp_reduce(value);

    if (laneId == 0) {
      sh[warpId] = value;
     // printf("%llu sh %f \n", warpId, sh[warpId]);
    }

    __syncthreads();

    if (warpId == 0){
        int num_warps = blockDim.x / warpSize;
       // printf("%llu ss %d \n", laneId, num_warps);
        value = laneId < num_warps ? sh[laneId] : 0;
        value = warp_reduce(value);
       // printf("%llu shd bratik %f \n", laneId, value);
        if (laneId == 0) {
            atomicAdd(res, value);
        }
    }
}

#endif