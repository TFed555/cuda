#ifndef KERNEL_VECRED_NOBR
#define KERNEL_VECRED_NOBR

#include "../vector_view.cuh"

template <AtomKind AtomT>
__global__ void kernel_vecred_nobr(VectorView<AtomT> a, AtomT* blockSum) {

    extern __shared__ AtomT sh[];

    std::size_t tid = threadIdx.x;
    std::size_t blockSize = blockDim.x;
    //std::size_t t = blockIdx.x * blockDim.x + tid;

    //printf("%llu %llu \n", tid, t);

    AtomT sum = static_cast<AtomT>(0);

    for (size_t j = blockIdx.x * blockDim.x + tid; j < a.size(); j += gridDim.x * blockDim.x) {
      sum += a[j];
    }

    sh[tid] = sum;

    //printf("%llu sh %f \n", tid, sh[tid]);

    __syncthreads();

  
    for (std::size_t i = blockSize/2; i>0; i/=2) {
        if (tid < i) {
          sh[tid] += sh[tid + i];
          //printf("%llu sh2 %f \n", tid, sh[tid]);
        }
        __syncthreads();
    }

    if (tid == 0){
        //printf("%llu res %f \n", tid, sh[0]);
        blockSum[blockIdx.x] = sh[0];
    }
}

template <AtomKind AtomT>
__global__ void kernel_vecred_final_nobr(AtomT* blockSum,
                                    AtomT* res,
                                    std::size_t blocks, std::size_t n)
{
    extern __shared__ AtomT sh[];
    std::size_t tid = threadIdx.x;

     if (tid < blocks) {
            sh[tid] = blockSum[tid];
        } else {
            sh[tid] = static_cast<AtomT>(0);
        }
        __syncthreads();

    for (std::size_t s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
            sh[tid] += sh[tid + s];
        __syncthreads();
    }

    if (tid == 0)
        *res = sh[0];
}


#endif