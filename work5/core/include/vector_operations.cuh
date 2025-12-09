#ifndef VECTOR_OPERATIONS_CUH
#define VECTOR_OPERATIONS_CUH

#include "vector.cuh"
#include "../strategy/vecsum_strategy.h"

template <AtomKind AtomT, template<typename> typename Strategy>
AtomT Vector<AtomT, Strategy>::sum() {
    Data<AtomT> res_device(1);
    AtomT res_host = 0;
    res_device.copy_to_device(&res_host);

    Strategy<AtomT>::add(view(), res_device.data());
    
    cudaError_t errSync  = cudaGetLastError();
    if (errSync != cudaSuccess) {
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    }
    cudaDeviceSynchronize();
    res_device.copy_to_host(&res_host);

    return res_host;
};

#endif