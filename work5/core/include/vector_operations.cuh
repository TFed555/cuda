#ifndef VECTOR_OPERATIONS_CUH
#define VECTOR_OPERATIONS_CUH

#include "vector.cuh"

template <AtomKind AtomT, typename Strategy>
AtomT Vector::sum() {
    Data<AtomT> res_device;
    AtomT res_host = 0;
    res_device.copy_to_device(&res_host);

    Strategy::add(view(), res_device.data());
    
    cudaError_t errSync  = cudaGetLastError();
    if (errSync != cudaSuccess) {
        printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
    }
    cudaDeviceSynchronize();
    res_device.copy_to_host(&res_host);

    return result;
};

#endif