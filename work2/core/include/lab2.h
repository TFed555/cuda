#define LAB2_H

#include <cstdlib>
#include <cuda_runtime.h>

__global__ void addMultiMatrix(float* x, float* w, float* b, float* res, int rows, int cols, int K);

void addMultiMatrix_cpu(float* x, float* w, float* b, float* res, int rows, int cols, int K);

void addMultiMatrix_gpu(float* x, float* w, float* b, float* res, int rows, int cols, int K);

enum class VecType {
    Host,
    Device
};

template<typename T>
class Vector {
private:
    struct Vec {
        T* ptr;
        VecType type;
    };
    Vec vector;
    size_t N;
    size_t size;

public:
    Vector(int N);
    Vector(int N, bool cudaDevice);
    
    ~Vector();
    
    T& operator[](size_t i);
    T* ptr();
    void init_vector();
    void copy_to_device(float* host_ptr);
    void copy_to_host(float* host_ptr);
};

template<typename T>
Vector<T>::Vector(int N) : N(N), size(N*sizeof(T)) {
    vector.ptr = static_cast<T*>(malloc(size));
    vector.type = VecType::Host;
}

template<typename T>
Vector<T>::Vector(int N, bool cudaDevice) : N(N), size(N*sizeof(T)) {
    T* devptr = nullptr;
    cudaMalloc(&devptr, size);
    vector.ptr = devptr;
    vector.type = VecType::Device;
}

template<typename T>
Vector<T>::~Vector() {
    if (vector.type == VecType::Host) {
        free(vector.ptr);
    } else {
        cudaFree(vector.ptr);
    }
}

template<typename T>
T& Vector<T>::operator[](size_t i) {
    return vector.ptr[i];
}

template<typename T>
T* Vector<T>::ptr() {
    return vector.ptr;
}

template<typename T>
void Vector<T>::init_vector() {
    for (size_t i = 0; i < N; i++) {
        vector.ptr[i] = static_cast<T>(i);
    }
}

template<typename T>
void Vector<T>::copy_to_device(float* host_ptr) {
    cudaMemcpy(vector.ptr, host_ptr, size, cudaMemcpyHostToDevice);
}

template<typename T>
void Vector<T>::copy_to_host(float* host_ptr) {
    cudaMemcpy(host_ptr, vector.ptr, size, cudaMemcpyDeviceToHost);
}