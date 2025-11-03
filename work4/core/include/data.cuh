#ifndef DATA_CUH
#define DATA_CUH

#include <cuda_runtime.h>
#include "kinds.h"

template <AtomKind AtomT>
class Data {
private:
  std::size_t size_;
  AtomT* data_;
public:
  Data(std::size_t size) : size_(size), data_(nullptr) {
    cudaMalloc(&data_, size_*sizeof(AtomT));
  }

  Data(const Data& other) : size_(other.size_), data_(nullptr) {
    cudaMalloc(&data_, size_*sizeof(AtomT));
    if (other.data_ != nullptr) {
      cudaMemcpy(data_, other.data_, size_*sizeof(AtomT), cudaMemcpyDeviceToDevice);
    }
  }

  ~Data() {
    if (data_) {
      cudaFree(data_);
    }
  }

  Data& operator= (const Data& obj) { 
    if (this == &obj) {
      return *this;
    }
    if (data_ != nullptr) {
      cudaFree(data_);
      size_ = obj.size_;
      cudaMalloc(&data_, size_*sizeof(AtomT));
      
      if (obj.data_ != nullptr) {
        cudaMemcpy(data_, obj.data_, size_*sizeof(AtomT), cudaMemcpyDeviceToDevice);
      }
    }
    return *this; 
  }

  AtomT* data() {
    return data_;
  }
  const AtomT* data() const {
    return data_;
  }

  std::size_t size() {
    return size_;
  }

  void copy_to_host(AtomT* host_ptr) {
    cudaMemcpy(host_ptr, data_, size_*sizeof(AtomT), cudaMemcpyDeviceToHost);
  }

  void copy_to_device(AtomT* host_ptr) {
    cudaMemcpy(data_, host_ptr, size_*sizeof(AtomT), cudaMemcpyHostToDevice);
  }
};


#endif