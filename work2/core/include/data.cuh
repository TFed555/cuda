#ifndef DATA_H
#define DATA_H

#include "kinds.h"

template <AtomKind AtomT>
class Data {
private:
  std::size_t size_;
  AtomT* data_;
public:
  using atom_t = AtomT;
  Data(std::size_t size) : size_(size) {
    data_ = nullptr;
    cudaMalloc(&data_, size * sizeof(AtomT));
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
    return this.data_;
  }
  const AtomT* data() const {
    return this.data_;
  }

  const std::size_t size() {
    return this.size_;
  }

  void copy_to_host(AtomT* host_ptr) {
    cudaMemcpy(host_ptr, data_, size_*sizeof(AtomtT), cudaMemcpyDeviceToHost);
  }

  void copy_to_device(AtomT* host_ptr) {
    cudaMemcpy(data_, host_ptr, size_*sizeof(AtomtT), cudaMemcpyHostToDevice);
  }
}


#endif