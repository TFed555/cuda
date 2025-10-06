#ifndef MATRIX_ACCESSOR
#define MATRIX_ACCESSOR

#include "kinds.h"

#include "matrix_view.cuh"

template <AtomKind AtomT>
struct MatrixAccessor {
  using atom_t = AtomT;
  private:
    atom_t* data_;
    std::size_t size_;

  public:
    __host__ __device__ MatrixAccessor(atom_t* data, std::size_t size_) 
        : data_(data), size_(size) { }

  __host__ __device__ std::size_t size() const {
    return size_;
  }

  __host__ __device__ atom_t& operator[](std::size_t i) {
    return data_[i];
  }

  __host__ __device__ const atom_t& operator[](std::size_t i) const {
    return data_[i];
  }  
};

#endif