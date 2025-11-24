#ifndef VECTOR_VIEW_CUH
#define VECTOR_VIEW_CUH

#include "data.cuh"

template <AtomKind AtomT>
class VectorView {
  using atom_t = AtomT;
  private:
    atom_t* data_;
    std::size_t size_;
  public:
    __host__ __device__
    VectorView(atom_t* data, std::size_t size) : data_(data),
              size_(size)
    {};

    __host__ __device__ ~VectorView() {};

    __host__ __device__ std::size_t size() const { return size_; }

    __host__ __device__ atom_t& operator[](std::size_t n) {
        return data_[n];
    }

    __host__ __device__ const atom_t& operator[](std::size_t n) const {
      return data_[n];
    }

    __host__ __device__ atom_t& operator() (std::size_t i, std::size_t j) {
      return data_[i];
    }

    __host__ __device__ const atom_t& operator() (std::size_t i, std::size_t j) const {
      return data_[i];
    }
};

#endif