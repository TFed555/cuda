#ifndef MATRIXVIEW_CUH
#define MATRIXVIEW_CUH

#include "data.cuh"

template<AtomKind AtomT>
class MatrixView {
  public:
    using atom_t = AtomT;
  private:
    AtomT* data_;
    std::size_t nrows_; 
    std::size_t ncols_;
    std::size_t stride_;
  public:
    __host__ __device__
    MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols, std::size_t stride)
    : data_(data), nrows_(nrows), ncols_(ncols), stride_(stride) {};

    __host__ __device__
    ~MatrixView() {};

     __host__ __device__  std::size_t size() const { return nrows_ * ncols_; }
     __host__ __device__  std::size_t stride() const { return stride_; }
     __host__ __device__  std::size_t nrows() const { return nrows_; }
     __host__ __device__  std::size_t ncols() const { return ncols_; }

    __host__ __device__ atom_t& operator[](std::size_t n) {
        return data_[n];
    }

    __host__ __device__ const atom_t& operator[](std::size_t n) const {
      return data_[n];
    }

    __host__ __device__ atom_t& operator() (std::size_t i, std::size_t j) {
      return data_[i * stride_ + j];
    }

    __host__ __device__ const atom_t& operator() (std::size_t i, std::size_t j) const {
      return data_[i * stride_ + j];
    }

};

#endif