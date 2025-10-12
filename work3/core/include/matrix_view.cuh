#ifndef MATRIXVIEW_H
#define MATRIXVIEW_H

#include "data.cuh"

template<AtomKind AtomT>
class MatrixView {
  public:
    using atom_t = AtomT;
  private:
    AtomT* data_;
    std::size_t nrows_; 
    std::size_t ncols_;
  public:
    MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols)
    : data_(data), nrows_(nrows), ncols_(ncols) {};

    ~MatrixView() {};

     __host__ __device__  std::size_t size() const { return nrows_ * ncols_; }
     __host__ __device__  std::size_t nrows() const { return nrows_; }
     __host__ __device__  std::size_t ncols() const { return ncols_; }

    __host__ __device__ atom_t& operator[](std::size_t n) {
        return data_[n];
    }

    __host__ __device__ const atom_t& operator[](std::size_t n) const {
      return data_[n];
    }

    __host__ __device__ atom_t& operator() (std::size_t i, std::size_t j) {
      return data_[i * ncols_ + j];
    }

    __host__ __device__ const atom_t& operator() (std::size_t i, std::size_t j) const {
      return data_[i * ncols_ + j];
    }
};

#endif