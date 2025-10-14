#ifndef MATRIXVIEW_CUH
#define MATRIXVIEW_CUH

#include "data.cuh"
#include "../strategy/matmul_strategy.h"

template<AtomKind AtomT>
class MatrixView {
  public:
    using atom_t = AtomT;
  private:
    AtomT* data_;
    std::size_t nrows_; 
    std::size_t ncols_;
    IMatmulStrategy<AtomT>* strategy_;
  public:
    __host__ __device__
    MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols)
    : data_(data), nrows_(nrows), ncols_(ncols), strategy_(nullptr) {};

    __host__ __device__
    ~MatrixView() {};

     __host__ __device__  std::size_t size() const { return nrows_ * ncols_; }
     __host__ __device__  std::size_t nrows() const { return nrows_; }
     __host__ __device__  std::size_t ncols() const { return ncols_; }
     __host__ __device__  IMatmulStrategy<AtomT>* strategy() const 
                          { return strategy_; }

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

    __host__ void set_strategy(IMatmulStrategy<AtomT>* strategy) {
      strategy_ = strategy;
    }

};

#endif