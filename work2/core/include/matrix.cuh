#ifndef MATRIX_H
#define MATRIX_H

#include "kernels/kernel_init.cuh"
#include "../utils/cuda_utils.cuh"

#include "matrix_view.cuh"
#include <memory>

template<AtomKind AtomT>
class Matrix {
  private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;
    //MatrixAccessor<AtomT> accessor_;
  public:
    Matrix(std::size_t nrows, std::size_t ncols)
              : data_(std::make_shared<Data<AtomT>>(nrows * ncols)),
                view_(data_->data(), nrows, ncols)
    {}
    std::size_t size() const { return view_.size(); }
    std::size_t nrows() const { return view_.nrows() ;}
    std::size_t ncols() const { return view_.ncols() ; }

    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    void init(AtomT val) {
      dim3 block(16, 16);
      dim3 grid = make_grid_2d(nrows(), ncols(), block);
      kernel_matrix_init<<<grid, block>>>(view_, val);
      cudaDeviceSynchronize();
    }

    MatrixView<AtomT>& view() { return view_; }
    const MatrixView<AtomT>& view() const { return view_; }
};

#endif