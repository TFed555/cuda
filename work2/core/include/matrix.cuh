#ifndef MATRIX_H
#define MATRIX_H

#include "matrix_view.cuh"

template<AtomKind AtomT>
class Matrix {
  private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;
  public:
    Matrix(std::size_t nrows, std::size_t ncols)
              : data_(std::make_shared<Data<AtomT>>(nrows_ * ncols_)){
      view_ = std::make_unique<MatrixView<AtomT>>(data_->data(), nrows, ncols);
    }
    std::size_t size() const { return ncols; }
    std::size_t nrows() const { return nrows_;}
    std::size_t ncols() const { return ncols_; }

    
}
#endif