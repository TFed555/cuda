#ifndef MATRIX_H
#define MATRIX_H

#include "matrix_view.cuh"

template<AtomKind AtomT>
class Matrix {
  private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;
    //MatrixAccessor<AtomT> accessor_;
  public:
    Matrix(std::size_t nrows, std::size_t ncols)
              : data_(std::make_shared<Data<AtomT>>(nrows * ncols))
    {}
    std::size_t size() const { return view_.size(); }
    std::size_t nrows() const { return view_.nrows_ ;}
    std::size_t ncols() const { return view_.ncols_ ; }

    MatrixView<AtomT>& view() { return view_; }
    const MatrixView<AtomT>& view() const { return view_; }
};

#endif