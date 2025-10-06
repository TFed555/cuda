#ifndef MATRIXVIEW_H
#define MATRIXVIEW_H

#include "data.cuh"

template<AtomKind AtomT>
class MatrixView {
  protected:
    AtomT* data_;
    std::size_t nrows_; 
    std::size_t ncols_;
  public:
    MatrixView(AtomT* data, std::size_t nrows, std::size_t ncols)
    : data  _(data), nrows_(nrows), ncols_(ncols) {};

    virtual ~IMatrix() {};

    virtual std::size_t size() const = 0;
    virtual std::size_t nrows() const = 0;
    virtual std::size_t ncols() const = 0;

    virtual MatrixAccessor<AtomT> accessor() = 0;
    virtual const MatrixAccessor<AtomT> accessor() const = 0;

    virtual AtomT& operator[] (std::size_t n) { return accessor()[n]; };
    virtual const AtomT& operator[] (std::size_t n) { return accessor()[n]; }
    virtual AtomT& operator() (std::size_t i, std::size_t j) { return accessor()(i,j);}
    virtual const AtomT& operator() (std::size_t i, std::size_t j) const { return accessor()(i,j);}
}

#endif