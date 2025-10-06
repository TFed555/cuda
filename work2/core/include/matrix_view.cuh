#ifndef MATRIXVIEW_H
#define MATRIXVIEW_H

#include "data.cuh"

template<AtomKind AtomT>
class IMatrix {
  protected:
    AtomT* data_;
    std::size_t nrows_; 
    std::size_t ncols_;
  public:
    IMatrix(AtomT* data, std::size_t nrows, std::size_t ncols)
    : data_(data), nrows_(nrows), ncols_(ncols) {};

    virtual ~IMatrix() {};

    virtual std::size_t size() const = 0;
    virtual std::size_t nrows() const = 0;
    virtual std::size_t ncols() const = 0;

    virtual AtomT& operator[] (std::size_t n) = 0;
    virtual const AtomT& operator[] (std::size_t n) const = 0;
    virtual AtomT& operator() (std::size_t i, std::size_t j) = 0;
    virtual const AtomT& operator() (std::size_t i, std::size_t j) const = 0;
}

#endif