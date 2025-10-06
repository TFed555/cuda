#ifndef MATRIX_H
#define MATRIX_H

#include "data.cuh"

template<AtomKind AtomT>
class IMatrix {
  private:
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;
}




#endif