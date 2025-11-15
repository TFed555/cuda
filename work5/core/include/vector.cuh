#ifndef VECTOR_CUH
#define VECTOR_CUH

#include <memory>

template <AtomKind AtomT, typename Strategy>
class Vector {
  private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;
  public:
    Vector
}