#ifndef VECTOR_CUH
#define VECTOR_CUH

#include "kernels/kernel_vector_fill.cuh"
#include <memory>


template <AtomKind AtomT, template<AtomKind> typename Strategy>
class Vector {
  private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;
  public:
    Vector(std::size_t size) : data_(std::make_shared<Data<AtomT>>(size)),
                              view_(data_->data(), size)
    {};

    std::size_t size() const { return view_.size(); }

    Data<AtomT>& data() { return *data_; }
    const Data<AtomT>& data() const { return *data_; }

    VectorView<AtomT>& view() { return view_; }
    const VectorView<AtomT>& view() const { return view_; }

    void fill(AtomT val) {
      dim3 block(16, 16);
      dim3 grid = dim3((size() + dim3(16, 16).x - 1) / dim3(16, 16).x);
      kernel_vector_fill<<<grid, block>>>(view_, val);
      cudaDeviceSynchronize();
    }

    AtomT sum();
};

#endif