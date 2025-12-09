#ifndef VECTOR_CUH
#define VECTOR_CUH

#include "kernels/kernel_vector_fill.cuh"
#include <memory>


template <AtomKind AtomT, template<typename> typename Strategy>
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
      int blockSize = 256;
      int gridSize  = (size() + blockSize - 1) / blockSize;

      kernel_vector_fill<<<gridSize, blockSize>>>(view_, val);
      cudaDeviceSynchronize();

      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
          throw std::runtime_error(std::string("CUDA error: ") +
                                  cudaGetErrorString(err));
      }
    }

    AtomT sum();
};

#endif