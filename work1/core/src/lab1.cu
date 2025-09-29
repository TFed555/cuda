#include <iostream>
#include <vector>
#include <cstdlib>

__global__ void addVec(float* a, float* b, float* c, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<N) {
   c[i] = a[i] + b[i];
  }
}

void addVec_cpu(float* a, float* b, float* c, int N) {
  for(int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}

void addVec_gpu(float* a, float* b, float* c, int N) {
  int threads_block = 128;
  int blocks_grid = (N + threads_block - 1) / threads_block;
  addVec<<<blocks_grid, threads_block>>>(a, b, c, N);
}

enum class VecType {
  Host,
  Device
};

template<typename T>
class Vector {
  private:
    struct Vec {
      T* ptr;
      VecType type;
    };
    Vec vector;
    size_t N;
    size_t size;

  public:
    Vector(int N) : N(N), size(N*sizeof(T)) {
          vector.ptr = static_cast<T*>(malloc(size));
          vector.type = VecType::Host;
    }

    Vector(int N, bool cudaDevice) : N(N), size(N*sizeof(T)) {
          T* devptr = nullptr;
          cudaMalloc(&devptr, size);
          vector.ptr = devptr;
          vector.type = VecType::Device;
    }

    ~Vector() {
        if (vector.type == VecType::Host) {
          free(vector.ptr);
        }
        else{
          cudaFree(vector.ptr);
        }
    }

    T& operator[](size_t i) {
        return vector.ptr[i];
    }

    T* ptr() {
      return vector.ptr;
    }

    void init_vector() {
        for (size_t i = 0; i < N; i++) {
          vector.ptr[i] = static_cast<T>(i);
      }
    }

    void copy_to_device(float* host_ptr) {
      cudaMemcpy(vector.ptr, host_ptr, size, cudaMemcpyHostToDevice);
    }

    void copy_to_host(float* host_ptr) {
      cudaMemcpy(host_ptr, vector.ptr, size, cudaMemcpyDeviceToHost);
    }
    
};

int main() {
  int N = 20;
  Vector<float> h_a(N);
  Vector<float> h_b(N);
  Vector<float> h_c(N);

  h_a.init_vector();
  h_b.init_vector();

  for (int i=0; i<N;i++) {
    std::cout<<"host_A: "<<h_a[i]<<" ";
    std::cout<<"host_B: "<<h_b[i]<<std::endl;
  }

  addVec_cpu(h_a.ptr(), h_b.ptr(), h_c.ptr(), N);

  for (int i=0; i<N;i++) {
    std::cout<<"host_C: "<< h_c[i]<< std::endl;
  }

  Vector<float> d_a(N, true);
  Vector<float> d_b(N, true);
  Vector<float> d_c(N, true);

  d_a.copy_to_device(h_a.ptr());
  d_b.copy_to_device(h_b.ptr());

  addVec_gpu(d_a.ptr(), d_b.ptr(), d_c.ptr(), N);

  d_c.copy_to_host(h_c.ptr());

  for (int i=0; i<N;i++) {
    std::cout<<"d_C: "<<h_c[i]<<std::endl;
  }

}
