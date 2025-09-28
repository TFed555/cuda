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
class Vectors {
  private:
    struct Vec {
      T* ptr;
      VecType type;
    };
    std::vector<Vec> vectors;
    size_t N;
    size_t size;

  public:
    Vectors(int N, int countVecs) : N(N), size(N*sizeof(T)) {
        vectors.resize(countVecs);

        for (size_t i = 0; i < countVecs; i++) {
          vectors[i].ptr = static_cast<T*>(malloc(size));
          vectors[i].type = VecType::Host;
        }
    }

    Vectors(int N, int countVecs, bool cudaDevice) : N(N), size(N*sizeof(T)) {
        vectors.resize(countVecs);

        for(size_t i = 0; i < countVecs; i++) {
          T* devptr = nullptr;
          cudaMalloc(&devptr, size);
          vectors[i].ptr = devptr;
          vectors[i].type = VecType::Device;
        }
    }

    ~Vectors() {
      for (auto vec:vectors) {
        if (vec.type == VecType::Host) {
          free(vec.ptr);
        }
        else{
          cudaFree(vec.ptr);
        }
      }
    }

    void init_vectors(std::initializer_list<T*> pointers) {
      for (auto ptr:pointers) {
        for (size_t i = 0; i < N; i++) {
          ptr[i] = static_cast<T>(i);
        }
      }
    }

    void copy_device_vectors(std::initializer_list<T*> host_pointers,
                          std::initializer_list<T*> device_pointers, bool fromDevice) {
      auto host_it = host_pointers.begin();
      auto device_it = device_pointers.begin();

      if (host_pointers.size() != device_pointers.size()) {
        std::cout << "Sizes of lists must be same" << std::endl;
        return;
      }

      while (host_it != host_pointers.end()) {
        if (fromDevice) {
          cudaMemcpy(*host_it, device_it, size, cudaMemcpyDeviceToHost);
        }
        else {
          cudaMemcpy(*device_it, host_it, size, cudaMemcpyHostToDevice);
        }
        host_it++;
        device_it++;
      }
    }
    
   std::vector<T*> getVectors(int inds) {
      std::vector<T*> vecs;
      for (int i = 0; i < inds; i++) {
        vecs.push_back(vectors.at(i).ptr);
      }
      return vecs;
    }
};

int main() {
  int N = 256;
  Vectors<float> host_vectors(N, 3);
  std::vector<float*> vecs = host_vectors.getVectors(3);

  float *a = vecs[0];
  float *b = vecs[1];
  float *c = vecs[2];

  host_vectors.init_vectors({a, b});
  // for (int i=0; i<N;i++) {
  //   std::cout<<"host_A: "<<a[i]<<" ";
  //   std::cout<<"host_B: "<<b[i]<<std::endl;
  // }

  addVec_cpu(a, b, c, N);

  // for (int i=0; i<N;i++) {
  //   std::cout<<"host_C: "<a[i]<<std::endl;
  // }

  Vectors<float> device_vectors(N, 3, true);
  std::vector<float*> d_vecs = device_vectors.getVectors(3);

  float *d_a = d_vecs[0];
  float *d_b = d_vecs[1];
  float *d_c = d_vecs[2];

  device_vectors.copy_device_vectors({a, b}, {d_a, d_b}, false);

  addVec_gpu(d_a, d_b, d_c, N);

  device_vectors.copy_device_vectors({c}, {d_c}, true);

  for (int i=0; i<N;i++) {
    std::cout<<"C: "<<c[i]<<std::endl;
  }

}
