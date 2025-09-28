#include <iostream>
#include <vector>
#include <cstdlib>

template<typename T>
class Vectors {
  private:
    std::vector<T*> vectors;
    size_t N;

  public:
    Vectors(int N, int countVecs) : N(N) {
        vectors.resize(countVecs);
        size_t size = N * sizeof(T);

        for (int i = 0; i < countVecs; i++) {
          vectors[i] = static_cast<T*>(malloc(size));
        }
    }

    ~Vectors() {
      for (auto vec:vectors) {
        free(vec);
      }
    }
    
    // template<typename... Pointers>
    // Vector(): Vector() {

    // }

    void init_vectors(std::initializer_list<T*> pointers) {
      for (auto ptr:pointers) {
        for (size_t i = 0; i < N; i++) {
          ptr[i] = static_cast<T>(i);
        }
      }
    }
    
    T* getVec(int ind) {
      return vectors.at(ind);
    }
};

// void init_vectors(float** a, float** b, float** c, int N) {
//     size_t size = N * sizeof(float);
    
//     *a = (float*)malloc(size);
//     *b = (float*)malloc(size);
//     *c = (float*)malloc(size);
    
//     for (int i = 0; i < N; i++) {
//         (*a)[i] = (float)i;
//         (*b)[i] = (float)i;
//     }
// }

// void copy_vectors(float* host_a, float* host_b, float* host_c, float** device_a, float** device_b, float** device_c, int N) {
//   size_t size = N * sizeof(float);

//   cudaMalloc(device_a, size);
//   cudaMalloc(device_b, size);
//   cudaMalloc(device_c, size);

//   cudaMemcpy(*device_a, host_a, size, cudaMemcpyHostToDevice);
//   cudaMemcpy(*device_b, host_b, size, cudaMemcpyHostToDevice);

// }

// void free_vectors(float* a, float* b, float* c) {
//     free(a);
//     free(b);
//     free(c);
// }

// void cudafree_vectors(float* device_a, float* device_b, float* device_c) {

//     cudaFree(device_a);
//     cudaFree(device_b);
//     cudaFree(device_c);
// }

// __global__ void addVec(float* a, float* b, float* c, int N) {
//   int i = blockDim.x * blockIdx.x + threadIdx.x;
//   if (i<N) {
//    c[i] = a[i] + b[i];
//   }
// }

// void addVec_cpu(float* a, float* b, float* c, int N) {
//   for(int i = 0; i < N; i++) {
//     c[i] = a[i] + b[i];
//   }
// }

// void addVec_gpu(float* a, float* b, float* c, int N) {
//   int threads_block = 128;
//   int blocks_grid = (N + threads_block - 1) / threads_block;
//   addVec<<<blocks_grid, threads_block>>>(a, b, c, N);
// }

int main() {
  int N = 256;
  Vectors<float> vectors(N, 3);
  float *a = vectors.getVec(0);
  float *b = vectors.getVec(1);
  vectors.init_vectors({a, b});
  float *c = vectors.getVec(2);

  for (int i=0; i<N;i++) {
    std::cout<<"A: "<<a[i]<<" ";
    std::cout<<"B: "<<b[i]<<std::endl;
  }
}
