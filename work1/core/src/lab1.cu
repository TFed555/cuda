#include <iostream>
#include "../includes/lab1.cuh"

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

__host__ int main() {
  const int N = 256;
  size_t size = N * sizeof(float);

  float* host_a = (float*) malloc(size);
  float* host_b = (float*) malloc(size);
  float* host_c = (float*) malloc(size);
  float* host_c_cpu = (float*) malloc(size);

  for (int i = 0; i < N; i++) {
    host_a[i] = i;
    host_b[i] = i;
  }
  
  addVec_cpu(host_a, host_b, host_c_cpu, N);

  float* device_a;
  float* device_b;
  float* device_c;

  cudaMalloc(&device_a, size);
  cudaMalloc(&device_b, size);
  cudaMalloc(&device_c, size);

  cudaMemcpy(device_a, host_a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, host_b, size, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;
  printf("blocksPerGrid: %d\n", blocksPerGrid);

  addVec<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c, N);

  cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

  //  for (int i = 0; i < N; i++) 
  // {
  //   printf("Element %i: %.1f\n", i , host_c_cpu[i]);
  // }

  // for (int i = 0; i < N; i++) 
  // {
  //   printf("Element %i: %.1f\n", i , host_c[i]);
  // }
  
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  delete[] host_a; host_a = 0;
  delete[] host_b; host_b = 0;
  delete[] host_c; host_c = 0;
}