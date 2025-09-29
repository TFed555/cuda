#include "lab1.h"

__global__ void addVec(float* a, float* b, float* c, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i<N) {
   c[i] = a[i] + b[i];
  }
}

void addVec_gpu(float* a, float* b, float* c, int N) {
  int threads_block = 128;
  int blocks_grid = (N + threads_block - 1) / threads_block;
  addVec<<<blocks_grid, threads_block>>>(a, b, c, N);
}