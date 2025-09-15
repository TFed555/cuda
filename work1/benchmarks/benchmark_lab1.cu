#include "../core/includes/lab1.cuh"
#include "benchmark/include/benchmark/benchmark.h"
#include <iostream>

static void BENCHMARK_addVec_cpu(benchmark::State &state)
{
    const int N = 256;
    size_t size = N * sizeof(float);
    float *a = new float[N];
    float *b = new float[N];
    float *c = new float[N];

    for (int i = 0; i <= N; i++)
        {
            a[i] = i;
            b[i] = i;
        }

    for (auto _ : state)
    {
        addVec_cpu(a, b, c, N);
    }
    delete [] a;
    delete [] b;
    delete [] c;
}

BENCHMARK(BENCHMARK_addVec_cpu);

static void BENCHMARK_addVec(benchmark::State &state)
{
    const N = 256;
    size_t size = N * sizeof(float);
    float* host_a = (float*) malloc(size);
    float* host_b = (float*) malloc(size);
    float* host_c = (float*) malloc(size);

    for (int i = 0; i <= N; i++)
        {
            host_a[i] = i;
            host_b[i] = i;
        }

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

    for (auto _ : state)
    {
        addVec(device_a, device_b, device_c, N);
    }

    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    delete [] a;
    delete [] b;
    delete [] c;
}

BENCHMARK(BENCHMARK_addVec);

BENCHMARK_MAIN();