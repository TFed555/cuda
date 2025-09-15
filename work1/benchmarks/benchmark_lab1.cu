#include "../core/includes/lab1.cuh"
#include "benchmark/include/benchmark/benchmark.h"
#include <iostream>

//замер без выделения памяти CPU
static void BENCHMARK_addVec_cpu(benchmark::State &state)
{
    const int N = 256;

    float *a, *b, *c;
    init_vectors(&a, &b, &c, N);

    for (auto _ : state)
    {
        addVec_cpu(a, b, c, N);
    }
    
    free_vectors(a, b, c);
}

BENCHMARK(BENCHMARK_addVec_cpu);

//замер без выделения памяти GPU
static void BENCHMARK_addVec(benchmark::State &state)
{
  const int N = 256;
  size_t size = N * sizeof(float);

  float *host_a, *host_b, *host_c;
  init_vectors(&host_a, &host_b, &host_c, N);

  float* device_a;
  float* device_b;
  float* device_c;

  copy_vectors(host_a, host_b, host_c, &device_a, &device_b, &device_c, N);

  int threadsPerBlock = 256;
  int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

    for (auto _ : state)
    {
        addVec<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c, N);
    }

    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    cudafree_vectors(device_a, device_b, device_c);
    // cudaFree(device_a);
    // cudaFree(device_b);
    // cudaFree(device_c);

    free_vectors(host_a, host_b, host_c);
    // free(host_a);
    // free(host_b);
    // free(host_c);
}

BENCHMARK(BENCHMARK_addVec);

//замер с выделением памяти CPU
static void BENCHMARK_addVec_cpu2(benchmark::State &state)
{
    const int N = 256;

    for (auto _ : state)
    {
        float *a, *b, *c;
        init_vectors(&a, &b, &c, N);
        addVec_cpu(a, b, c, N); 
        free_vectors(a, b, c);
    }

}

BENCHMARK(BENCHMARK_addVec_cpu2);

//замер с выделением памяти GPU
static void BENCHMARK_addVec2(benchmark::State &state)
{
  const int N = 256;
  size_t size = N * sizeof(float);

    for (auto _ : state)
    {
        float *host_a, *host_b, *host_c;
        init_vectors(&host_a, &host_b, &host_c, N);

        float* device_a;
        float* device_b;
        float* device_c;

        copy_vectors(host_a, host_b, host_c, &device_a, &device_b, &device_c, N);

        int threadsPerBlock = 256;
        int blocksPerGrid =(N + threadsPerBlock - 1) / threadsPerBlock;

        addVec<<<blocksPerGrid, threadsPerBlock>>>(device_a, device_b, device_c, N);

        cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

        cudafree_vectors(device_a, device_b, device_c);
        free_vectors(host_a, host_b, host_c);
    }

    // cudaFree(device_a);
    // cudaFree(device_b);
    // cudaFree(device_c);

    // free(host_a);
    // free(host_b);
    // free(host_c);
}

BENCHMARK(BENCHMARK_addVec2);

BENCHMARK_MAIN();