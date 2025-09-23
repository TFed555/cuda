#include "lab1.cuh"
#include "benchmark/benchmark.h"
#include <iostream>
#include <cmath>
#include <chrono>


//замер без выделения памяти CPU
static void BENCHMARK_addVec_cpu(benchmark::State& state) {
  const int N = 256;

  float *a, *b, *c;
  init_vectors(&a, &b, &c, N);

  for (auto _ : state) {
    addVec_cpu(a, b, c, N);
    benchmark::DoNotOptimize(c);
  }

  free_vectors(a, b, c);
}

BENCHMARK(BENCHMARK_addVec_cpu)->Name("addVecCPU")->RangeMultiplier(2)->Range(1<<10, 1<<22);

//замер без выделения памяти GPU
static void BENCHMARK_addVec(benchmark::State &state)
{
  int N = state.range(0);

  size_t size = N * sizeof(float);

  float *host_a, *host_b, *host_c;
  init_vectors(&host_a, &host_b, &host_c, N);

  float* device_a;
  float* device_b;
  float* device_c;

  copy_vectors(host_a, host_b, host_c, &device_a, &device_b, &device_c, N);


    for (auto _ : state)
    {
        addVec_gpu(device_a, device_b, device_c, N);
        benchmark::DoNotOptimize(device_c);
    }

    cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

    cudafree_vectors(device_a, device_b, device_c);

    free_vectors(host_a, host_b, host_c);
}

BENCHMARK(BENCHMARK_addVec)->Name("addVecGPUCore")->RangeMultiplier(2)->Range(1<<10, 1<<22);

//замер с выделением памяти CPU
// static void BENCHMARK_addVec_cpu2(benchmark::State &state)
// {
//     int N = state.range(0);
//     for (auto _ : state)
//     {
//         float *a, *b, *c;
//         init_vectors(&a, &b, &c, N);
//         addVec_cpu(a, b, c, N); 
//         free_vectors(a, b, c);
//     }

// }

// BENCHMARK(BENCHMARK_addVec_cpu2)->Name("addVec")->Arg(1<<10)->Arg(1<<16)->Arg(1<<25);

//замер с выделением памяти GPU
static void BENCHMARK_addVec2(benchmark::State &state)
{
  int N = state.range(0);
  size_t size = N * sizeof(float);

    for (auto _ : state)
    {
        float *host_a, *host_b, *host_c;
        init_vectors(&host_a, &host_b, &host_c, N);

        float* device_a;
        float* device_b;
        float* device_c;

        copy_vectors(host_a, host_b, host_c, &device_a, &device_b, &device_c, N);

        addVec_gpu(device_a, device_b, device_c, N);
        benchmark::DoNotOptimize(device_c);

        cudaMemcpy(host_c, device_c, size, cudaMemcpyDeviceToHost);

        cudafree_vectors(device_a, device_b, device_c);
        free_vectors(host_a, host_b, host_c);
    }

}

BENCHMARK(BENCHMARK_addVec2)->Name("addVecGPUFull")->RangeMultiplier(2)->Range(1<<10, 1<<22);

// BENCHMARK_MAIN();
int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
}