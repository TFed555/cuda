#include "matrix_operators.cuh"
#include "benchmark/benchmark.h"
#include <iostream>
#include <cmath>
#include <chrono>

/замер без выделения памяти CPU
static void BENCHMARK_addVec_cpu(benchmark::State& state) {
  const int N = state.range(0);

  Vector<float> h_a(N);
  Vector<float> h_b(N);
  Vector<float> h_c(N);

  h_a.init_vector();
  h_b.init_vector();

  for (auto _ : state) {
    addVec_cpu(h_a.ptr(), h_b.ptr(), h_c.ptr(), N);
    benchmark::DoNotOptimize(h_c.ptr());
  }
}

BENCHMARK(BENCHMARK_addVec_cpu)->Name("addVecCPU")->RangeMultiplier(2)->Range(1<<10, 1<<22);

//замер без выделения памяти GPU
static void BENCHMARK_addVec(benchmark::State &state)
{
  int N = state.range(0);

  Vector<float> h_a(N);
  Vector<float> h_b(N);
  Vector<float> h_c(N);

  h_a.init_vector();
  h_b.init_vector();

  Vector<float> d_a(N, true);
  Vector<float> d_b(N, true);
  Vector<float> d_c(N, true);

  d_a.copy_to_device(h_a.ptr());
  d_b.copy_to_device(h_b.ptr());

    for (auto _ : state)
    {
        addVec_gpu(d_a.ptr(), d_b.ptr(), d_c.ptr(), N);
        benchmark::DoNotOptimize(d_c.ptr());
    }

}

BENCHMARK(BENCHMARK_addVec)->Name("addVecGPUCore")->RangeMultiplier(2)->Range(1<<10, 1<<22);


BENCHMARK_MAIN();