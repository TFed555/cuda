#include "matrix_operators.cuh"
#include "benchmark/benchmark.h"
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>

#define EIGEN_NO_CUDA
#include <Eigen/Dense>
using Eigen::MatrixXf;

// Замер без выделения памяти CPU
static void BENCHMARK_matMul_cpu(benchmark::State& state) {
    auto n = state.range(0);

    MatrixXf a = MatrixXf::Random(n, n);
    MatrixXf b = MatrixXf::Random(n, n);
    MatrixXf c(n, n);

    for (auto _ : state) {
        c = a * b;
        benchmark::DoNotOptimize(c.data());
    }
}

BENCHMARK(BENCHMARK_matMul_cpu)->Name("matMulCPU")->RangeMultiplier(2)->Range(1 << 4, 1 << 10);

// Замер без выделения памяти GPU
static void BENCHMARK_matMul(benchmark::State& state) {
    auto n = state.range(0);
    using atom_t = float;

    srand(time(0));
    Matrix<atom_t> A(n, n);
    Matrix<atom_t> B(n, n);
    Matrix<atom_t> C(n, n);

    A.init((float)(rand()) / (float)(rand()));
    cudaDeviceSynchronize();
    B.init((float)(rand()) / (float)(rand()));
    cudaDeviceSynchronize();

    for (auto _ : state) {
        C = A.view() * B.view();
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(C.data());
    }
}

BENCHMARK(BENCHMARK_matMul)->Name("matMulGPU")->RangeMultiplier(2)->Range(1 << 4, 1 << 10);

BENCHMARK_MAIN();