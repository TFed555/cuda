#include "matrix_operators.cuh"
#include "kernels/kernel_matmul_shmem.cuh"
#include "cuda_utils.cuh"
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

    A.fill((float)(rand()) / (float)(rand()));
    cudaDeviceSynchronize();
    B.fill((float)(rand()) / (float)(rand()));
    cudaDeviceSynchronize();

    dim3 block_size(16, 16);
    dim3 grid_size = make_grid_2d(C.nrows(), C.ncols(), block_size);

    for (auto _ : state) {
        // C = A.view() * B.view();
        kernel_matmul_shmem<<<grid_size, block_size>>>(A.view(), B.view(), C.view());
        cudaDeviceSynchronize();
        benchmark::DoNotOptimize(C.data());
    }
    
}

BENCHMARK(BENCHMARK_matMul)->Name("matMulGPU")->RangeMultiplier(2)->Range(1 << 4, 1 << 10);

BENCHMARK_MAIN();