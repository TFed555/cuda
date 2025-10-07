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
    B.init((float)(rand()) / (float)(rand()));

    for (auto _ : state) {
        C = A.view() * B.view();
        benchmark::DoNotOptimize(C.data());
    }

    std::vector<atom_t> hostA(n * n);
    A.data().copy_to_host(hostA.data());
    std::vector<atom_t> hostB(n * n);
    B.data().copy_to_host(hostB.data());
    std::vector<atom_t> hostC(n * n);
    C.data().copy_to_host(hostC.data());

    std::cout << "Matrix A from GPU:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << hostA[i * n + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Matrix B from GPU:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << hostB[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "Matrix C from GPU:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << hostC[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }
}

BENCHMARK(BENCHMARK_matMul)->Name("matMulGPU")->RangeMultiplier(2)->Range(1 << 4, 1 << 5);

BENCHMARK_MAIN();