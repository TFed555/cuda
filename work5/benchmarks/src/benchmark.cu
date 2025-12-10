#include "vector_operations.cuh"
#include "../strategy/vecsum_nobr_strategy.cuh"
#include "../strategy/vecsum_br_strategy.cuh"
#include "kernels/kernel_vecred_nobr.cuh"
#include "kinds.h"
#include "benchmark/benchmark.h"
#include <iostream>
#include <random>
#include <cmath>
#include <chrono>
#include <vector>


static void BENCHMARK_VecsumNobrStrategy(benchmark::State& state) {
    auto n = state.range(0);
    using atom_t = float;

    Vector<atom_t, VecsumNobrStrategy> vec(n);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<atom_t> dist(-1.0f, 1.0f);
    
    vec.fill(dist(gen));
    cudaDeviceSynchronize(); 
    
    Data<atom_t> result_device(1);
    atom_t result_host = 0.0f;
    result_device.copy_to_device(&result_host);

    std::size_t m = vec.size();
    constexpr std::size_t block_size = 256;
    std::size_t blocks = (((m + block_size - 1) / block_size) < 128) ? ((m + block_size - 1) / block_size) : 128;
    std::size_t shm = block_size * sizeof(atom_t);

    Data<atom_t> partSum_device(blocks);
    
    for (auto _ : state) {
        kernel_vecred_nobr<<<blocks, block_size, shm>>>(vec.view(), partSum_device.data());
        kernel_vecred_final_nobr<<<1, block_size, shm>>>(partSum_device.data(), result_device.data(), blocks, m);
    }


    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(result_device);
}

BENCHMARK(BENCHMARK_VecsumNobrStrategy)->Name("VecsumNobrStrategy")->RangeMultiplier(2)->Range(8, 8 << 20);

static void BENCHMARK_VecsumBrStrategy(benchmark::State& state) {
    auto n = state.range(0);
    using atom_t = float;

    Vector<atom_t, VecsumBrStrategy> vec(n);
    
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_real_distribution<atom_t> dist(-1.0f, 1.0f);
    
    vec.fill(dist(gen));
    cudaDeviceSynchronize(); 
    
    Data<atom_t> result_device(1);
    atom_t result_host = 0.0f;
    result_device.copy_to_device(&result_host);

    std::size_t m = vec.size();
    constexpr std::size_t block_size = 256;
    std::size_t blocks = (((m + block_size - 1) / block_size) < 128) ? ((m + block_size - 1) / block_size) : 128;
    std::size_t shm = block_size * sizeof(atom_t);

    Data<atom_t> partSum_device(blocks);
    
    for (auto _ : state) {
       kernel_vecred_br<<<blocks, block_size, ((block_size + 31) / 32) * sizeof(atom_t)>>>(vec.view(), result_device.data());
    }

    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(result_device);
}

BENCHMARK(BENCHMARK_VecsumBrStrategy)->Name("VecsumBrStrategy")->RangeMultiplier(2)->Range(8, 8 << 20);

BENCHMARK_MAIN();