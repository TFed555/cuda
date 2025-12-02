#include "vector_operations.cuh"
#include "../strategy/vecsum_nobr_strategy.cuh"
#include "../strategy/vecsum_br_strategy.cuh"
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
    
    atom_t result = 0.0f;
    
    for (auto _ : state) {
        result = vec.sum();
    }

    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(result);
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
    
    atom_t result = 0.0f;
    
    for (auto _ : state) {
        result = vec.sum();
    }

    cudaDeviceSynchronize();
    benchmark::DoNotOptimize(result);
}

BENCHMARK(BENCHMARK_VecsumBrStrategy)->Name("VecsumBrStrategy")->RangeMultiplier(2)->Range(8, 8 << 20);

BENCHMARK_MAIN();