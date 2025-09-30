cmake_minimum_required(VERSION 3.10)
project(work2_benchmarks LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.2
)
FetchContent_MakeAvailable(benchmark)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cu
)

add_executable(work2_benchmarks ${SRC})

target_compile_features(work2_benchmarks PRIVATE cxx_std_20)
target_compile_options(work2_benchmarks PRIVATE -Wall -O3)

target_include_directories(work2_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${benchmark_SOURCE_DIR}/include
    ${benchmark_BINARY_DIR}/include
)

target_link_libraries(work2_benchmarks PRIVATE
    work2::core
    CUDA::cudart
    benchmark::benchmark
)

set_target_properties(work2_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)

add_custom_target(run_benchmarks_json
    COMMAND work1_benchmarks
        --benchmark_format=json > ${CMAKE_BINARY_DIR}/../plots/benchmark_results.json
        --benchmark_out_format=json
    DEPENDS work1_benchmarks
    COMMENT "Running benchmarks with JSON output"
    VERBATIM
)