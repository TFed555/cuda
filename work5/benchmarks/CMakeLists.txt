cmake_minimum_required(VERSION 3.10)
project(work5_benchmarks LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.2
)

FetchContent_MakeAvailable(eigen benchmark)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE CUDA_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work5_benchmarks ${CUDA_SRC})

set_target_minimum_cuda_arch(work5_benchmarks 75)

target_compile_features(work5_benchmarks PRIVATE cxx_std_20)
target_compile_options(work5_benchmarks PRIVATE -Wall -O3)

target_compile_options(work5_benchmarks PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcudafe --diag_suppress=2739>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
)

target_include_directories(work5_benchmarks PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${PROJECT_SOURCE_DIR}/core/utils
    ${benchmark_SOURCE_DIR}/include
    ${benchmark_BINARY_DIR}/include
    ${eigen_SOURCE_DIR}
)

target_link_libraries(work5_benchmarks PRIVATE
    work5::core
    CUDA::cudart
    benchmark::benchmark
)

set_target_properties(work5_benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/benchmarks
)

set_target_properties(work5_benchmarks PROPERTIES CUDA_ARCHITECTURES "75")

add_custom_target(run_work5_benchmarks_json
    COMMAND work5_benchmarks
        --benchmark_format=json > ${CMAKE_BINARY_DIR}/../plots/benchmark_results.json
        --benchmark_out_format=json
    DEPENDS work5_benchmarks
    COMMENT "Running benchmarks with JSON output"
    VERBATIM
)