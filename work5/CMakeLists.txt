cmake_minimum_required(VERSION 3.10)
project(work5 LANGUAGES CUDA CXX)

option(ENABLE_TESTS "Enable tests" ON)
option(ENABLE_BENCH "Enable benchmarks" ON)

add_subdirectory(core)

if(ENABLE_TESTS)
    add_subdirectory(tests)
endif()

