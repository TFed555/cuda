cmake_minimum_required(VERSION 3.10)
project(work4_tests LANGUAGES CXX CUDA)

include(FetchContent)

FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
)
FetchContent_MakeAvailable(googletest eigen)

find_package(CUDAToolkit REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
)

add_executable(work4_tests ${SRC})


target_compile_features(work4_tests PRIVATE cxx_std_20)
target_compile_options(work4_tests PRIVATE -Wall -O3)

target_include_directories(work4_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_BINARY_DIR}/include
    ${eigen_SOURCE_DIR}
)

target_link_libraries(work4_tests PRIVATE
    work4::core
    CUDA::cudart
    GTest::gtest_main
)

enable_testing()
add_test(NAME work4_tests COMMAND work4_tests)

set_target_properties(work4_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)

set_target_properties(work4_core PROPERTIES CUDA_ARCHITECTURES "75")