cmake_minimum_required(VERSION 3.10)
project(work2_tests LANGUAGES CXX CUDA)

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

set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/test1.cu
)

add_executable(work2_tests ${SRC})

set_target_minimum_cuda_arch(work2_tests 75)

target_compile_features(work2_tests PRIVATE cxx_std_20)
target_compile_options(work2_tests PRIVATE -Wall -O3)

target_include_directories(work2_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/core/include
    ${gtest_SOURCE_DIR}/include
    ${gtest_BINARY_DIR}/include
    ${eigen_SOURCE_DIR}
)

target_link_libraries(work2_tests PRIVATE
    GTest::gtest_main
    work2::core
    CUDA::cudart
)

enable_testing()
add_test(NAME work2_tests COMMAND work2_tests)

set_target_properties(work2_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/tests
)
