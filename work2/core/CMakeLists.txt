cmake_minimum_required(VERSION 3.10)
project(work2_core LANGUAGES CUDA CXX)

include(../cmake/config/cuda_arch.cmake)

find_package(CUDA REQUIRED)

file(GLOB_RECURSE SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_library(work2_core SHARED ${SRC})
add_library(work2::core ALIAS work2_core)

set_target_properties(work2_core PROPERTIES
    CUDA_RUNTIME_LIBRARY Shared
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/core
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/core
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/core
)

set_target_minimum_cuda_arch(work2_core 75)

target_compile_features(work2_core PRIVATE
    cuda_std_20
)

target_compile_options(work2_core PRIVATE
    -Wall -O3 -Wno-deprecated-gpu-targets
)

target_link_libraries(work2_core PRIVATE
    ${CUDA_LIBRARIES}
)

target_include_directories(work2_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(work2_core PROPERTIES CUDA_ARCHITECTURES "75")
