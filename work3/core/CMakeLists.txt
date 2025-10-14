cmake_minimum_required(VERSION 3.10)
project(work3_core LANGUAGES CUDA CXX)

include(../cmake/config/cuda_arch.cmake)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF) 
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_library(work3_core INTERFACE)
add_library(work3::core ALIAS work3_core)

target_compile_features(work3_core INTERFACE
    cuda_std_20
)

target_compile_options(work3_core INTERFACE
    -Wall -O3 -Wno-deprecated-gpu-targets
)

target_link_libraries(work3_core INTERFACE
    ${CUDA_LIBRARIES}
)

target_include_directories(work3_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/strategies
    ${CUDA_INCLUDE_DIRS}
)

set_target_properties(work3_core PROPERTIES CUDA_ARCHITECTURES "75")