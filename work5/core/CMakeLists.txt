cmake_minimum_required(VERSION 3.10)
project(work5_core LANGUAGES CUDA CXX)

include(../cmake/config/cuda_arch.cmake)

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF) 
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS OFF)

add_subdirectory(../../work4 work4_build)

if(NOT TARGET work4::core)
    message(FATAL_ERROR "work4::core target not found")
endif()

add_library(work5_core INTERFACE)
add_library(work5::core ALIAS work5_core)

target_compile_features(work5_core INTERFACE
    cuda_std_20
)

target_compile_options(work5_core INTERFACE
    -Wall -O3 -Wno-deprecated-gpu-targets --generate-code=arch=compute_75,code=sm_75
)

target_link_libraries(work5_core INTERFACE
    work4::core
    ${CUDA_LIBRARIES}
)

target_include_directories(work5_core INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/strategy
    #${CUDA_INCLUDE_DIRS}
)

set_target_properties(work5_core PROPERTIES CUDA_ARCHITECTURES "75")