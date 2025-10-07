#pragma once
#include <cuda_runtime.h>

inline dim3 make_grid_2d(size_t nrows, size_t ncols, dim3 block = dim3(16, 16)) {
    return dim3(
        (ncols + block.x - 1) / block.x,
        (nrows + block.y - 1) / block.y
    );
}
