#pragma once
#include <cuda_runtime.h>

inline dim3 make_grid_2d(size_t nrows, size_t ncols, dim3 block = dim3(16, 16)) {
    return dim3(
        (ncols + block.x - 1) / block.x,
        (nrows + block.y - 1) / block.y
    );
}

inline dim3 make_wmma_grid_2d(size_t nrows, size_t ncols,
                              dim3 block = dim3(128, 4),
                              int WMMA_M = 16, int WMMA_N = 16)
{
    int warps_per_block_x = block.x / 32;
    int warps_per_block_y = block.y;

    int tiles_x = (ncols + WMMA_N - 1) / WMMA_N;
    int tiles_y = (nrows + WMMA_M - 1) / WMMA_M;

    int grid_x = (tiles_x + warps_per_block_x - 1) / warps_per_block_x;
    int grid_y = (tiles_y + warps_per_block_y - 1) / warps_per_block_y;

    return dim3(grid_x, grid_y);
}
