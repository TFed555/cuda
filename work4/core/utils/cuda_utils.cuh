#pragma once
#include <cuda_runtime.h>
namespace cuda_utils {
inline dim3 make_grid_2d(size_t nrows, size_t ncols, dim3 block = dim3(16, 16)) {
    return dim3(
        (ncols + block.x - 1) / block.x,
        (nrows + block.y - 1) / block.y
    );
}

inline std::pair<dim3, dim3> make_wmma_grid_block_2d(size_t matrix_rows, size_t matrix_cols)
{
        constexpr size_t warp_size = 32;
        constexpr size_t wmma_tile_size = 16;
        
        size_t warps_in_m = (matrix_rows + wmma_tile_size - 1) / wmma_tile_size;
        size_t warps_in_n = (matrix_cols + wmma_tile_size - 1) / wmma_tile_size;
        
        dim3 block_dim(128, 4);
        
        size_t warps_per_block_x = block_dim.x / warp_size;
        size_t warps_per_block_y = block_dim.y;
        
        dim3 grid_dim((warps_in_m + warps_per_block_x - 1) / warps_per_block_x,
                    (warps_in_n + warps_per_block_y - 1) / warps_per_block_y);
    
        return {grid_dim, block_dim};
}
}
