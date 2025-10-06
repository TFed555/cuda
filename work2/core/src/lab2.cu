// __global__ void addMultiMatrix(float* x, float* w, float* b, float* res, int rows, int cols, int K) {
//   int i = blockIdx.y * blockDim.y + threadIdx.y;
//   int j = blockIdx.x * blockDim.x + threadIdx.x;
//   if (i < rows && j < cols) {
//     float sum = 0.0f;
//     for (int k = 0; k < K; k++) {
//       sum += x[i * K + k] * w[j * K + k];
//     }
//     res[i * cols + j] = sum + b[j];
//   }
// }

// void addMultiMatrix_gpu(float* x, float* w, float* b, float* res, int rows, int cols, int K) {
//   dim3 block_size(16, 16);
//   dim3 grid_size((cols + block_size.x - 1) / block_size.x, (rows + block_size.y - 1) / block_size.y);
//   addMultiMatrix<<<grid_size, block_size>>>(x, w, b, res, rows, cols, K);
// }
