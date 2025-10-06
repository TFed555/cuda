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

#include <iostream>
#include "../include/matrix.cuh"

// допустим, AtomKind == float
int main() {
    using atom_t = float;

    // создаем матрицу 3x3
    Matrix<atom_t> A(3, 3);

    // получаем MatrixView — прямое "окно" на данные
    auto view = A.view();

    // инициализация элементов
    for (std::size_t i = 0; i < view.nrows(); ++i) {
        for (std::size_t j = 0; j < view.ncols(); ++j) {
            view(i, j) = static_cast<float>(i * 10 + j);
        }
    }

    // вывод на экран
    std::cout << "Matrix A:" << std::endl;
    for (std::size_t i = 0; i < view.nrows(); ++i) {
        for (std::size_t j = 0; j < view.ncols(); ++j) {
            std::cout << view(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // создаем еще одну матрицу, разделяющую данные через shared_ptr
    Matrix<atom_t> B = A;

    std::cout << "\nMatrix B (shared data):" << std::endl;
    auto vB = B.view();
    for (std::size_t i = 0; i < vB.nrows(); ++i) {
        for (std::size_t j = 0; j < vB.ncols(); ++j) {
            std::cout << vB(i, j) << " ";
        }
        std::cout << std::endl;
    }

    // изменим один элемент через B
    vB(0, 0) = -99.0f;

    std::cout << "\nAfter modification via B:" << std::endl;
    std::cout << "A(0,0) = " << view(0,0) << " (shared)" << std::endl;

    return 0;
}