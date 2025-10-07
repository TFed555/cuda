#include <iostream>
#include "../include/matrix_operators.cuh"
#include "../include/kernel_init.cuh"
#include <vector>

int main() {
  using atom_t = float;

  Matrix<atom_t> A(4,4);
  Matrix<atom_t> B(4,4);
  Matrix<atom_t> C(4,4);

  dim3 threads(4, 4);
  dim3 blocks(1, 1);
  kernel_matrix_init<<<blocks, threads>>>(A.view(), 1.0f);
  cudaDeviceSynchronize();
  kernel_matrix_init<<<blocks, threads>>>(B.view(), 10.0f);
  cudaDeviceSynchronize();

  C = A.view() * B.view();
    std::vector<atom_t> hostA(16);
  A.data().copy_to_host(hostA.data());
    std::vector<atom_t> hostB(16);
  B.data().copy_to_host(hostB.data());
  std::vector<atom_t> hostC(16);
  C.data().copy_to_host(hostC.data());

    std::cout << "Matrix A from GPU:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << hostA[i * 4 + j] << " ";
        }
        std::cout << "\n";
  }

    std::cout << "Matrix B from GPU:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << hostB[i * 4 + j] << " ";
        }
        std::cout << "\n";
  }

  std::cout << "Matrix C from GPU:\n";
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << hostC[i * 4 + j] << " ";
        }
        std::cout << "\n";
  }
  return 0;
}
