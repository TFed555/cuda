#include <iostream>
#include "../include/matrix_operators.cuh"
#include <vector>

int main() {
  using atom_t = float;

  Matrix<atom_t> A(4,4);
  Matrix<atom_t> B(4,4);
  Matrix<atom_t> C(4,4);
  
  A.fill(1.0f);
  B.fill(10.0f);

  C = A.view() * B.view();
  std::vector<atom_t> hostA(64);
  A.data().copy_to_host(hostA.data());
  std::vector<atom_t> hostB(64);
  B.data().copy_to_host(hostB.data());
  std::vector<atom_t> hostC(64);
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
