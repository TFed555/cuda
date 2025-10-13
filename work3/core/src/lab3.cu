#include <iostream>
#include "../include/matrix_operators.cuh"
#include <vector>


//пример
int main() {
  using atom_t = float;
  int M = 10;
  int N = 8;
  int K = 5;  
  int hostN_A = M*K;
  int hostN_B = K*N;
  int hostN_C = M*N;

  Matrix<atom_t> A(M,K);
  Matrix<atom_t> B(K,N);
  Matrix<atom_t> C(M,N);
  
  A.fill(1.0f);
  B.fill(10.0f);

  C = A.view() * B.view();
  std::vector<atom_t> hostA(hostN_A);
  A.data().copy_to_host(hostA.data());
  std::vector<atom_t> hostB(hostN_B);
  B.data().copy_to_host(hostB.data());
  std::vector<atom_t> hostC(hostN_C);
  C.data().copy_to_host(hostC.data());

    std::cout << "Matrix A from GPU:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << hostA[i * K + j] << " ";
        }
        std::cout << "\n";
  }

    std::cout << "Matrix B from GPU:\n";
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << hostB[i * K + j] << " ";
        }
        std::cout << "\n";
  }

  std::cout << "Matrix C from GPU:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << hostC[i * K + j] << " ";
        }
        std::cout << "\n";
  }

  return 0;
}
