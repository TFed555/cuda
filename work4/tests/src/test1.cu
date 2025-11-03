#include <gtest/gtest.h>
#include "matrix_operators.cuh"
#include "../strategy/matmul_naive_strategy.cuh"
#include "../strategy/matmul_wmma_strategy.cuh"
#include <vector>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
using Eigen::MatrixXf;

int main() {
  int M = 16;
  int N = 16;
  int K = 16;  
  int hostN_A = M*K;
  int hostN_B = K*N;
  int hostN_C = M*N;

  using Naive = MatmulNaiveStrategy<half, half>;
  using Wmma = MatmulWmmaStrategy<half, float>;

  Matrix<half, Wmma> A(M,K);
  Matrix<half, Wmma> B(K,N);
  Matrix<float, Wmma> C(M,N);
  
  A.fill(1.0f);
  B.fill(10.0f);

  C = A * B;
  std::vector<half> hostA(hostN_A);
  A.data().copy_to_host(hostA.data());
  std::vector<half> hostB(hostN_B);
  B.data().copy_to_host(hostB.data());
  std::vector<float> hostC(hostN_C);
  C.data().copy_to_host(hostC.data());


    std::cout << "Matrix A from GPU:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            std::cout << __half2float(hostA[i * K + j]) << " ";
        }
        std::cout << "\n";
  }

    std::cout << "Matrix B from GPU:\n";
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << __half2float(hostB[i * K + j]) << " ";
        }
        std::cout << "\n";
  }

  std::cout << "Matrix C from GPU:\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << __half2float(hostC[i * K + j]) << " ";
        }
        std::cout << "\n";
  }

  return 0;
}