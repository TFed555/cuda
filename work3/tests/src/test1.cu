#include <gtest/gtest.h>
#include "matrix_operators.cuh"
#include "../strategy/matmul_naive_strategy.cuh"
#include "../strategy/matmul_shmem_strategy.cuh"
#include <vector>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
using Eigen::MatrixXf;

template <typename T>
class MatrixTest : public ::testing::Test { };
using MatrixTypes = ::testing::Types<Matrix<float>>;

TYPED_TEST_SUITE(MatrixTest, MatrixTypes);

MatrixXf toEigenMatrix(Matrix<float>& mat, int rows, int cols) {
    std::vector<float> host_data(rows * cols);
    mat.data().copy_to_host(host_data.data());
    
    MatrixXf eigen_mat(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            eigen_mat(i, j) = host_data[i * cols + j];
        }
    }
    return eigen_mat;
}

TYPED_TEST(MatrixTest, MultiMatrixSmallSize) {
  MatmulNaiveStrategy<float> naive;
  MatmulShmemStrategy<float> shmem;
  std::vector<MatmulStrategy<float>*> strategies = {&naive, &shmem};

  const std::vector<float> sizes = {1, 2, 3};
  const float val = 2.0f;
  for (auto* strategy : strategies) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      int m = sizes[i];
      for (size_t j = 0; j < sizes.size(); ++j) {
        int k = sizes[j];
        for (size_t l = 0; l < sizes.size(); ++l) {
          int n = sizes[l];

          Matrix<float> a(m, k);
          Matrix<float> b(k, n);
          a.fill(val);
          b.fill(val);

          a.set_strategy(strategy);

          Matrix<float> res = a * b;

          MatrixXf eigen_a = MatrixXf::Constant(m, k, val);
          MatrixXf eigen_b = MatrixXf::Constant(k, n, val);
          MatrixXf expected = eigen_a * eigen_b;
          MatrixXf test_val = toEigenMatrix(res, m, n);

          EXPECT_TRUE(test_val.isApprox(expected, 1e-5f));
        }
      }
    }
  }
}

TYPED_TEST(MatrixTest, MultiMatrixBigSize) {
  MatmulNaiveStrategy<float> naive;
  MatmulShmemStrategy<float> shmem;
  std::vector<MatmulStrategy<float>*> strategies = {&naive, &shmem};

  const std::vector<float> sizes = {127, 128, 129, 512};
  const float val = 2.0f;
  for (auto* strategy : strategies) {
    for (size_t i = 0; i < sizes.size(); ++i) {
      int m = sizes[i];
      for (size_t j = 0; j < sizes.size(); ++j) {
        int k = sizes[j];
        for (size_t l = 0; l < sizes.size(); ++l) {
          int n = sizes[l];

          Matrix<float> a(m, k);
          Matrix<float> b(k, n);
          a.fill(val);
          b.fill(val);

          a.set_strategy(strategy);

          Matrix<float> res = a * b;

          MatrixXf eigen_a = MatrixXf::Constant(m, k, val);
          MatrixXf eigen_b = MatrixXf::Constant(k, n, val);
          MatrixXf expected = eigen_a * eigen_b;
          MatrixXf test_val = toEigenMatrix(res, m, n);

          EXPECT_TRUE(test_val.isApprox(expected, 1e-5f));
        }
      }
    }
  }
}