#include <gtest/gtest.h>
#include "matrix_operators.cuh"
#include "../strategy/matmul_naive_strategy.cuh"
#include "../strategy/matmul_wmma_strategy.cuh"
#include <vector>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
using Eigen::MatrixXf;

using Wmma = MatmulWmmaStrategy<half, float>;

class MatrixTest : public ::testing::Test { };
MatrixXf convertOurMatrix(::Matrix<float, Wmma>& mat, int rows, int cols) {
  std::vector<float> host_data(rows * cols);
  mat.data().copy_to_host(host_data.data());
  
  MatrixXf result(rows, cols);
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < cols; ++j) {
      result(i, j) = host_data[i * cols + j];
    }
  }
  return result;
}

MatrixXf convertEigenMatrix(const Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat) {
  MatrixXf result(mat.rows(), mat.cols());
  for (int i = 0; i < mat.rows(); ++i) {
    for (int j = 0; j < mat.cols(); ++j) {
      result(i, j) = static_cast<float>(mat(i, j));
    }
  }
  return result;
}

TEST_F(MatrixTest, MultiMatrixSmallSize) {
  const std::vector<int> sizes = {16, 32, 64};
  const float val = 2.0f;
    
  for (size_t i = 0; i < sizes.size(); ++i) {
    int m = sizes[i];
    for (size_t j = 0; j < sizes.size(); ++j) {
      int k = sizes[j];
      for (size_t l = 0; l < sizes.size(); ++l) {
        int n = sizes[l];

        ::Matrix<half, Wmma> a(m, k);
        ::Matrix<half, Wmma> b(k, n);

        a.fill(val);
        b.fill(val);

        ::Matrix<float, Wmma> res = a * b;

        Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_a(m, k);
        Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_b(k, n);

        eigen_a.setConstant(Eigen::half(val));
        eigen_b.setConstant(Eigen::half(val));

        auto eigen_res = eigen_a * eigen_b;

        MatrixXf res_final = convertOurMatrix(res, m, n);
        MatrixXf expected = convertEigenMatrix(eigen_res);

        EXPECT_TRUE(res_final.isApprox(expected, 1e-2f));
      }
    }
  }
}

TEST_F(MatrixTest, MultiMatrixBigSize) {
  const std::vector<int> sizes = {128, 256, 512, 1024};
  const float val = 2.0f;
  
  for (size_t i = 0; i < sizes.size(); ++i) {
    int m = sizes[i];
    for (size_t j = 0; j < sizes.size(); ++j) {
      int k = sizes[j];
      for (size_t l = 0; l < sizes.size(); ++l) {
        int n = sizes[l];

        ::Matrix<half, Wmma> a(m, k);
        ::Matrix<half, Wmma> b(k, n);

        a.fill(val);
        b.fill(val);

        ::Matrix<float, Wmma> res = a * b;

        Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_a(m, k);
        Eigen::Matrix<Eigen::half, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> eigen_b(k, n);

        eigen_a.setConstant(Eigen::half(val));
        eigen_b.setConstant(Eigen::half(val));

        auto eigen_res = eigen_a * eigen_b;

        MatrixXf res_final = convertOurMatrix(res, m, n);
        MatrixXf expected = convertEigenMatrix(eigen_res);

        EXPECT_TRUE(res_final.isApprox(expected, 1e-2f));
      }
    }
  }
}