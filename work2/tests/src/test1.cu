#include <gtest/gtest.h>
#include "matrix_operators.cuh"
#include <vector>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
using Eigen::MatrixXf;

template <typename T>
class MatrixTest: public ::testing::Test { };
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
  const std::vector<float> sizes = {1, 2, 3};
  const float val = 2.0f;
  for (size_t i = 0; i < sizes.size(); ++i) {
    int m = sizes[i];
    for (size_t j = 0; j < sizes.size(); ++j) {
      int k = sizes[j];
      for (size_t l = 0; l < sizes.size(); ++l) {
        int n = sizes[l];

        Matrix<float> a(m, k);
        Matrix<float> b(k, n);
        a.init(val);
        b.init(val);

        Matrix<float> res = a.view() * b.view();

        MatrixXf eigen_a = MatrixXf::Constant(m, k, val);
        MatrixXf eigen_b = MatrixXf::Constant(k, n, val);
        MatrixXf expected = eigen_a * eigen_b;
        MatrixXf test_val = toEigenMatrix(res, m, n);
        EXPECT_TRUE(test_val.isApprox(expected, 1e-5f));
      }
    }
  }
}

TYPED_TEST(MatrixTest, MultiMatrixBigSize) {
  const std::vector<int> sizes = {127, 128, 129, 512};
  const float val = 3.0f;
  for (size_t i = 0; i < sizes.size(); ++i) {
    int m = sizes[i];
    for (size_t j = 0; j < sizes.size(); ++j) {
      int k = sizes[j];
      for (size_t l = 0; l < sizes.size(); ++l) {
        int n = sizes[l];

        Matrix<float> a(m, k);
        Matrix<float> b(k, n);
        a.init(val);
        b.init(val);

        Matrix<float> res = a.view() * b.view();

        MatrixXf eigen_a = MatrixXf::Constant(m, k, val);
        MatrixXf eigen_b = MatrixXf::Constant(k, n, val);
        MatrixXf expected = eigen_a * eigen_b;
        MatrixXf test_val = toEigenMatrix(res, m, n);
        EXPECT_TRUE(test_val.isApprox(expected, 1e-5f));
      }
    }
  }
}

TYPED_TEST(MatrixTest, SubMatrix) {
  Matrix<float> a(2, 2);
  Matrix<float> b(2, 2);
  Matrix<float> res(2, 2);
  MatrixXf expected = MatrixXf::Constant(2, 2, 0.0f);

  float val = 1.0f;

  a.init(val);
  b.init(val);

  std::vector<float> hostRes(4);

  res = a.view() - b.view();
  res.data().copy_to_host(hostRes.data());

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(hostRes[i*2+j], expected(i, j));
    }
  }
}

