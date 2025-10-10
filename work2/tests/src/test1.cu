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

TYPED_TEST(MatrixTest, SumMatrix) {
  Matrix<float> a(2, 2);
  Matrix<float> b(2, 2);
  Matrix<float> res(2, 2);
  MatrixXf expected = MatrixXf::Constant(2, 2, 2.0f);

  float val = 1.0f;

  a.init(val);
  b.init(val);

  std::vector<float> hostRes(4);

  res = a.view() + b.view();
  res.data().copy_to_host(hostRes.data());

  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      EXPECT_EQ(hostRes[i*2+j], expected(i, j));
    }
  }
}