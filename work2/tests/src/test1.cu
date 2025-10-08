#include <gtest/gtest.h>
#include "matrix_operators.cuh"

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

  res = a.view() + b.view();

  EXPECT_FLOAT_EQ(res.view()(0, 0), expected(0, 0));
}