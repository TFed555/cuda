#include <gtest/gtest.h>
#include "vector_operations.cuh"
#include "../strategy/vecsum_nobr_strategy.cuh"
#include "../strategy/vecsum_br_strategy.cuh"
#include <vector>
#include <random>
#define EIGEN_NO_CUDA
#include <Eigen/Dense>
using Eigen::VectorXf;

class VectorTest : public ::testing::Test { };

TEST_F(VectorTest, VecsumNobrStrategy) {
    const std::vector<int> sizes = {1, 2, 3, 127, 129, 512, 541, 1037};
    
    for (int i : sizes) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

      float rand_const = dist(gen);
      
      Vector<float, VecsumNobrStrategy> vec(i);
      vec.fill(rand_const);
      
      VectorXf eigen_vec = VectorXf::Constant(i, rand_const);
      
      EXPECT_NEAR(vec.sum(), eigen_vec.sum(), 1e-2f);
    }
}

TEST_F(VectorTest, VecsumBrStrategy) {
    const std::vector<int> sizes = {1, 2, 3, 127, 129, 512, 541, 1037};
    
    for (int i : sizes) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

      float rand_const = dist(gen);
      
      Vector<float, VecsumBrStrategy> vec(i);
      vec.fill(rand_const);
      
      VectorXf eigen_vec = VectorXf::Constant(i, rand_const);
      
      EXPECT_NEAR(vec.sum(), eigen_vec.sum(), 1e-2f);
    }
}