#include <gtest/gtest.h>
#include "lab1.h"

TEST(VecTests, InitVectors) {
    const int N = 4000000;

    Vector<float> h_a(N);
    Vector<float> h_b(N);
    Vector<float> h_c(N);

    h_a.init_vector();
    h_b.init_vector();
    
    for (int i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(h_a[i], (float)i);
      EXPECT_FLOAT_EQ(h_b[i], (float)i);
    }
}

TEST(VecTests, AddCPU) {
    const int N = 1000;

    Vector<float> h_a(N);
    Vector<float> h_b(N);
    Vector<float> h_c(N);

    h_a.init_vector();
    h_b.init_vector();
    
    addVec_cpu(h_a.ptr(), h_b.ptr(), h_c.ptr(), N);
    
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(h_c[i], h_a[i] + h_b[i]);
    }
    //EXPECT_FLOAT_EQ(c[2], a[2] + b[2]);
    EXPECT_FLOAT_EQ(h_a[2] + h_b[2], 4.0f);
}

TEST(VecTests, AddGPUandCPU) {
  const int N = 256;
  
  Vector<float> h_a(N);
  Vector<float> h_b(N);
  Vector<float> h_c(N);

  h_a.init_vector();
  h_b.init_vector();

  Vector<float> d_a(N, true);
  Vector<float> d_b(N, true);
  Vector<float> d_c(N, true);

  d_a.copy_to_device(h_a.ptr());
  d_b.copy_to_device(h_b.ptr());

  addVec_gpu(d_a.ptr(), d_b.ptr(), d_c.ptr(), N);

  d_c.copy_to_host(h_c.ptr());

  float* cpu_result = new float[N];
  addVec_cpu(h_a.ptr(), h_b.ptr(), cpu_result, N);

  for (size_t i = 0; i < N; i++) {
    EXPECT_NEAR(h_c[i], cpu_result[i], 1e-5);
  }

  delete[] cpu_result;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
