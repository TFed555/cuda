#include <gtest/gtest.h>
#include "lab1.cuh"

TEST(VecTests, InitVectors) {
    float *a, *b, *c;
    const int N = 1000000;
    //size_t size = N * sizeof(float);
    init_vectors(&a, &b, &c, N);
    
    // ASSERT_NE(a, nullptr);
    // ASSERT_NE(b, nullptr);
    // ASSERT_NE(c, nullptr);
    
    for (int i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(a[i], (float)i);
      EXPECT_FLOAT_EQ(b[i], (float)i);
    }
    // EXPECT_FLOAT_EQ(a[N-1], (float)(N-1));
    // EXPECT_FLOAT_EQ(b[N-1], (float)(N-1));
    
    free_vectors(a, b, c);
}

TEST(VecTests, AddCPU) {
    float *a, *b, *c;
    const int N = 1000;
    
    init_vectors(&a, &b, &c, N);
    
    addVec_cpu(a, b, c, N);
    
    for (int i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(c[i], a[i] + b[i]);
    }
    //EXPECT_FLOAT_EQ(c[2], a[2] + b[2]);
    EXPECT_FLOAT_EQ(a[2] + b[2], 4.0f);
    free_vectors(a, b, c);
}

TEST(VecTests, AddGPUandCPU) {
    float *host_a, *host_b, *host_c;
    float *device_a, *device_b, *device_c;
    const int N = 1024;
    
    init_vectors(&host_a, &host_b, &host_c, N);
    
    copy_vectors(host_a, host_b, host_c, &device_a, &device_b, &device_c, N);
    
    int threadsBlock = 128;
    int blocksGrid = (N + threadsBlock - 1) / threadsBlock;
    addVec<<<blocksGrid, threadsBlock>>>(device_a, device_b, device_c, N);
    cudaDeviceSynchronize();
    
    float* gpu_result = new float[N];
    cudaMemcpy(gpu_result, device_c, N * sizeof(float), cudaMemcpyDeviceToHost);
    
    float* cpu_result = new float[N];
    addVec_cpu(host_a, host_b, cpu_result, N);
    
    for (int i = 0; i < N; i++) {
      EXPECT_FLOAT_EQ(gpu_result[i], cpu_result[i]);
    }
    
    delete[] gpu_result;
    delete[] cpu_result;
    cudafree_vectors(device_a, device_b, device_c);
    free_vectors(host_a, host_b, host_c);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}