#include "lab2.h"

void addMultiMatrix_cpu(float* x, float* w, float* b, float* res, int rows, int cols, int K) {
   for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++) {
        sum += x[i * K + k] * w[j * K + k];
      }
      res[i * cols + j] = sum + b[j];
    }
  }
}