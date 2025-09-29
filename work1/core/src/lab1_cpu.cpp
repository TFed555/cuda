#include "lab1.h"

void addVec_cpu(float* a, float* b, float* c, int N) {
  for(int i = 0; i < N; i++) {
    c[i] = a[i] + b[i];
  }
}