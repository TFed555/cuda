#include <iostream>
#include "vector_operations.cuh"
#include "../strategy/vecsum_nobr_strategy.cuh"
#include <vector>


//пример
int main() {
  using atom_t = float;
  int M = 3;
  VecsumNobrStrategy nobr;
  atom_t result;
  Vector<atom_t, nobr> A(M);
  
  A.fill(1.0f);

  result = A.sum();
  
  std::cout << result;

  return 0;
}
