#include <iostream>
#include "vector_operations.cuh"
#include "../strategy/vecsum_nobr_strategy.cuh"
#include "../strategy/vecsum_br_strategy.cuh"
#include <vector>


//пример
int main() {
  using atom_t = float;
  int M = 2;
  atom_t result;
  Vector<atom_t, VecsumBrStrategy> A(M);
  
  A.fill(1.0f);

  result = A.sum();
  
  std::cout << "Result " << result;

  return 0;
}
