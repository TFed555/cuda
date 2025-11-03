#ifndef MATRIXOPERATORS_CUH
#define MATRIXOPERATORS_CUH

#include "matrix.cuh"
#include "../utils/cuda_utils.cuh"

template <AtomKind AtomA, typename Strategy>
Matrix<typename Strategy::Output, Strategy> operator*(const Matrix<AtomA, Strategy>& a, const Matrix<AtomA, Strategy>& b){
  if (a.ncols() != b.nrows()) { throw std::runtime_error("Can't multiply"); }
  using AtomC = typename Strategy::Output;
  Matrix<AtomC, Strategy> res(a.nrows(), b.ncols());

  Strategy strat;
  strat.multiply(a.view(), b.view(), res.view());

  cudaError_t errSync  = cudaGetLastError();
  if (errSync != cudaSuccess) {
      printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
  }

  return res;
}


#endif