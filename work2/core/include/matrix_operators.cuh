#ifndef MATRIXOPERATORS_H
#define MATRIXOPERATORS_H

#include "matrix.cuh"
#include "kernel_matsum.cuh"

template <AtomKind AtomT>
Matrix<AtomT> operator+(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b){
  if (a.size() != b.size()) { throw std::runtime_error("Not equal sizes"); }
  Matrix<AtomT> res(a.nrows(), a.ncols());

  dim3 block_size(16, 16);
  dim3 grid_size((a.ncols() + block_size.x - 1) / block_size.x, 
  (a.nrows() + block_size.y - 1) / block_size.y);

  kernel_matrix_sum<<<grid_size, block_size>>>(a, b, res.view(), a.nrows(), a.ncols());

  return res;
}

#endif