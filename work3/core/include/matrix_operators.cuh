#ifndef MATRIXOPERATORS_CUH
#define MATRIXOPERATORS_CUH

#include "matrix.cuh"
#include "kernels/kernel_matsum.cuh"
#include "kernels/kernel_matmul_naive.cuh"  
#include "kernels/kernel_matmul_scalar.cuh"
#include "kernels/kernel_matmul_shmem.cuh"
#include "../utils/cuda_utils.cuh"

template <AtomKind AtomT>
Matrix<AtomT> operator+(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b){
  if (a.size() != b.size()) { throw std::runtime_error("Not equal sizes"); }
  Matrix<AtomT> res(a.nrows(), a.ncols());

  dim3 block_size(16, 16);
  dim3 grid_size = make_grid_2d(a.nrows(), a.ncols(), block_size);

  kernel_matrix_sum<<<grid_size, block_size>>>(a, b, res.view());
  cudaDeviceSynchronize();

  return res;
}

template <AtomKind AtomT>
Matrix<AtomT> operator*(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b){
  if (a.ncols() != b.nrows()) { throw std::runtime_error("Can't multiply"); }
  Matrix<AtomT> res(a.nrows(), b.ncols());

  dim3 block_size(16, 16);
  dim3 grid_size = make_grid_2d(a.nrows(), a.ncols(), block_size);

  kernel_matmul_shmem<<<grid_size, block_size>>>(a, b, res.view());
  cudaDeviceSynchronize();

  return res;
}

template <AtomKind AtomT>
Matrix<AtomT> operator*(const MatrixView<AtomT>& a, const AtomT val){
  Matrix<AtomT> res(a.nrows(), a.ncols());

  dim3 block_size(16, 16);
  dim3 grid_size = make_grid_2d(a.nrows(), a.ncols(), block_size);

  kernel_matmul_scalar<<<grid_size, block_size>>>(a, val, res.view());
  cudaDeviceSynchronize();

  return res;
}

template <AtomKind AtomT>
Matrix<AtomT> operator-(const MatrixView<AtomT>& a, const MatrixView<AtomT>& b){
  if (a.size() != b.size()) { throw std::runtime_error("Not equal sizes"); }
  Matrix<AtomT> res(a.nrows(), a.ncols());
  Matrix<AtomT> b_neg = b * static_cast<AtomT>(-1);
  
  dim3 block_size(16, 16);
  dim3 grid_size = make_grid_2d(a.nrows(), a.ncols(), block_size);
  
  kernel_matrix_sum<<<grid_size, block_size>>>(a, b_neg.view(), res.view());
  cudaDeviceSynchronize();

  return res;
}

#endif