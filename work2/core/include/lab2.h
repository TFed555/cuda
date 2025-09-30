#define LAB2_H

#include <cstdlib>
#include <cuda_runtime.h>

__global__ void addMultiMatrix(float* x, float* w, float* b, float* res, int rows, int cols, int K);

void addMultiMatrix_cpu(float* x, float* w, float* b, float* res, int rows, int cols, int K);

void addMultiMatrix_gpu(float* x, float* w, float* b, float* res, int rows, int cols, int K);

enum class MatType {
    Host,
    Device
};

template<typename T>
class Matrix {
private:
    struct Mat {
        T* ptr;
        MatType type;
    };
    Mat matrix;
    size_t rows;
    size_t cols;
    size_t total_size;
    size_t byte_size;

public:
    Matrix(size_t rows, size_t cols);
    Matrix(size_t rows, size_t cols, bool cudaDevice);
    
    ~Matrix();
    
    T& operator()(size_t i, size_t j);
    const T& operator()(size_t i, size_t j) const;

    T* ptr() { return matrix.ptr; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }
    size_t getTotalSize() const { return total_size; }

    void init_matrix();
    void init_matrix_value(T value);
    void copy_to_host(const Matrix<T>& device_mat);
    void copy_to_device(Matrix<T>& device_mat) const;
};

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols) 
    : rows(rows), cols(cols), total_size(rows * cols), byte_size(total_size * sizeof(T)) {
    matrix.ptr = static_cast<T*>(malloc(byte_size));
    matrix.type = MatType::Host;
}

template<typename T>
Matrix<T>::Matrix(size_t rows, size_t cols, bool cudaDevice) 
    : rows(rows), cols(cols), total_size(rows * cols), byte_size(total_size * sizeof(T)) {
    T* devptr = nullptr;
    cudaMalloc(&devptr, byte_size);
    matrix.ptr = devptr;
    matrix.type = MatType::Device;
}

template<typename T>
Matrix<T>::~Matrix() {
  if (matrix.ptr) {
    if (matrix.type == MatType::Host) {
      free(matrix.ptr);
    } else {
      cudaFree(matrix.ptr);
    }
  }
}

template<typename T>
T& Matrix<T>::operator()(size_t i, size_t j) {
    return matrix.ptr[i * cols + j];
}

template<typename T>
const T& Matrix<T>::operator()(size_t i, size_t j) const {
    return matrix.ptr[i * cols + j];
}

template<typename T>
void Matrix<T>::init_matrix() {
    if (matrix.type != MatType::Host) {
        return;
    }
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            (*this)(i, j) = static_cast<T>(i * cols + j);
        }
    }
}

template<typename T>
void Matrix<T>::init_matrix_value(T value) {
    if (matrix.type != MatType::Host) {
        return;
    }
    for (size_t i = 0; i < total_size; i++) {
        matrix.ptr[i] = value;
    }
}

template<typename T>
void Matrix<T>::copy_to_host(const Matrix<T>& device_mat) {
    if (matrix.type == MatType::Host && device_mat.matrix.type == MatType::Device) {
        cudaMemcpy(matrix.ptr, device_mat.matrix.ptr, byte_size, cudaMemcpyDeviceToHost);
    }
}

template<typename T>
void Matrix<T>::copy_to_device(Matrix<T>& device_mat) const {
    if (matrix.type == MatType::Host && device_mat.matrix.type == MatType::Device) {
        cudaMemcpy(device_mat.matrix.ptr, matrix.ptr, byte_size, cudaMemcpyHostToDevice);
    }
}