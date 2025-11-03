#ifndef KINDS_H
#define KINDS_H

#include <concepts>
#include <cuda_fp16.h>

template <typename T>
concept AtomKind = std::integral<T> || std::floating_point<T> || std::is_same_v<half, T>;

// template <typename M>
// concept MatrixKind = requires(M mtx, size_t i, size_t j) {
//   //дописать
// };

#endif