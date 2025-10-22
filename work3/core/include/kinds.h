#ifndef KINDS_H
#define KINDS_H

#include <concepts>

template <typename T>
concept AtomKind = std::integral<T> || std::floating_point<T>;

// template <typename M>
// concept MatrixKind = requires(M mtx, size_t i, size_t j) {
//   //дописать
// };

#endif