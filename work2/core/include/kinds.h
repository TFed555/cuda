#ifndef KINDS_H
#define KINDS_H

#include <concepts>

template <typename T>
concept AtomKind = std::convertible_to<T, double> || std::convertible_to<T, float> ||
   std::same_as<T, float> || std::same_as<T, double>;

// template <typename M>
// concept MatrixKind = requires(M mtx, size_t i, size_t j) {
//   //дописать
// };

#endif