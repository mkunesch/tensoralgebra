#ifndef _TENSORALGEBRA_TENSOROPERATIONS_HPP
#define _TENSORALGEBRA_TENSOROPERATIONS_HPP

#include "TensorExpression.hpp"

// Defines the outer product using expression templates
#include "Outer.hpp"

// Defines the dot product using expression templates
#include "Dot.hpp"

namespace tensoralgebra {
/// Computes the trace of a 2-tensor with lower inverse given an inverse metric
// Always returns an evaluated expression so it is safe to take a const &
template <class T, size_t N>
auto trace(const TensorExpression<2, T, N> &tensor_LL,
           const TensorExpression<2, T, N> &inverse_metric) {
  return trace(dot(inverse_metric, tensor_LL));
}

/// Computes the trace of a tensor with one index up, the other down,
/// ie a matrix.
// Always returns an evaluated expression so it is safe to take a const &
template <class T, size_t N>
auto trace(const TensorExpression<2, T, N> &matrix) {
  auto trace = matrix[0][0];
  for (size_t i = 1; i < N; ++i) {
    trace += matrix[i][i];
  }
  return trace;
}

/// Raises the index of a covector
template <typename T1, typename T2>
std::enable_if_t<is_tensor_expression<T1>::value && has_size<T1, 1>::value &&
                     has_size<T2, 2>::value,
                 Dot<T2, T1>>
raise_all(T1 &&tensor_L, T2 &&inverse_metric) {
  return dot(std::forward<T2>(inverse_metric), std::forward<T1>(tensor_L));
}

/// Raises the index of a 2-tensor with 2 lower indices
template <typename T1, typename T2>
std::enable_if_t<is_tensor_expression<T1>::value && has_size<T1, 2>::value &&
                     has_size<T2, 2>::value,
                 Dot<T2, Dot<T1, T2>>>
raise_all(T1 &&tensor_LL, T2 &&inverse_metric) {
  return dot(
      std::forward<T2>(inverse_metric),
      dot(std::forward<T1>(tensor_LL), std::forward<T2>(inverse_metric)));
}

/// Lowers the indices of a vector
template <typename T1, typename T2>
std::enable_if_t<is_tensor_expression<T1>::value && has_size<T1, 1>::value &&
                     has_size<T2, 2>::value,
                 Dot<T2, T1>>
lower_all(T1 &&tensor_L, T2 &&inverse_metric) {
  return raise_all(std::forward<T1>(tensor_L),
                   std::forward<T2>(inverse_metric));
}

/// Lowers the indices of a rank 2 tensor with all indices up
template <typename T1, typename T2>
std::enable_if_t<is_tensor_expression<T1>::value && has_size<T1, 2>::value &&
                     has_size<T2, 2>::value,
                 Dot<T2, Dot<T1, T2>>>
lower_all(T1 &&tensor_LL, T2 &&inverse_metric) {
  return raise_all(std::forward<T1>(tensor_LL),
                   std::forward<T2>(inverse_metric));
}
} // namespace tensoralgebra

#endif
