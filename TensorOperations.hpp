#ifndef _TENSORALGEBRA_TENSOROPERATIONS_HPP
#define _TENSORALGEBRA_TENSOROPERATIONS_HPP

#include "TensorExpression.hpp"

// Defines the outer product using expression templates
#include "Outer.hpp"

// Defines the dot product using expression templates
#include "Dot.hpp"

namespace tensoralgebra {
/// Computes the trace of a 2-tensor with lower inverse given an inverse metric
template <class T, size_t N>
auto trace(const TensorExpression<2, T, N> &tensor_LL,
           const TensorExpression<2, T, N> &inverse_metric) {
  return trace(dot(inverse_metric, tensor_LL));
}

/// Computes the trace of a tensor with one index up, the other down,
/// ie a matrix.
template <class T, size_t N>
auto trace(const TensorExpression<2, T, N> &matrix) {
  auto trace = matrix[0][0];
  for (size_t i = 1; i < N; ++i) {
    trace += matrix[i][i];
  }
  return trace;
}

/// Removes the trace from a tensor with lower indices
template <class T, size_t N>
auto make_trace_free(const TensorExpression<2, T, N> &tensor_LL,
                     const TensorExpression<2, T, N> &metric,
                     const TensorExpression<2, T, N> &inverse_metric) {
  T trace = trace(tensor_LL, inverse_metric);
  return tensor_LL - (1. / N) * metric * trace;
}

/// Raises the index of a covector
template <class T, size_t N>
auto raise_all(const TensorExpression<1, T, N> &tensor_L,
               const TensorExpression<2, T, N> &inverse_metric) {
  return dot(inverse_metric, tensor_L);
}

/// Raises the index of a 2-tensor with 2 lower indices
template <class T, size_t N>
auto raise_all(const TensorExpression<2, T, N> &tensor_LL,
               const TensorExpression<2, T, N> &inverse_metric) {
  return dot(inverse_metric, dot(tensor_LL, inverse_metric));
}

/// Lowers the indices of a vector
template <class T, size_t N>
auto lower_all(const TensorExpression<1, T, N> &tensor_U,
               const TensorExpression<2, T, N> &metric) {
  // The code for lowering is exactly the same as for raising
  return raise_all(tensor_U, metric);
}

/// Lowers the indices of a rank 2 tensor with all indices up
template <class T, size_t N>
auto lower_all(const TensorExpression<2, T, N> &tensor_UU,
               const TensorExpression<2, T, N> &metric) {
  // The code for lowering is exactly the same as for raising
  return raise_all(tensor_UU, metric);
}
} // namespace tensoralgebra

#endif
