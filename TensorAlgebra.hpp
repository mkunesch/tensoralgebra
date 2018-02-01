#ifndef TENSORALGEBRA_HPP
#define TENSORALGEBRA_HPP

#include "Tensor.hpp"

namespace tensoralgebra {
/// Computes the trace of a 2-tensor with lower inverse given an inverse metric
template <class T, size_t N>
T trace(const Tensor<2, T, N> &tensor_LL,
        const Tensor<2, T, N> &inverse_metric) {
  T trace = 0.;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      trace += inverse_metric[i][j] * tensor_LL[i][j];
    }
  }
  return trace;
}

/// Computes the trace of a tensor with one index up, the other down,
/// ie a matrix.
template <class T, size_t N> T trace(const Tensor<2, T, N> &matrix) {
  T trace = 0.;
  for (size_t i = 0; i < N; ++i) {
    trace += matrix[i][i];
  }
  return trace;
}

/// Computes the dot product of a vector and a covector
template <class T, size_t N>
T dot(const Tensor<1, T, N> &vector, const Tensor<1, T, N> &covector) {
  T dot_product = 0.;
  for (size_t i = 0; i < N; ++i) {
    dot_product += vector[i] * covector[i];
  }
  return dot_product;
}

/// Computes the dot product of two vectors given a metric
template <class T, size_t N>
T dot(const Tensor<1, T, N> &vector1, const Tensor<1, T, N> &vector2,
      const Tensor<2, T, N> &metric) {
  T dot_product = 0.;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      dot_product += metric[i][j] * vector1[i] * vector2[j];
    }
  }
  return dot_product;
}

/// Removes the trace from a tensor with lower indices
template <class T, size_t N>
void make_trace_free(Tensor<2, T, N> &tensor_LL, const Tensor<2, T, N> &metric,
                     const Tensor<2, T, N> &inverse_metric) {
  T trace = trace(tensor_LL, inverse_metric);
  tensor_LL -= (1. / N) * metric * trace;
}

/// Raises the index of a covector
template <class T, size_t N>
Tensor<1, T, N> raise_all(const Tensor<1, T, N> &tensor_L,
                          const Tensor<2, T, N> &inverse_metric) {
  Tensor<1, T, N> tensor_U = 0.;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      tensor_U[i] += inverse_metric[i][j] * tensor_L[j];
    }
  }
  return tensor_U;
}

/// Raises the index of a 2-tensor with 2 lower indices
template <class T, size_t N>
Tensor<2, T, N> raise_all(const Tensor<2, T, N> &tensor_LL,
                          const Tensor<2, T, N> &inverse_metric) {
  Tensor<2, T, N> tensor_UU = 0.;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      for (size_t k = 0; k < N; ++k) {
        for (size_t l = 0; l < N; ++l) {
          tensor_UU[i][j] +=
              inverse_metric[i][k] * inverse_metric[j][l] * tensor_LL[k][l];
        }
      }
    }
  }
  return tensor_UU;
}

/// Lowers the indices of a vector
template <class T, size_t N>
Tensor<1, T, N> lower_all(const Tensor<1, T, N> &tensor_U,
                          const Tensor<2, T, N> &metric) {
  // The code for lowering is exactly the same as for raising
  return raise_all(tensor_U, metric);
}

/// Lowers the indices of a rank 2 tensor with all indices up
template <class T, size_t N>
Tensor<2, T, N> lower_all(const Tensor<2, T, N> &tensor_UU,
                          const Tensor<2, T, N> &metric) {
  // The code for lowering is exactly the same as for raising
  return raise_all(tensor_UU, metric);
}
} // namespace tensoralgebra

#endif /* TENSORALGEBRA_HPP */
