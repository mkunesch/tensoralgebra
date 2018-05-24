#ifndef _TENSORALGEBRA_DOT_HPP
#define _TENSORALGEBRA_DOT_HPP

#include "Outer.hpp"
#include "TensorExpression.hpp"
#include "TypeChecks.hpp"
#include <cstddef>
#include <type_traits>

namespace tensoralgebra {

template <typename T1, typename T2>
class Dot : public TensorExpression<std::decay_t<T1>::rank() +
                                        std::decay_t<T2>::rank() - 2,
                                    Dot<T1, T2>, std::decay_t<T1>::size()> {
  T1 t1;
  T2 t2;

public:
  Dot(T1 &&t1, T2 &&t2) : t1(std::forward<T1>(t1)), t2(std::forward<T2>(t2)) {}

  template <typename... Indices> auto eval(Indices... js) const {
    auto outer_product = outer(t1, t2);
    constexpr size_t rank_T1 = std::decay_t<T1>::rank();
    auto dot =
        IndexContracter<rank_T1, rank_T1 + 1>::eval(outer_product, 0, js...);
    for (size_t i = 1; i < t1.size(); ++i) {
      dot +=
          IndexContracter<rank_T1, rank_T1 + 1>::eval(outer_product, i, js...);
    }
    return dot;
  }
};

/// Calculates the dot product of two tensors.
///(contracts the last index of the first with the first index of the second)
// The case where the result will have rank greater than 0
template <typename T1, typename T2>
typename std::enable_if_t<
    is_tensor_expression<T1>::value && is_tensor_expression<T2>::value &&
        (std::decay_t<T1>::rank() + std::decay_t<T2>::rank() > 2),
    Dot<T1, T2>>
dot(T1 &&t1, T2 &&t2) {
  return Dot<T1, T2>(std::forward<T1>(t1), std::forward<T2>(t2));
}

// The case where the result will be a scalar
// This will immediately be evaluated so temporaries as input require no special
// treatment.
template <typename T1, typename T2, size_t Size>
auto dot(const TensorExpression<1, T1, Size> &t1,
         const TensorExpression<1, T2, Size> &t2) {
  auto dot_product = t1[0] * t2[0];
  for (size_t i = 1; i < t1.size(); ++i) {
    dot_product += t1[i] * t2[i];
  }
  return dot_product;
}

/// Computes the dot product of two tensors given a metric
// Equivalent to tensor1.(metric.tensor2)
template <typename T1, size_t Rank1, typename T2, size_t Rank2, typename T3,
          size_t Size>
auto dot(const TensorExpression<Rank1, T1, Size> &tensor1,
         const TensorExpression<Rank2, T2, Size> &tensor2,
         const TensorExpression<2, T3, Size> &metric) {
  return dot(tensor1, dot(metric, tensor2));
}

} // namespace tensoralgebra

#endif
