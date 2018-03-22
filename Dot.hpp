#ifndef _TENSORALGEBRA_DOT_HPP
#define _TENSORALGEBRA_DOT_HPP

#include "Outer.hpp"
#include "TensorExpression.hpp"
#include "TypeChecks.hpp"
#include <cstddef>
#include <type_traits>

namespace tensoralgebra {

template <typename T1, typename T2>
class Dot : public TensorExpression<T1::rank() + T2::rank() - 2, Dot<T1, T2>,
                                    T1::size()> {
  const T1 &t1;
  const T2 &t2;

public:
  Dot(const T1 &t1, const T2 &t2) : t1(t1), t2(t2) {}

  template <typename... Indices> auto eval(Indices... js) const {
    auto outer_product = outer(t1, t2);
    auto dot = IndexContracter<T1::rank(), T1::rank() + 1>::eval(outer_product,
                                                                 0, js...);
    for (size_t i = 1; i < t1.size(); ++i) {
      dot += IndexContracter<T1::rank(), T1::rank() + 1>::eval(outer_product, i,
                                                               js...);
    }
    return dot;
  }
};

/// Calculates the dot product of two tensors.
///(contracts the last index of the first with the first index of the second)
// The case where the result will have rank greater than 0
template <size_t Rank1, typename T1, size_t Rank2, typename T2, size_t Size>
typename std::enable_if_t<
    (Rank1 + Rank2 > 2),
    Dot<TensorExpression<Rank1, T1, Size>, TensorExpression<Rank2, T2, Size>>>
dot(const TensorExpression<Rank1, T1, Size> &t1,
    const TensorExpression<Rank2, T2, Size> &t2) {
  return Dot<TensorExpression<Rank1, T1, Size>,
             TensorExpression<Rank2, T2, Size>>(t1, t2);
}

// The case where the result will be a scalar
template <typename T1, typename T2, size_t Size>
auto dot(const TensorExpression<1, T1, Size> &t1,
         const TensorExpression<1, T2, Size> &t2) {
  auto dot_product = t1[0] * t2[0];
  for (size_t i = 1; i < t1.size(); ++i) {
    dot_product += t1[i] * t2[i];
  }
  return dot_product;
}

/// Computes the dot product of two vectors given a metric
template <typename T1, typename T2, typename T3, size_t Size>
auto dot(const TensorExpression<1, T1, Size> &vector1,
         const TensorExpression<1, T2, Size> &vector2,
         const TensorExpression<2, T3, Size> &metric) {
  return dot(vector1, dot(metric, vector2));
}

} // namespace tensoralgebra

#endif
