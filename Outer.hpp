#ifndef _TENSORALGEBRA_OUTER_HPP
#define _TENSORALGEBRA_OUTER_HPP

#include "Tensor.hpp"
#include "TensorExpression.hpp"
#include <cstddef>
#include <utility>

namespace tensoralgebra {
template <size_t Position> struct OuterHelper {
  template <typename T1, typename T2, typename... IndexTs>
  static auto eval(const T1 &t1, const T2 &t2, size_t dir, IndexTs... dirs) {
    return OuterHelper<Position - 1>::eval(t1[dir], t2, dirs...);
  }
};

template <> struct OuterHelper<0> {
  template <typename T1, typename T2, typename... IndexTs>
  static auto eval(const T1 &t1, const T2 &t2, IndexTs... dirs) {
    return t1 * t2.eval(dirs...);
  }
};

template <typename T1, typename T2>
class Outer : public TensorExpression<T1::rank() + T2::rank(), Outer<T1, T2>,
                                      T1::size()> {
  const T1 &t1;
  const T2 &t2;

public:
  Outer(const T1 &t1, const T2 &t2) : t1(t1), t2(t2) {}

  template <typename... Indices> auto eval(Indices... dirs) const {
    return OuterHelper<T1::rank()>::eval(t1, t2, dirs...);
  }
};

template <size_t Rank1, typename T1, size_t Rank2, typename T2, size_t Size>
auto outer(const TensorExpression<Rank1, T1, Size> &t1,
           const TensorExpression<Rank2, T2, Size> &t2) {
  return Outer<TensorExpression<Rank1, T1, Size>,
               TensorExpression<Rank2, T2, Size>>(t1, t2);
}

} // namespace tensoralgebra

#endif
