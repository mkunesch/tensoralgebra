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
class Outer : public TensorExpression<std::decay_t<T1>::rank() +
                                          std::decay_t<T2>::rank(),
                                      Outer<T1, T2>, std::decay_t<T1>::size()> {
  T1 t1;
  T2 t2;

public:
  Outer(T1 &&t1, T2 &&t2)
      : t1(std::forward<T1>(t1)), t2(std::forward<T2>(t2)) {}

  template <typename... Indices> auto eval(Indices... dirs) const {
    return OuterHelper<std::decay_t<T1>::rank()>::eval(t1, t2, dirs...);
  }
};

template <typename T1, typename T2>
std::enable_if_t<is_tensor_expression<T1>::value &&
                     are_same_size<T1, T2>::value,
                 Outer<T1, T2>>
outer(T1 &&t1, T2 &&t2) {
  return Outer<T1, T2>(std::forward<T1>(t1), std::forward<T2>(t2));
}

} // namespace tensoralgebra

#endif
