#ifndef _TENSORALGEBRA_TENSOREXPRESSION_HPP
#define _TENSORALGEBRA_TENSOREXPRESSION_HPP

#include "IndexUtilities.hpp"
#include "TypeChecks.hpp"
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tensoralgebra {

template <typename T> class SquareBracket;

/// Base class for everything that is a tensor expression, from just a
/// tensor itself to complicated unevaluated operations of tensors
template <size_t Rank, typename T, size_t Size> class TensorExpression {
protected:
  // Protected since creating expressions outside derived classes is dangerous.
  TensorExpression() = default;

public:
  auto operator[](size_t i) const {
    return SquareBracket<T>(static_cast<const T &>(*this), i);
  }
  static constexpr size_t size() { return Size; }
  static constexpr size_t rank() { return Rank; }
  using TensorExpressionType = T;

  template <typename... Indices> decltype(auto) eval(Indices... is) const {
    return static_cast<const T &>(*this).eval(is...);
  }
};

template <typename T, size_t Size> class TensorExpression<1, T, Size> {
protected:
  // Protected since creating expressions outside derived classes is dangerous.
  TensorExpression() = default;

public:
  auto operator[](size_t i) const { return eval(i); }
  static constexpr size_t size() { return Size; }
  static constexpr size_t rank() { return 1; }
  using TensorExpressionType = T;

  decltype(auto) eval(size_t i) const {
    return static_cast<const T &>(*this).eval(i);
  }
};

// Rank zero tensor expressions are forbidden
template <typename T, size_t Size> class TensorExpression<0, T, Size>;

// Expression template corresponding to the [] operator. This remains
// unevaluated until enough indices (given by the rank) are supplied.
template <typename T>
class SquareBracket
    : public TensorExpression<T::rank() - 1, SquareBracket<T>, T::size()> {
  const T &t;
  size_t i;

public:
  SquareBracket(const T &tensor, size_t i) : t(tensor), i(i) {}

  template <typename... Indices> auto eval(Indices... js) const {
    return t.eval(i, js...);
  }
};

/// Operator == for tensor expressions.
// The result is immediately evaluated.
template <size_t Rank, typename T1, typename T2, size_t Size>
bool operator==(const TensorExpression<Rank, T1, Size> &t1,
                const TensorExpression<Rank, T2, Size> &t2) {
  bool are_equal = true;
  for (size_t i = 0; i < Size; ++i) {
    are_equal &= (t1[i] == t2[i]);
  }
  return are_equal;
}

/// Operator != for tensor expressions.
// The result is immediately evaluated.
template <size_t Rank, typename T1, typename T2, size_t Size>
bool operator!=(const TensorExpression<Rank, T1, Size> &t1,
                const TensorExpression<Rank, T2, Size> &t2) {
  return !(t1 == t2);
}
} // namespace tensoralgebra

#endif
