#ifndef _TENSORALGEBRA_TENSOR_HPP
#define _TENSORALGEBRA_TENSOR_HPP

#include "ComponentOperations.hpp"
#include "NestedInitializerList.hpp"
#include "TensorExpression.hpp"
#include "TypeChecks.hpp"
#include <array>
#include <iostream>
#include <type_traits>

namespace tensoralgebra {

/// Tensor<Rang, T, Dim> represents a tensor of rank Rank, element type T, and
/// size/dimension Size
/** The defaults are T = double and Size = 3 (the physical number of dimensions)
 */
template <size_t Rank, typename T = double, size_t Size = 3> class Tensor;

// Implemented recursively: a rank R tensor is an array of rank R-1
// tensors.
template <size_t Rank, typename T, size_t Size> struct tensor_type_recursion {
  using type = std::array<Tensor<Rank - 1, T, Size>, Size>;
};

// The base case: a rank 1 tensor is just an array
template <typename T, size_t Size> struct tensor_type_recursion<1, T, Size> {
  using type = std::array<T, Size>;
};

// Zero tensors are forbidden
template <typename T, size_t Size> struct tensor_type_recursion<0, T, Size> {};

template <size_t Rank, typename T, size_t Size>
class Tensor : public TensorExpression<Rank, Tensor<Rank, T, Size>, Size> {

  using ContainedType = typename tensor_type_recursion<Rank, T, Size>::type;
  ContainedType data;

public:
  Tensor() = default;

  /// Create a Tensor by evaluating an expression (implicit conversion allowed)
  template <typename T1>
  Tensor(const TensorExpression<Rank, T1, Size> &expression);

  template <typename T1>
  Tensor &operator=(const TensorExpression<Rank, T1, Size> &expression);

  Tensor(const T &value) { operator=(value); }
  Tensor &operator=(const T &value) {
    data.fill(value);
    return *this;
  }

  Tensor(const NestedInitializerList<T, Rank> &list) {
    std::copy(list.begin(), list.end(), data.begin());
  }

  static constexpr size_t size() { return Size; }
  static constexpr size_t rank() { return Rank; }

  using iterator = typename ContainedType::iterator;
  using const_iterator = typename ContainedType::const_iterator;

  const auto &operator[](size_t i) const { return data[i]; }

  auto &operator[](size_t i) { return data[i]; }

  iterator begin() { return data.begin(); }

  iterator end() { return data.end(); }

  const_iterator begin() const { return data.begin(); }

  const_iterator end() const { return data.end(); }

  template <typename... Indices> const auto &eval(Indices... is) const {
    return apply_indices(*this, is...);
  }

  template <typename T1>
  Tensor<Rank, T, Size> &
  operator+=(const TensorExpression<Rank, T1, Size> &expression);

  template <typename T1>
  Tensor<Rank, T, Size> &
  operator-=(const TensorExpression<Rank, T1, Size> &expression);

  template <typename T1>
  Tensor<Rank, T, Size> &
  operator*=(const TensorExpression<Rank, T1, Size> &expression);

  template <typename T1>
  Tensor<Rank, T, Size> &
  operator/=(const TensorExpression<Rank, T1, Size> &expression);

  // to avoid ambiguous function calls the following versions of OP= are only
  // visible if T does not have the same size as T1 or isn't a tensor at all
  template <typename T1>
  typename std::enable_if_t<!has_size<T1, Size>::value, Tensor<Rank, T, Size> &>
  operator+=(const T1 &value);

  template <typename T1>
  typename std::enable_if_t<!has_size<T1, Size>::value, Tensor<Rank, T, Size> &>
  operator-=(const T1 &value);

  template <typename T1>
  typename std::enable_if_t<!has_size<T1, Size>::value, Tensor<Rank, T, Size> &>
  operator*=(const T1 &value);

  template <typename T1>
  typename std::enable_if_t<!has_size<T1, Size>::value, Tensor<Rank, T, Size> &>
  operator/=(const T1 &value);
};

template <size_t Rank, typename T, size_t Size>
template <typename T1>
inline __attribute__((always_inline)) Tensor<Rank, T, Size>::Tensor(
    const TensorExpression<Rank, T1, Size> &expression) {
  operator=(expression);
}

template <size_t Rank, typename T, size_t Size>
template <typename T1>
inline __attribute__((always_inline)) Tensor<Rank, T, Size> &
Tensor<Rank, T, Size>::
operator=(const TensorExpression<Rank, T1, Size> &expression) {
  for (size_t i = 0; i < Size; ++i) {
    data[i] = expression[i];
  }
  return *this;
}

template <size_t Rank, typename T, size_t Size>
inline std::ostream &operator<<(std::ostream &os,
                                const TensorExpression<Rank, T, Size> &tensor) {
  os << "{";
  for (size_t i = 0; i < Size - 1; ++i) {
    os << tensor[i] << ",";
  }
  os << tensor[Size - 1] << "}"; // No comma after last element
  return os;
}

#define define_arithmetic_op(OP, OPName)                                       \
  template <size_t Rank, typename T, size_t Size>                              \
  template <typename T1>                                                       \
  inline __attribute__((always_inline))                                        \
      Tensor<Rank, T, Size> &Tensor<Rank, T, Size>::operator OP##=(            \
          const TensorExpression<Rank, T1, Size> &expression) {                \
    for (size_t i = 0; i < Size; ++i)                                          \
      (*this)[i] OP## = expression[i];                                         \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <size_t Rank, typename T, size_t Size>                              \
  template <typename T1>                                                       \
  inline __attribute__((always_inline))                                        \
      typename std::enable_if_t<!has_size<T1, Size>::value,                    \
                                Tensor<Rank, T, Size> &>                       \
          Tensor<Rank, T, Size>::operator OP##=(const T1 &value) {             \
    for (auto &element : *this)                                                \
      element OP## = value;                                                    \
    return *this;                                                              \
  }

// clang-format off
define_arithmetic_op(+, Sum)
define_arithmetic_op(-, Difference)
define_arithmetic_op(*, Product)
define_arithmetic_op(/, Quotient)
// clang-format on

} // namespace tensoralgebra

#endif
