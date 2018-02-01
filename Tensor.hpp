#pragma once

#include "Array.hpp"
#include "TensorExpression.hpp"
#include "TypeChecks.hpp"
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
  using type = Array<Tensor<Rank - 1, T, Size>, Size>;
};

// The base case: a rank 1 tensor is just an array
template <typename T, size_t Size> struct tensor_type_recursion<1, T, Size> {
  using type = Array<T, Size>;
};

// Zero tensors are forbidden
template <typename T, size_t Size> struct tensor_type_recursion<0, T, Size> {};

template <size_t Rank, typename T, size_t Size>
class Tensor : public tensor_type_recursion<Rank, T, Size>::type,
               public TensorExpression<Tensor<Rank, T, Size>, Size> {
public:
  using Base = typename tensor_type_recursion<Rank, T, Size>::type;
  using Base::Base;
  using Base::operator[];
  static constexpr size_t size() { return Size; }
  static constexpr size_t rank() { return Rank; }

  /// Create a Tensor by evaluating an expression (implicit conversion allowed)
  template <typename T1> Tensor(const TensorExpression<T1, Size> &expression);

  template <typename T1>
  Tensor<Rank, T, Size> &
  operator+=(const TensorExpression<T1, Size> &expression);

  template <typename T1>
  Tensor<Rank, T, Size> &
  operator-=(const TensorExpression<T1, Size> &expression);

  template <typename T1>
  Tensor<Rank, T, Size> &
  operator*=(const TensorExpression<T1, Size> &expression);

  template <typename T1>
  Tensor<Rank, T, Size> &
  operator/=(const TensorExpression<T1, Size> &expression);

  // Saves us some writing below
  using this_type = Tensor<Rank, T, Size>;

  // to avoid ambiguous function calls the following versions of OP= are only
  // visible if T does not have the same size as T1 or isn't a tensor at all
  template <typename T1>
  typename std::enable_if_t<!are_same_size<T1, this_type>::value,
                            Tensor<Rank, T, Size> &>
  operator+=(const T1 &value);

  template <typename T1>
  typename std::enable_if_t<!are_same_size<T1, this_type>::value,
                            Tensor<Rank, T, Size> &>
  operator-=(const T1 &value);

  template <typename T1>
  typename std::enable_if_t<!are_same_size<T1, this_type>::value,
                            Tensor<Rank, T, Size> &>
  operator*=(const T1 &value);

  template <typename T1>
  typename std::enable_if_t<!are_same_size<T1, this_type>::value,
                            Tensor<Rank, T, Size> &>
  operator/=(const T1 &value);
};

template <size_t Rank, typename T, size_t Size>
template <typename T1>
Tensor<Rank, T, Size>::Tensor(const TensorExpression<T1, Size> &expression) {
  for (size_t i = 0; i < Size; ++i) {
    (*this)[i] = expression[i];
  }
}

template <typename T, size_t Size>
std::ostream &operator<<(std::ostream &os,
                         const TensorExpression<T, Size> &arr) {
  os << "{";
  for (size_t i = 0; i < Size - 1; ++i) {
    os << arr[i] << ",";
  }
  os << arr[Size - 1] << "}"; // No comma after last element
  return os;
}

#define define_arithmetic_op(OP, OPName)                                       \
  template <size_t Rank, typename T, size_t Size>                              \
  template <typename T1>                                                       \
  Tensor<Rank, T, Size> &Tensor<Rank, T, Size>::operator OP##=(                \
      const TensorExpression<T1, Size> &expression) {                          \
    for (size_t i = 0; i < Size; ++i)                                          \
      (*this)[i] OP## = expression[i];                                         \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  template <size_t Rank, typename T, size_t Size>                              \
  template <typename T1>                                                       \
  typename std::enable_if_t<!are_same_size<T1, Tensor<Rank, T, Size>>::value,  \
                            Tensor<Rank, T, Size> &>                           \
      Tensor<Rank, T, Size>::operator OP##=(const T1 &value) {                 \
    for (auto &element : *this)                                                \
      element OP## = value;                                                    \
    return *this;                                                              \
  }                                                                            \
                                                                               \
  /* To avoid ambiguous function calls this version of OP is only defined      \
   * if both T1 and T2 are tensor expressions and they have the same size */   \
  template <typename T1, typename T2>                                          \
  typename std::enable_if_t<are_same_size<T1, T2>::value &&                    \
                                is_tensor_expression<T1>::value &&             \
                                is_tensor_expression<T2>::value,               \
                            OPName##Tensor<T1, T2>>                            \
  operator OP(T1 &&in1, T2 &&in2) {                                            \
    return OPName##Tensor<T1, T2>(std::forward<T1>(in1),                       \
                                  std::forward<T2>(in2));                      \
  }                                                                            \
                                                                               \
  /* to avoid ambiguous function calls this version of OP is only visible if   \
   * TTensor is a tensor expression and TScalar's size doesn't match.*/        \
  template <typename TTensor, typename TScalar>                                \
  typename std::enable_if_t<is_tensor_expression<TTensor>::value &&            \
                                !are_same_size<TScalar, TTensor>::value,       \
                            OPName##ScalarRight<TTensor, TScalar>>             \
  operator OP(TTensor &&arr, TScalar &&value) {                                \
    return OPName##ScalarRight<TTensor, TScalar>(                              \
        std::forward<TTensor>(arr), std::forward<TScalar>(value));             \
  }                                                                            \
                                                                               \
  /* to avoid ambiguous function calls this version of OP is only visible if   \
   * TTensor is a tensor expression and TScalar's size doesn't match.*/        \
  template <typename TTensor, typename TScalar>                                \
  typename std::enable_if_t<is_tensor_expression<TTensor>::value &&            \
                                !are_same_size<TScalar, TTensor>::value,       \
                            OPName##ScalarRight<TTensor, TScalar>>             \
  operator OP(TScalar &&value, TTensor &&arr) {                                \
    return OPName##ScalarRight<TTensor, TScalar>(                              \
        std::forward<TTensor>(arr), std::forward<TScalar>(value));             \
  }

#define define_unary_function(function, Name)                                  \
  /* to avoid ambiguous function calls, function is only defined if TTensor    \
   * is a tensor expression. */                                                \
  template <typename TTensor>                                                  \
  std::enable_if_t<is_tensor_expression<TTensor>::value, Name<TTensor>>        \
  function(TTensor &&arr) {                                                    \
    return Name<TTensor>(std::forward<TTensor>(arr));                          \
  }

define_arithmetic_op(+, Sum);
define_arithmetic_op(-, Difference);
define_arithmetic_op(*, Product);
define_arithmetic_op(/, Quotient);

define_unary_function(exp, Exp);
define_unary_function(log, Log);
define_unary_function(log10, Log10);
define_unary_function(sqrt, Sqrt);
define_unary_function(sin, Sin);
define_unary_function(cos, Cos);
define_unary_function(tan, Tan);
define_unary_function(asin, Asin);
define_unary_function(acos, Acos);
define_unary_function(atan, Atan);
define_unary_function(sinh, Sinh);
define_unary_function(cosh, Cosh);
define_unary_function(tanh, Tanh);
define_unary_function(abs, Abs);

#undef define_arithmetic_op
#undef define_unary_function

} // namespace tensoralgebra
