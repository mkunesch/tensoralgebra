#ifndef _TENSORALGEBRA_COMPONENTOPERATIONS_HPP
#define _TENSORALGEBRA_COMPONENTOPERATIONS_HPP

#include "TensorExpression.hpp"
#include "TypeChecks.hpp"

// This file defines all component wise operations which return a tensor of the
// same rank (as all these operations are implemented in the same way).

namespace tensoralgebra {

// Define the expression templates corresponding to various operations:
#define define_binary_expression_template(Name, expression)                    \
  /* Expression template for the operation between a tensor and an arbitrary   \
   * type. */                                                                  \
  template <typename TTensor, typename TAny>                                   \
  class Name : public TensorExpression<std::decay_t<TTensor>::rank(),          \
                                       Name<TTensor, TAny>,                    \
                                       std::decay_t<TTensor>::size()> {        \
    TTensor tensor;                                                            \
    TAny any;                                                                  \
                                                                               \
  public:                                                                      \
    Name(TTensor &&tensor, TAny &&any)                                         \
        : tensor(std::forward<TTensor>(tensor)),                               \
          any(std::forward<TAny>(any)) {}                                      \
                                                                               \
    template <typename... Indices> auto eval(Indices... js) const {            \
      return expression;                                                       \
    }                                                                          \
  }

#define define_unary_expression_template(Name, expression)                     \
  /* Expression template for functions of tensors. */                          \
  template <typename TTensor>                                                  \
  class Name                                                                   \
      : public TensorExpression<std::decay_t<TTensor>::rank(), Name<TTensor>,  \
                                std::decay_t<TTensor>::size()> {               \
    TTensor tensor;                                                            \
                                                                               \
  public:                                                                      \
    Name(TTensor &&t) : tensor(std::forward<TTensor>(t)) {}                    \
                                                                               \
    template <typename... Indices> auto eval(Indices... js) const {            \
      return expression;                                                       \
    }                                                                          \
  }

#define define_binary_templates(OP, OPName)                                    \
  /*Define the expression templates needed for the binary operations*/         \
  define_binary_expression_template(OPName##ScalarRight,                       \
                                    tensor.eval(js...) OP any);                \
  define_binary_expression_template(OPName##ScalarLeft,                        \
                                    any OP tensor.eval(js...));                \
  define_binary_expression_template(OPName##Tensor,                            \
                                    tensor.eval(js...) OP any.eval(js...));

#define define_unary_template(function, Name)                                  \
  define_unary_expression_template(Name, function(tensor.eval(js...)));

// clang-format off
define_binary_templates(+, Sum)
define_binary_templates(-, Difference)
define_binary_templates(*, Product)
define_binary_templates(/, Quotient)

define_binary_templates(>=, IsGreaterEqual)
define_binary_templates(<=, IsLessEqual)
define_binary_templates(>, IsGreater)
define_binary_templates(<, IsLess)

define_unary_template(exp, Exp)
define_unary_template(log, Log)
define_unary_template(log10, Log10)
define_unary_template(sqrt, Sqrt)
define_unary_template(sin, Sin)
define_unary_template(cos, Cos)
define_unary_template(tan, Tan)
define_unary_template(asin, Asin)
define_unary_template(acos, Acos)
define_unary_template(atan, Atan)
define_unary_template(sinh, Sinh)
define_unary_template(cosh, Cosh)
define_unary_template(tanh, Tanh)
using std::abs; // Prevents bug whereby C's abs(int) is called
define_unary_template(abs, Abs)
// clang-format on

#undef define_binary_expression_template
#undef define_unary_expression_template
#undef define_binary_templates
#undef define_unary_template

#define define_binary_op(OP, OPName)                                           \
  /* Accepts only tensors of same rank and size */                             \
  template <typename T1, typename T2>                                          \
  std::enable_if_t<is_tensor_expression<T1>::value &&                          \
                       is_tensor_expression<T2>::value,                        \
                   OPName##Tensor<T1, T2>>                                     \
  operator OP(T1 &&in1, T2 &&in2) {                                            \
    return OPName##Tensor<T1, T2>(std::forward<T1>(in1),                       \
                                  std::forward<T2>(in2));                      \
  }                                                                            \
                                                                               \
  /* To avoid ambiguous function calls this version of OP is only visible if   \
   * TScalar's size doesn't match.*/                                           \
  template <typename T, typename TScalar>                                      \
  typename std::enable_if_t<is_tensor_expression<T>::value &&                  \
                                !are_same_size<T, TScalar>::value,             \
                            OPName##ScalarRight<T, TScalar>>                   \
  operator OP(T &&arr, TScalar &&value) {                                      \
    return OPName##ScalarRight<T, TScalar>(std::forward<T>(arr),               \
                                           std::forward<TScalar>(value));      \
  }                                                                            \
                                                                               \
  /* To avoid ambiguous function calls this version of OP is only visible if   \
   * TScalar's size doesn't match.*/                                           \
  template <typename T, typename TScalar>                                      \
  typename std::enable_if_t<is_tensor_expression<T>::value &&                  \
                                !are_same_size<T, TScalar>::value,             \
                            OPName##ScalarLeft<T, TScalar>>                    \
  operator OP(TScalar &&value, T &&arr) {                                      \
    return OPName##ScalarLeft<T, TScalar>(std::forward<T>(arr),                \
                                          std::forward<TScalar>(value));       \
  }

#define define_unary_function(function, Name)                                  \
  template <typename T>                                                        \
  std::enable_if_t<is_tensor_expression<T>::value, Name<T>> function(          \
      T &&arr) {                                                               \
    return Name<T>(std::forward<T>(arr));                                      \
  }

    // clang-format off
define_binary_op(+, Sum)
define_binary_op(-, Difference)
define_binary_op(*, Product)
define_binary_op(/, Quotient)

define_binary_op(>=, IsGreaterEqual)
define_binary_op(<=, IsLessEqual)
define_binary_op(>, IsGreater)
define_binary_op(<, IsLess)

define_unary_function(exp, Exp)
define_unary_function(log, Log)
define_unary_function(log10, Log10)
define_unary_function(sqrt, Sqrt)
define_unary_function(sin, Sin)
define_unary_function(cos, Cos)
define_unary_function(tan, Tan)
define_unary_function(asin, Asin)
define_unary_function(acos, Acos)
define_unary_function(atan, Atan)
define_unary_function(sinh, Sinh)
define_unary_function(cosh, Cosh)
define_unary_function(tanh, Tanh)
define_unary_function(abs, Abs)
// clang-format on

#undef define_binary_op
#undef define_arithmetic_op
#undef define_unary_function
} // namespae tensoralgebra

#endif
