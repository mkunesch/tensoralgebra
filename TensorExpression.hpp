#pragma once

#include "TypeChecks.hpp"
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace tensoralgebra {

template <typename T, size_t N> class TensorExpression;

/// Compile time check whether a given type is a tensor expression
/** Member "value" is false if the given type is not a tensor expression */
template <typename T, typename Helper = void>
struct is_tensor_expression : std::false_type {};

template <typename T>
struct is_tensor_expression<
    T, make_void<decltype(std::remove_reference_t<T>::size())>> {
  using T_stripped = std::decay_t<T>;
  static const bool value =
      std::is_base_of<TensorExpression<T_stripped, T_stripped::size()>,
                      T_stripped>::value;
};
// End: compile time check whether a given type is a tensor expression

/// Base class for everything that is a tensor expression, from just a
/// tensor itself to complicated unevaluated operations of tensors
template <typename T, size_t N> class TensorExpression {
protected:
  // Protected since creating exressions outside derived classes is dangerous.
  TensorExpression() = default;
  TensorExpression(const TensorExpression<T, N> &) = default;

public:
  auto operator[](size_t i) const { return static_cast<const T &>(*this)[i]; }
  static constexpr size_t size() { return N; }
};

// Define the expression templates corresponding to various operations:
#define define_binary_expression_template(Name, expression)                    \
  /* Expression template for the operation between a tensor and an arbitrary   \
   * type. The members are either lvalues or lvalue references depending on    \
   * whether the class is initialised with lvalues or rvalues. */              \
  template <typename TTensor, typename TAny>                                   \
  class Name                                                                   \
      : public TensorExpression<Name<TTensor, TAny>,                           \
                                std::remove_reference_t<TTensor>::size()> {    \
    const TTensor tensor;                                                      \
    const TAny any;                                                            \
                                                                               \
  public:                                                                      \
    Name(TTensor &&tensor, TAny &&any)                                         \
        : tensor(std::forward<TTensor>(tensor)),                               \
          any(std::forward<TAny>(any)) {}                                      \
                                                                               \
    auto operator[](size_t i) const { return expression; }                     \
  };

#define define_unary_expression_template(Name, expression)                     \
  /* Expression template for functions of tensors. The member variable is      \
   * either an lvalue or an lvalue reference depending on whether the class    \
   * is initialised with an lvalue or an rvalue.*/                             \
  template <typename TTensor>                                                  \
  class Name                                                                   \
      : public TensorExpression<Name<TTensor>,                                 \
                                std::remove_reference_t<TTensor>::size()> {    \
    const TTensor tensor;                                                      \
                                                                               \
  public:                                                                      \
    Name(TTensor &&t) : tensor(std::forward<TTensor>(t)) {}                    \
                                                                               \
    auto operator[](size_t i) const { return expression; };                    \
  };

#define define_binary_templates(OP, OPName)                                    \
  /*Define the expression templates needed for the binary operations*/         \
  define_binary_expression_template(OPName##ScalarRight, tensor[i] OP any);    \
  define_binary_expression_template(OPName##Tensor, tensor[i] OP any[i]);

#define define_unary_template(function, Name)                                  \
  define_unary_expression_template(Name, function(tensor[i]));

define_binary_templates(+, Sum);
define_binary_templates(-, Difference);
define_binary_templates(*, Product);
define_binary_templates(/, Quotient);

define_unary_template(exp, Exp);
define_unary_template(log, Log);
define_unary_template(log10, Log10);
define_unary_template(sqrt, Sqrt);
define_unary_template(sin, Sin);
define_unary_template(cos, Cos);
define_unary_template(tan, Tan);
define_unary_template(asin, Asin);
define_unary_template(acos, Acos);
define_unary_template(atan, Atan);
define_unary_template(sinh, Sinh);
define_unary_template(cosh, Cosh);
define_unary_template(tanh, Tanh);
using std::abs; // Prevents bug whereby C's abs(int) is called
define_unary_template(abs, Abs);

#undef define_binary_expression_template
#undef define_unary_expression_template
#undef define_binary_templates
#undef define_unary_template
} // namespace tensoralgebra
