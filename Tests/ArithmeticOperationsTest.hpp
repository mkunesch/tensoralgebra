#ifndef _TENSORALGEBRA_TESTS_ARITHMETICOPERATIONSTEST_HPP
#define _TENSORALGEBRA_TESTS_ARITHMETICOPERATIONSTEST_HPP

#include "Tensor.hpp"
#include "TestingUtilities.hpp"

#define define_mixed_test_function(OP, Name)                                   \
  template <typename T> T Name##_functions(T t) {                              \
    /*Checks all combinations with several layers of expression templates*/    \
    t OP## = (t OP 2.)OP(t OP 3.) OP(4. OP t);                                 \
    t OP## = 5.;                                                               \
    return t;                                                                  \
  }

// clang-format off
define_mixed_test_function(+, sum)
define_mixed_test_function(-, diff)
define_mixed_test_function(*, prod)
define_mixed_test_function(/, div)
// clang-format on
#undef define_mixed_test_function

        bool test_arithmetic_operations() {
  const double val = 2.;
  tensoralgebra::Tensor<2, double, 2> test_tensor = val;

  bool failed = false;
  failed |= verify_result(sum_functions(test_tensor), sum_functions(val));
  failed |= verify_result(diff_functions(test_tensor), diff_functions(val));
  failed |= verify_result(prod_functions(test_tensor), prod_functions(val));
  failed |= verify_result(div_functions(test_tensor), div_functions(val));

  print_result("Arithmetic operations test", !failed);

  return failed;
}

#endif
