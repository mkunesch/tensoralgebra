#ifndef _TENSORALGEBRA_TESTS_RELATIONALOPERATORSTEST_HPP
#define _TENSORALGEBRA_TESTS_RELATIONALOPERATORSTEST_HPP

#include "Tensor.hpp"
#include "TestingUtilities.hpp"
#include <iostream>

#define define_binary_test_function(OP, Name)                                  \
  template <typename T> T Name##_functions(T t) {                              \
    /*Checks all combinations with several layers of expression templates*/    \
    return (2. OP t)OP(t OP 3.) OP(4. OP t);                                   \
  }

// clang-format off
define_binary_test_function(>=, is_greater_equal)
define_binary_test_function(<=, is_less_equal)
define_binary_test_function(>, is_greater)
define_binary_test_function(<, is_less)
// clang-format on

#undef define_binary_test_function

                bool test_relational_operations() {
  const double val = 3.;
  // Set up several different input tensors so that we test both true and false
  // for the relational operators
  tensoralgebra::Tensor<2, double, 2> test_tensor = val;
  tensoralgebra::Tensor<2, double, 2> test_tensor1 = -1 * test_tensor;
  tensoralgebra::Tensor<2, double, 2> test_tensor2 = 3 * test_tensor;

  bool failed = false;
  failed |= verify_result(is_greater_equal_functions(test_tensor),
                          is_greater_equal_functions(val));
  failed |= verify_result(is_greater_equal_functions(test_tensor1),
                          is_greater_equal_functions(-val));

  failed |= verify_result(is_less_equal_functions(test_tensor),
                          is_less_equal_functions(val));
  failed |= verify_result(is_less_equal_functions(test_tensor2),
                          is_less_equal_functions(3 * val));

  failed |= verify_result(is_greater_functions(test_tensor),
                          is_greater_functions(val));
  failed |= verify_result(is_greater_functions(test_tensor1),
                          is_greater_functions(-val));

  failed |=
      verify_result(is_less_functions(test_tensor), is_less_functions(val));
  failed |=
      verify_result(is_less_functions(test_tensor1), is_less_functions(-val));

  print_result("Relational operators test", !failed);

  return failed;
}

#endif
