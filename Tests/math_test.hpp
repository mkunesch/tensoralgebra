#pragma once

#include "Tensor.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>

// The tests in this file check the accuracy of arithmetic operations and
// transcendental functions.

#define define_test_function(OP, Name)                                         \
  template <typename T> T Name##_functions(T t) {                              \
    double value = 3.;                                                         \
    /*Checks all combinations with several layers of expression temapltes*/    \
    t OP## = (value OP t)OP(t OP value) OP(value OP t);                        \
    t OP## = value;                                                            \
    return t;                                                                  \
  }

define_test_function(+, sum);
define_test_function(-, diff);
define_test_function(*, prod);
define_test_function(/, div);

#undef define_test_function

#define define_transcendental_test_function(FUNCTION)                          \
  template <typename T> T FUNCTION##_functions(T t) {                          \
    t = FUNCTION(FUNCTION(t));                                                 \
    return t;                                                                  \
  }

define_transcendental_test_function(exp);
define_transcendental_test_function(log);
define_transcendental_test_function(log10);
define_transcendental_test_function(sqrt);
define_transcendental_test_function(sin);
define_transcendental_test_function(cos);
define_transcendental_test_function(tan);
define_transcendental_test_function(asin);
define_transcendental_test_function(acos);
define_transcendental_test_function(atan);
define_transcendental_test_function(sinh);
define_transcendental_test_function(cosh);
define_transcendental_test_function(tanh);
using std::abs; // Prevents bug whereby C's abs(int) is called
define_transcendental_test_function(abs);
#undef define_transcendental_test_function

bool verify_result(const tensoralgebra::Tensor<2, double, 2> &tensor,
                   double correct_result, double precision = 1e-14) {
  bool failed = false;
  for (auto &row : tensor) {
    for (auto &element : row) {
      failed |= !(std::abs(element - correct_result) < precision);
    }
  }
  return failed;
}

bool test_arithmetic_operations() {
  const double val = 2.;
  tensoralgebra::Tensor<2, double, 2> test_tensor = {{val, val}, {val, val}};

  bool failed = false;
  failed |= verify_result(sum_functions(test_tensor), sum_functions(val));
  failed |= verify_result(diff_functions(test_tensor), diff_functions(val));
  failed |= verify_result(prod_functions(test_tensor), prod_functions(val));
  failed |= verify_result(div_functions(test_tensor), div_functions(val));

  std::cout << "Arithmetic operations test passed: " << !failed << std::endl;

  return failed;
}

bool test_transcendental_functions() {
  // Need several values to please functions which are picky about their domain
  const double val = 1.1;
  const double val1 = -0.1;
  const double val2 = 0.9;
  using TwoTensor = tensoralgebra::Tensor<2, double, 2>;
  TwoTensor test_tensor = {{val, val}, {val, val}};
  TwoTensor test_tensor1 = {{val1, val1}, {val1, val1}};
  TwoTensor test_tensor2 = {{val2, val2}, {val2, val2}};

  bool failed = false;
  failed |= verify_result(exp_functions(test_tensor), exp_functions(val));
  failed |= verify_result(log_functions(test_tensor), log_functions(val));
  failed |= verify_result(log10_functions(test_tensor), log10_functions(val));
  failed |= verify_result(sqrt_functions(test_tensor), sqrt_functions(val));
  failed |= verify_result(sin_functions(test_tensor), sin_functions(val));
  failed |= verify_result(cos_functions(test_tensor), cos_functions(val));
  failed |= verify_result(tan_functions(test_tensor), tan_functions(val));
  failed |= verify_result(asin_functions(test_tensor1), asin_functions(val1));
  failed |= verify_result(acos_functions(test_tensor2), acos_functions(val2));
  failed |= verify_result(atan_functions(test_tensor1), atan_functions(val1));
  failed |= verify_result(sinh_functions(test_tensor), sinh_functions(val));
  failed |= verify_result(cosh_functions(test_tensor), cosh_functions(val));
  failed |= verify_result(tanh_functions(test_tensor), tanh_functions(val));
  failed |= verify_result(abs_functions(test_tensor1), abs_functions(val1));

  std::cout << "Transcendental functions test passed: " << !failed << std::endl;

  return failed;
}
