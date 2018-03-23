#ifndef _TENSORALGEBRA_TESTS_FUNCTIONSTEST_HPP
#define _TENSORALGEBRA_TESTS_FUNCTIONSTEST_HPP

#include "Tensor.hpp"
#include "TestingUtilities.hpp"
#include <cmath>

#define define_transcendental_test_function(FUNCTION)                          \
  template <typename T> T FUNCTION##_functions(T t) {                          \
    t = FUNCTION(FUNCTION(t));                                                 \
    return t;                                                                  \
  }

// clang-format off
define_transcendental_test_function(exp)
define_transcendental_test_function(log)
define_transcendental_test_function(log10)
define_transcendental_test_function(sqrt)
define_transcendental_test_function(sin)
define_transcendental_test_function(cos)
define_transcendental_test_function(tan)
define_transcendental_test_function(asin)
define_transcendental_test_function(acos)
define_transcendental_test_function(atan)
define_transcendental_test_function(sinh)
define_transcendental_test_function(cosh)
define_transcendental_test_function(tanh)
using std::abs; // Prevents bug whereby C's abs(int) is called
define_transcendental_test_function(abs)
// clang-format on
#undef define_transcendental_test_function

    bool test_transcendental_functions() {
  // Need several values to please functions which are picky about their domain
  const double val = 1.1;
  const double val1 = -0.1;
  const double val2 = 0.9;
  using TwoTensor = tensoralgebra::Tensor<2, double, 2>;
  TwoTensor test_tensor = val;
  TwoTensor test_tensor1 = val1;
  TwoTensor test_tensor2 = val2;

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

  print_result("Transcendental functions test", !failed);

  return failed;
}

#endif
