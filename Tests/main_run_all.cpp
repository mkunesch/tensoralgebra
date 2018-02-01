#include <cmath>
#include <iostream>

#include "Tensor.hpp"
#include "math_test.hpp"
#include "sum_evaluation_order_test.hpp"
#include "transcendental_evaluation_order_test.hpp"

int main() {
  using namespace tensoralgebra;
  using test_tensor = Tensor<2, double, 2>;
  using test_tensor1 = Tensor<3, double, 3>;

  // Test the type checking tools (if they are wrong the code would not
  // compile but the compiler errors would be somewhat unhelpful)
  static_assert(are_same_size<test_tensor, test_tensor>::value == true,
                "Should yield: same size.");
  static_assert(are_same_size<test_tensor, int>::value == false,
                "Should yield: not same size.");
  static_assert(are_same_size<test_tensor, test_tensor1>::value == false,
                "Should yield: not same size.");
  static_assert(are_same_rank<test_tensor, test_tensor>::value == true,
                "Should yield: same rank.");
  static_assert(are_same_rank<test_tensor, test_tensor1>::value == false,
                "Should yield: not same rank.");
  static_assert(is_tensor_expression<test_tensor>::value == true,
                "Should yield: it is a tensor expression.");
  static_assert(is_tensor_expression<double>::value == false,
                "Should yield: not a tensor expression.");

  bool failed = false;
  failed |= test_sum_evaluation_order();
  failed |= test_transcendental_evaluation_order();
  failed |= test_arithmetic_operations();
  failed |= test_transcendental_functions();

  return failed;
}
