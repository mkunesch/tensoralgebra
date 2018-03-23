#include <cmath>
#include <iostream>

#include "ArithmeticOperationsTest.hpp"
#include "FunctionsEvaluationOrderTest.hpp"
#include "FunctionsTest.hpp"
#include "RelationalOperatorsTest.hpp"
#include "SumEvaluationOrderTest.hpp"
#include "Tensor.hpp"
#include "TensorOperationsTest.hpp"

int main() {
  using test_tensor = tensoralgebra::Tensor<2, double, 2>;
  using test_tensor1 = tensoralgebra::Tensor<3, double, 3>;
  using test_expression = tensoralgebra::TensorExpression<2, test_tensor, 2>;

  // Test the type checking tools (if they are wrong the code would not
  // compile but the compiler errors would be somewhat unhelpful)
  static_assert(tensoralgebra::are_same_size<test_tensor, test_tensor>::value,
                "Should yield: same size.");
  static_assert(!tensoralgebra::are_same_size<test_tensor, int>::value,
                "Should yield: not same size.");
  static_assert(!tensoralgebra::are_same_size<test_tensor, test_tensor1>::value,
                "Should yield: not same size.");
  static_assert(tensoralgebra::are_same_rank<test_tensor, test_tensor>::value,
                "Should yield: same rank.");
  static_assert(!tensoralgebra::are_same_rank<test_tensor, test_tensor1>::value,
                "Should yield: not same rank.");

  bool failed = false;
  failed |= test_sum_evaluation_order();
  failed |= test_transcendental_evaluation_order();
  failed |= test_arithmetic_operations();
  failed |= test_transcendental_functions();
  failed |= test_relational_operations();
  failed |= test_rank_changing_operations();

  return failed;
}
