#ifndef _TENSORALGEBRA_TESTS_TESTINGUTILITIES_HPP
#define _TENSORALGEBRA_TESTS_TESTINGUTILITIES_HPP

#include <iostream>

void print_result(const std::string &test_name, bool passed) {
  std::cout << test_name << ": ";
  if (passed) {
    std::cout << "passed.\n";
  } else {
    std::cout << "!!! FAILED !!!.\n";
  }
}

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
#endif
