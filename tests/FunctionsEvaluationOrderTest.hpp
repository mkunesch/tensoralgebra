#ifndef _TENSORALGEBRA_TESTS_FUNCTIONSEVALUATIONORDERTEST_HPP
#define _TENSORALGEBRA_TESTS_FUNCTIONSEVALUATIONORDERTEST_HPP

#include "Tensor.hpp"
#include "TestingUtilities.hpp"
#include <cmath>
#include <vector>

// This test ensures that expressions involving functions are evaluated lazily
// and in the correct order. In particular, for a two tensor the [0][0]
// component should be evaluated completely first, followed by [0][1] etc. The
// test fails if expressions are evaluated prematurely or they are evaluated in
// the wrong order, e.g. if part of [0][1] is evaluated before [0][0] has been
// finished.

// A helper class that detects exponentiation and stores its position in the
// tensor to the operation tracker
class ExpDetector {
  double m_data;
  size_t m_comp_index;
  std::vector<size_t> *m_operation_tracker;

public:
  ExpDetector() = default;
  ExpDetector(double d, size_t index, std::vector<size_t> *operation_tracker)
      : m_data(d), m_comp_index(index), m_operation_tracker(operation_tracker) {
  }

  friend ExpDetector exp(ExpDetector detector) {
    detector.m_operation_tracker->push_back(detector.m_comp_index);
    detector.m_data = exp(detector.m_data);
    return detector;
  }
};

int test_transcendental_evaluation_order() {
  std::vector<size_t> operation_tracker;
  tensoralgebra::Tensor<2, ExpDetector, 2> test = {
      {ExpDetector(0., 0, &operation_tracker),
       ExpDetector(1., 1, &operation_tracker)},
      {ExpDetector(2., 2, &operation_tracker),
       ExpDetector(3., 3, &operation_tracker)}};

  test = exp(exp(exp(test)));

  // Correct order is all the [0][0] = 0 first, then [0][1] = 1, etc.
  std::vector<size_t> correct_order = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  const bool failed = (operation_tracker != correct_order);
  print_result("Transcendental evaluation order test", !failed);

  if (failed) {
    std::cout << "The evaluation order was:\n";
    for (auto &element : operation_tracker) {
      std::cout << " " << element;
    }
  }

  return failed;
}

#endif
