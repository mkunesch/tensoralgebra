#ifndef _TENSORALGEBRA_TESTS_SUMEVALUATIONORDERTEST_HPP
#define _TENSORALGEBRA_TESTS_SUMEVALUATIONORDERTEST_HPP

#include "Tensor.hpp"
#include "TestingUtilities.hpp"
#include <vector>

// This test ensures that arithmetic expressions are evaluated lazily and in the
// correct order. In particular, for a two tensor the [0][0] component should be
// evaluated completely first, followed by [0][1] etc. The test fails if
// expressions are evaluated prematurely or they are evaluated in the wrong
// order, e.g. if part of [0][1] is evaluated before [0][0] has been finished.

// A helper class that detects additions and stores its position in the tensor
// to the operation tracker
class AdditionDetector {
  double m_data;
  size_t m_comp_index;
  std::vector<size_t> *m_operation_tracker;

public:
  AdditionDetector() = default;
  AdditionDetector(double d, size_t index,
                   std::vector<size_t> *operation_tracker)
      : m_data(d), m_comp_index(index), m_operation_tracker(operation_tracker) {
  }

  AdditionDetector &operator+=(double value) {
    m_operation_tracker->push_back(m_comp_index);
    m_data += value;
    return *this;
  }

  AdditionDetector &operator+=(AdditionDetector value) {
    m_operation_tracker->push_back(m_comp_index);
    m_data += value.m_data;
    return *this;
  }
};

AdditionDetector operator+(AdditionDetector sum1, AdditionDetector sum2) {
  sum1 += sum2;
  return sum1;
}

bool test_sum_evaluation_order() {
  std::vector<size_t> operation_tracker;
  tensoralgebra::Tensor<2, AdditionDetector, 2> test = {
      {AdditionDetector(0., 0, &operation_tracker),
       AdditionDetector(1., 1, &operation_tracker)},
      {AdditionDetector(2., 2, &operation_tracker),
       AdditionDetector(3., 3, &operation_tracker)}};

  test += test + test + test;

  // Correct order is all the [0][0] = 0 first, then [0][1] = 1, etc.
  std::vector<size_t> correct_order = {0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3};
  const bool failed = (operation_tracker != correct_order);
  print_result("Sum evaluation order test", !failed);

  if (failed) {
    std::cout << "The evaluation order was:\n";
    for (auto &element : operation_tracker) {
      std::cout << " " << element;
    }
  }

  return failed;
}

#endif
