#ifndef _TENSORALGEBRA_TESTS_TENSOROPERATIONSTEST_HPP
#define _TENSORALGEBRA_TESTS_TENSOROPERATIONSTEST_HPP

#include "Tensor.hpp"
#include "TensorOperations.hpp"
#include "TestingUtilities.hpp"

// This file tests all (currently implemented) tensor operations which change
// the rank of the result, or at least involve some contraction (e.g. matrix
// multiplication).

bool test_dot() {
  bool failed = false;

  // Note: since vectors are treated separately in the implementation of Dot, we
  // have to check several possible combinations of ranks.

  // Test dot for a vector product
  tensoralgebra::Tensor<1, double, 3> vector1 = {1., 2., 3.};
  tensoralgebra::Tensor<1, double, 3> vector2 = {3., 2., 1.};

  failed |= (dot(vector1, vector2) != 10);

  // Test dot for a matrix product
  tensoralgebra::Tensor<2, double, 2> tensor1 = {{1., 2.}, {3., 4.}};
  tensoralgebra::Tensor<2, double, 2> tensor2 = {{4., 3.}, {2., 1.}};

  tensoralgebra::Tensor<2, double, 2> correct_matrix = {{8., 5.}, {20., 13.}};
  failed |= (dot(tensor1, tensor2) != correct_matrix);

  // Test dor for matrix-vector
  tensoralgebra::Tensor<2, double, 3> tensor3 = {
      {1., 2., 3.}, {4., 5., 6.}, {7., 8., 9.}};
  tensoralgebra::Tensor<1, double, 3> correct_vector = {14., 32., 50.};
  failed |= (dot(tensor3, vector1) != correct_vector);

  return failed;
}

bool test_outer() {
  tensoralgebra::Tensor<1, double, 2> vector = {2., 3};
  tensoralgebra::Tensor<2, double, 2> tensor = {{1., 2.}, {3., 4.}};
  tensoralgebra::Tensor<3, double, 2> correct_tensor = {{{2., 4.}, {6., 8.}},
                                                        {{3., 6.}, {9., 12.}}};
  tensoralgebra::Tensor<3, double, 2> correct_tensor1 = {{{2., 3.}, {4., 6.}},
                                                         {{6., 9.}, {8., 12.}}};

  bool failed = false;
  failed |= (outer(vector, tensor) != correct_tensor);
  failed |= (outer(tensor, vector) != correct_tensor1);

  // Return failed = true if the tensors don't match
  return failed;
}

bool test_rank_changing_operations() {

  bool failed = false;
  failed |= test_dot();
  failed |= test_outer();

  print_result("Rank-changing operations test", !failed);

  return failed;
}

#endif
