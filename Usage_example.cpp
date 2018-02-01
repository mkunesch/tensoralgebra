#include "Tensor.hpp"
#include "TensorAlgebra.hpp"
#include <iostream>

int main() {
  using TwoTensor = tensoralgebra::Tensor<2, double, 2>;

  // By default a tensor is 3 dimensional (in applications this is the most
  // common due to our world being 3 dimensional) and dataype double.
  // Hence the following is the same as Tensor<1, double, 3>:
  using OneTensor = tensoralgebra::Tensor<1>;

  // Initialisation works with nested initialiser lists:
  TwoTensor tensor = {{1., 2.}, {3., 4.}};
  TwoTensor inverse_metric = {{2., 2.}, {2., 2.}};
  OneTensor vector = {1., 2., 3.};

  // Evaluation is done lazily. For example, the following is not evaluated:
  auto temp = 3 * tensor + 1 / exp(tensor) - sin(tensor);

  // Expressions are evaluated e.g. when writing output:
  std::cout << "Result of 3*tensor + 1/exp(tensor) - sin(tensor): " << temp
            << ".\n";

  // Expressions are also evaluated when assigning to another tensor
  TwoTensor tensor1 = temp; // causes evaluation of temp

  // Basic tensor operations for differential geometry are supported
  // e.g. the trace with respect to a given inverse metric
  std::cout << "Trace: " << tensoralgebra::trace(tensor1, inverse_metric)
            << std::endl;
  // or the dot product (with or without metric)
  std::cout << "Dot: " << tensoralgebra::dot(vector, vector) << std::endl;

  return 0;
}
