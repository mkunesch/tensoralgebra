#include "Tensor.hpp"
#include "TensorOperations.hpp"
#include "TypeChecks.hpp"
#include <array>
#include <iostream>

int main() {
  using TwoTensor = tensoralgebra::Tensor<2, double, 2>;

  // By default a tensor is 3 dimensional (in applications this is the most
  // common due to our world being 3 dimensional) and dataype double.
  // Hence the following is the same as Tensor<1, double, 3>:
  using OneTensor = tensoralgebra::Tensor<1>;

  // Initialisation works with nested initialiser lists:
  TwoTensor tensor = {{1., 2.}, {3., 4.}};
  TwoTensor inverse_metric = 2.;
  OneTensor vector = {1., 2., 3.};

  // Evaluation is done lazily. For example, the following is _not_ evaluated:
  auto unevaluated = 3 * tensor + 1 / exp(tensor) - sin(tensor);

  // Expressions are evaluated when writing output:
  std::cout << "Result of 3*tensor + 1/exp(tensor) - sin(tensor): "
            << unevaluated << ".\n";

  // ... when assigning to another tensor
  TwoTensor tensor1 = unevaluated;

  // ... when applying as many indices as given by the rank:
  auto evaluated = unevaluated[0][0];

  // ... but not if the number of supplied indices is smaller than the rank
  auto unevaluated1 = unevaluated[0];

  // Basic tensor operations for differential geometry are supported
  // e.g. the trace with respect to a given inverse metric
  std::cout << "Trace: " << tensoralgebra::trace(tensor1, inverse_metric)
            << std::endl;
  // ... the outer product
  std::cout << "Outer: " << tensoralgebra::outer(vector, vector) << std::endl;
  // ... or the dot product (with or without metric)
  std::cout << "Dot: " << tensoralgebra::dot(vector, vector) << std::endl;
  std::cout << "Dot: " << tensoralgebra::dot(tensor, inverse_metric)
            << std::endl;

  // Tensors have an iterator for each dimension. This allows e.g. the use of
  // range based for loops:
  double sum = 0.;
  for (auto &row : tensor) {
    for (auto &element : row) {
      sum += element;
    }
  }

  return 0;
}
