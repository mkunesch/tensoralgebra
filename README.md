# tensoralgebra

Tensoralgebra provides an implementation of a tensor in physics and differential
geometry: a multidimensional array of the same size in all directions.

The main features of tensoralgebra are:
* Operations involving tensors are evaluated lazily to avoid unnecessary
  temporary tensors and repeated loops. Evaluation is done component by
  component in row-major order once the final expression is known.
* The most important tensor operations are included; e.g. trace, dot, outer
  product, raising/lowering indices, ...
* C-style indices (such as `tensor[i][j]`) can be used
  without a performance penalty.
* Size and rank are both fixed at compile time. This makes the code faster
  for small values of `size*rank`, makes it possible to overload based on rank or
  size, and to check the compatibility of two tensors at compile time.
* While functions for lowering and raising indices are implemented,
  tensoralgebra does not check the index type (e.g. it does not differentiate
  between a vector and a covector). In my opinion, it is much easier to do this by hand.
* The implementation is optimised for small sizes (i.e. for a 4-vector or a 4x4
  matrix, not for a vector with 100,000 components).
* Storing and passing around unevaluated expressions of several terms is allowed and should not
  result in undefined behaviour.

Tensoralgebra is a feasibility study for a better tensor implementation in the
numerical general relativity code [GRChombo](https://github.com/GRChombo/GRChombo).

## Prerequisites and usage example
Tensoralgebra requires C++14 or higher and has been tested with gcc-7, clang-6, and
icc-17. Some older compiler versions lead to significantly worse performance.
Tensoralgebra is header-only; all that is needed is to include `Tensor.hpp` and
`TensorOperations.hpp` (for tensor operations like the dot product).

Initialisation of a tensor works with nested initialiser lists:
```
  tensoralgebra::Tensor<2, double, 2> tensor = {{1., 2.}, {3., 4.}};
  tensoralgebra::Tensor<2, double, 2> inverse_metric = {{2., 2.}, {2., 2.}};
```

By default, a tensor is 3 dimensional (as this is the most common case in
physics) and tensor components are of type double.
Hence, the following is the same as Tensor<1, double, 3>:
```
  using Vector = tensoralgebra::Tensor<1>;
  Vector vector = {1., 2., 3.};
```

Evaluation is done lazily. For example, the following is not evaluated:
```
  auto unevaluated = 3 * tensor + 1 / exp(tensor) - sin(tensor);
```

Expressions are evaluated e.g. when writing output
```
  std::cout << "Result: " << unevaluated << ".\n";
```

when assigning to another tensor
```
  tensoralgebra::Tensor<2, double, 2> tensor1 = unevaluated;
```

when applying as many indices as given by the rank
```
  auto evaluated = unevaluated[0][0];
```

but not if the number of supplied indices is smaller than the rank
```
  auto unevaluated1 = unevaluated[0];
```

Basic tensor operations for differential geometry are supported,
e.g. the trace with respect to a given inverse metric
```
  std::cout << "Trace: " << tensoralgebra::trace(tensor1, inverse_metric);
```

the outer product
```
  std::cout << "Outer: " << tensoralgebra::outer(vector, vector);
```

or the dot product (with or without metric)
```
  std::cout << "Dot: " << tensoralgebra::dot(vector, vector);
  std::cout << "Dot: " << tensoralgebra::dot(tensor, inverse_metric);
```

Tensors have an iterator for each dimension. Among others, this allows the use of
range-based for loops:
```
  double sum = 0.;
  for (auto &row : tensor) {
    for (auto &element : row) {
      sum += element;
    }
  }
```

## Tests
The tests folder contains several tests which ensure that
* the operations are correct (even for more complicated expressions with nested
  functions etc.).
* evaluation takes place lazily, that is nothing is evaluated until
  it is necessary and the full expression is known.
* evaluation takes place in the right order, that is component by component in
  row major order. E.g. for a rank-2 expression [0][0] should be evaluated
  completely before starting with [0][1].

## Performance
The benchmark folder includes a benchmark which compares the runtime of a naive
tensor implementation, an implementation using explicit loops, and the expression
template implementation provided in tensoralgebra.

The example output below was obtained with clang-6 and shows that, with a recent
compiler, the expression template implementation gives the same
performance as an implementation using explicit loops.

```
Run on (4 X 3300 MHz CPUs)
------------------------------------------------------------------------
Benchmark                                 Time           CPU Iterations
------------------------------------------------------------------------
run_naive/repeats:10_mean              1162 ns       1152 ns     587224
run_naive/repeats:10_stddev              18 ns         12 ns          0
run_loop/repeats:10_mean                233 ns        231 ns    3064986
run_loop/repeats:10_stddev                6 ns          6 ns          0
run_expression/repeats:10_mean          231 ns        230 ns    3088285
run_expression/repeats:10_stddev          4 ns          4 ns          0
```

## Implementation notes
A rank-R tensor is implemented recursively as an array of rank-(R-1) tensors.
Lazy evaluation is achieved using expression templates.
Expression templates involving rvalues store lvalues instead of lvalue
references, so that they can be passed around without running into dangling
references.

## Contributing
I welcome all feedback, comments, criticism, feature requests, and contributions,
including pull requests.

## License
Since tensoralgebra is a testing ground for
[GRChombo](https://github.com/GRChombo/GRChombo), it is released under the same
license (3-clause BSD).
