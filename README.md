# tensoralgebra

Tensoralgebra provides an implementation of a tensor in physics and differential
geometry: a multidimensional array of the same size in all directions.
It is a fesibility study for a better tensor implementation in the
numerical general relativity code [GRChombo](https://github.com/GRChombo/GRChombo).

The main features of tensoralgebra:
* Operations involving tensors are evaluated lazily to avoid unnecessary temporary tensors.
Evaluation takes place only once the final expression is known and it is done component by
component in row-major order.
* Dimension and rank are both fixed at compile time. This makes the code faster
  and allows compile time checks of compatibility and easy specialisation of
functions for a specific dimension.
* The most important tensor operations are included: e.g. trace, dot,
  make_trace_free, ...
* While functions for lowering and raising indices are implemented and parameter
  names reflect whether indices are up or down, the package does not convert the
index type automatically. In my opinion it is much easier and clearer to do this
by hand explicitly.
* The implementation is optimised for small sizes (i.e. for a 4-vector or a 4x4
  matrix, not for a vector with 100,000 components) and small datatypes (e.g.
for Tensor<3, double> or Tensor<3, int>).

## Implementation note
A rank-R tensor is implemented recursively as an array of rank-(R-1) tensors.
Lazy evaluation is achieved using expression templates.

## Prerequisites and usage examples
Tensoralgebra requires C++14 or higher and has been tested with gcc-7, clang-6, and
icc-17. Some older compiler versions yield significantly worse performance. Tensoralgebra is header-only so all that is needed is to include `Tensor.hpp`
and `TensorOperations.hpp` (for tensor operations like the dot product).

Initialisation of a tensor works with nested initialiser lists:
```
  tensoralgebra::Tensor<2, double, 2> tensor = {{1., 2.}, {3., 4.}};
  tensoralgebra::Tensor<2, double, 2> inverse_metric = {{2., 2.}, {2., 2.}};
```

By default a tensor is 3 dimensional (in applications this is the most
common due to our world being 3 dimensional) and dataype double.
Hence the following is the same as Tensor<1, double, 3>:
```
  using Vector = tensoralgebra::Tensor<1>;
  Vector vector = {1., 2., 3.};
```

Evaluation is done lazily. For example, the following is not evaluated:
```
  auto temp = 3 * tensor + 1 / exp(tensor) - sin(tensor);
```

Expressions are evaluated e.g. when writing output
```
  std::cout << "Result of 3*tensor + 1/exp(tensor) - sin(tensor): " << temp << ".\n";
```

when assigning to another tensor
```
  TwoTensor tensor1 = temp; // temp is evaluated, results are written into tensor1
```

when applying as many indices as given by the rank
```
  auto evaluated = tensor[0][0];
```

but not if the number of supplied indices is smaller than the rank
```
  auto unevaluated1 = tensor[0];
```

Basic tensor operations for differential geometry are supported
e.g. the trace with respect to a given inverse metric
```
  std::cout << "Trace: " << tensoralgebra::trace(tensor1, inverse_metric) << ".\n";
```

the outer product
```
  std::cout << "Outer: " << tensoralgebra::outer(vector, vector) << std::endl;
```

or the dot product (with or without metric)
```
  std::cout << "Dot: " << tensoralgebra::dot(vector, vector) << std::endl;
  std::cout << "Dot: " << tensoralgebra::dot(tensor, inverse_metric)
            << std::endl;
```

Tensors have an iterator for each dimension. This allows e.g. the use of
range based for loops:
```
  double sum = 0.;
  for (auto &row : tensor) {
    for (auto &element : row) {
      sum += element;
    }
  }
```

## Tests
The folder `Tests` contains several tests which ensure that
* the operations are correct (even for more complicated expressions with nested
  functions etc.)
* evaluation actually takes place lazily, that is nothing is evaluated until
  it really is necessary and the full expression is known.
* evaluation takes place in the right order, that is component by component in
  row major order. E.g. for a rank-2 expression [0][0] should be evaluated
  completely before starting with [0][1].

## Contributing
I welcome all feedback, comments, criticism, feature requests, contributions,
and pull requests.

## License
Since tensoralgebra is a testing ground for
[GRChombo](https://github.com/GRChombo/GRChombo), it is released under the same
license (3-clause BSD).

## Acknowledgements
A huge thank you goes to Matt Godbolt for his [compiler explorer](https://godbolt.org).

## Performance
The Benchmark folder includes a benchmark which compares the runtime of a naive
tensor implementation, an implementation using explicit loops, and the expression
template implementation provided in this tensoralgebra package.

The example output below was obtained with clang-6 and shows that indeed
expression templates can yield the same performance as an explicit loop
implementation on a recent compiler.

Run on (4 X 3300 MHz CPUs)
2018-03-21 19:28:31
------------------------------------------------------------------------
Benchmark                                 Time           CPU Iterations
------------------------------------------------------------------------
run_naive/repeats:10_mean              1162 ns       1152 ns     587224
run_naive/repeats:10_stddev              18 ns         12 ns          0
run_loop/repeats:10_mean                233 ns        231 ns    3064986
run_loop/repeats:10_stddev                6 ns          6 ns          0
run_expression/repeats:10_mean          231 ns        230 ns    3088285
run_expression/repeats:10_stddev          4 ns          4 ns          0
