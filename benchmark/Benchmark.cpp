#include "NaiveTensor.hpp"
#include "Tensor.hpp"
#include "TensorOperations.hpp"
#include <benchmark/benchmark.h>

static const size_t SIZE = 4;

static void run_naive(benchmark::State &state) {
  tensoralgebra::NaiveTensor<4, double, SIZE> tensor;
  const tensoralgebra::NaiveTensor<2, double, SIZE> tensor1 = 1.;
  const tensoralgebra::NaiveTensor<2, double, SIZE> tensor2 = 2.;
  const tensoralgebra::NaiveTensor<2, double, SIZE> tensor3 = 3.;
  while (state.KeepRunning()) {
    tensor = outer(tensor1, tensor2) + outer(tensor2, tensor1) +
             outer(tensor1, tensor3) + outer(tensor3, tensor1) +
             outer(tensor2, tensor3) + outer(tensor3, tensor2);
    benchmark::DoNotOptimize(tensor);
  }
}

static void run_expression(benchmark::State &state) {
  tensoralgebra::Tensor<4, double, SIZE> tensor;
  const tensoralgebra::Tensor<2, double, SIZE> tensor1 = 1.;
  const tensoralgebra::Tensor<2, double, SIZE> tensor2 = 2.;
  const tensoralgebra::Tensor<2, double, SIZE> tensor3 = 3.;
  while (state.KeepRunning()) {
    tensor = outer(tensor1, tensor2) + outer(tensor2, tensor1) +
             outer(tensor1, tensor3) + outer(tensor3, tensor1) +
             outer(tensor2, tensor3) + outer(tensor3, tensor2);
    benchmark::DoNotOptimize(tensor);
  }
}

static void run_loop(benchmark::State &state) {
  tensoralgebra::Tensor<4, double, SIZE> tensor;
  const tensoralgebra::Tensor<2, double, SIZE> tensor1 = 1.;
  const tensoralgebra::Tensor<2, double, SIZE> tensor2 = 2.;
  const tensoralgebra::Tensor<2, double, SIZE> tensor3 = 3.;
  while (state.KeepRunning()) {
    for (size_t i = 0; i < SIZE; ++i) {
      for (size_t j = 0; j < SIZE; ++j) {
        for (size_t k = 0; k < SIZE; ++k) {
          for (size_t l = 0; l < SIZE; ++l) {
            tensor[i][j][k][l] =
                tensor1[i][j] * tensor2[k][l] + tensor2[i][j] * tensor1[k][l] +
                tensor1[i][j] * tensor3[k][l] + tensor3[i][j] * tensor1[k][l] +
                tensor2[i][j] * tensor3[k][l] + tensor3[i][j] * tensor2[k][l];
          }
        }
      }
    }
    benchmark::DoNotOptimize(tensor);
  }
}

BENCHMARK(run_naive)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(run_expression)->Repetitions(10)->ReportAggregatesOnly(true);
BENCHMARK(run_loop)->Repetitions(10)->ReportAggregatesOnly(true);

BENCHMARK_MAIN()
