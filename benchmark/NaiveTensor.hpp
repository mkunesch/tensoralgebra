#ifndef _TENSORALGEBRA_BENCHMARK_NAIVETENSOR_HPP
#define _TENSORALGEBRA_BENCHMARK_NAIVETENSOR_HPP

#include <array>
#include <cstddef>
#include <iostream>

namespace tensoralgebra {
template <size_t Rank, typename T = double, size_t Size = 3> class NaiveTensor;

template <size_t Rank, typename T, size_t Size>
struct naive_tensor_type_recursion {
  using type = std::array<NaiveTensor<Rank - 1, T, Size>, Size>;
};

// The base case: a rank 1 tensor is just an array
template <typename T, size_t Size>
struct naive_tensor_type_recursion<1, T, Size> {
  using type = std::array<T, Size>;
};

template <size_t Rank, typename T, size_t Size>
class NaiveTensor : public naive_tensor_type_recursion<Rank, T, Size>::type {
public:
  NaiveTensor() = default;

  NaiveTensor(T value) { (*this).fill(value); }

  NaiveTensor<Rank, T, Size> &operator+=(const NaiveTensor<Rank, T, Size> &t) {
    for (size_t i = 0; i < Size; ++i) {
      (*this)[i] += t[i];
    }
    return *this;
  }

  NaiveTensor<Rank, T, Size> &operator-=(const NaiveTensor<Rank, T, Size> &t) {
    for (size_t i = 0; i < Size; ++i) {
      (*this)[i] -= t[i];
    }
    return *this;
  }
};

template <size_t Rank, typename T, size_t Size>
NaiveTensor<Rank, T, Size> operator+(NaiveTensor<Rank, T, Size> in1,
                                     const NaiveTensor<Rank, T, Size> &in2) {
  in1 += in2;
  return in1;
}

template <size_t Rank, typename T, size_t Size>
NaiveTensor<Rank, T, Size> operator-(NaiveTensor<Rank, T, Size> in1,
                                     const NaiveTensor<Rank, T, Size> &in2) {
  in1 -= in2;
  return in1;
}

template <class T, size_t Size>
NaiveTensor<4, T, Size> outer(const NaiveTensor<2, T, Size> &tensor1,
                              const NaiveTensor<2, T, Size> &tensor2) {
  NaiveTensor<4, T, Size> out;
  for (size_t i = 0; i < Size; ++i) {
    for (size_t j = 0; j < Size; ++j) {
      for (size_t k = 0; k < Size; ++k) {
        for (size_t l = 0; l < Size; ++l) {
          out[i][j][k][l] = tensor1[i][j] * tensor2[k][l];
        }
      }
    }
  }
  return out;
}

} // namespace tensoralgebra
#endif
