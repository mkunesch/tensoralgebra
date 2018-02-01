#pragma once

#include <array>
#include <initializer_list>

namespace tensoralgebra {

/// tensoralgebra::Array is almost exactly like std::array.
/** The only difference to std:array is that it provides a initializer
 * list constructor which allows nested list initialisation
 * of nested Array types.
 */
template <typename data_t, size_t N>
class Array : public std::array<data_t, N> {
public:
  Array() = default;

  Array(const std::initializer_list<data_t> &list);
};

// This constructor (usually) introduces an extra copy unfortunately but it
// allows initialisation using a nested initializer lists.
template <typename data_t, size_t N>
Array<data_t, N>::Array(const std::initializer_list<data_t> &list) {
  std::copy(list.begin(), list.end(), (*this).begin());
}

} // namespace tensoralgebra
