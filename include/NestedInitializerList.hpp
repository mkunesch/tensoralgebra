#ifndef _TENSORALGEBRA_NESTEDINITIALIZERLIST_HPP
#define _TENSORALGEBRA_NESTEDINITIALIZERLIST_HPP
#include <initializer_list>

namespace tensoralgebra {

// Nested list type
template <typename T, size_t Depth> struct nested_list_helper {
  using type =
      std::initializer_list<typename nested_list_helper<T, Depth - 1>::type>;
};

template <typename T> struct nested_list_helper<T, 0> { using type = T; };

template <typename T, size_t Depth>
using NestedInitializerList = typename nested_list_helper<T, Depth>::type;

} // namespace tensoralgebra

#endif
