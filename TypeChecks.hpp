#ifndef _TENSORALGEBRA_TYPECHECKS_HPP
#define _TENSORALGEBRA_TYPECHECKS_HPP

#include <type_traits>
#include <utility>

namespace tensoralgebra {
/// make_void is just C++17's std::void - included to be C++14 compatible.
template <typename...> using make_void = void;

/// Compile time check whether the template parameter has a given size
/** Member "value" is false if the size doesn't match or the template parameter
 * has no size() function defined. Otherwise it is true. */
template <typename T1, size_t Size, typename Helper = void>
struct has_size : public std::false_type {};

template <typename T1, size_t Size>
struct has_size<T1, Size, make_void<decltype(std::decay_t<T1>::size())>> {
  static constexpr bool value = (std::decay_t<T1>::size() == Size);
};
// End: compile time check whether the template parameters have the same size

/// Compile time check whether the template parameter has a given rank
/** Member "value" is false if the rank doesn't match or the template parameter
 * has no size() function defined. Otherwise it is true. */
template <typename T1, size_t Rank, typename Helper = void>
struct has_rank : public std::false_type {};

template <typename T1, size_t Rank>
struct has_rank<T1, Rank, make_void<decltype(std::decay_t<T1>::rank())>> {
  static constexpr bool value = (std::decay_t<T1>::rank() == Rank);
};
// End: compile time check whether the template parameters have the same size

/// Compile time check whether the template parameters have the same size.
/** Member "value" is false if the sizes don't match or the template parameters
 * have no size() function defined. Otherwise it is true. */
template <typename T1, typename T2, typename Helper = void>
struct are_same_size : public std::false_type {};

template <typename T1, typename T2>
struct are_same_size<T1, T2,
                     make_void<decltype(std::decay_t<T1>::size()),
                               decltype(std::decay_t<T2>::size())>> {
  static constexpr bool value =
      (std::decay_t<T1>::size() == std::decay_t<T2>::size());
};
// End: compile time check whether the template parameters have the same size

/// Compile time check whether the template parameters have the same rank.
/** Member "value" is false if the ranks don't match or the template parameters
 * have no rank() function defined. Otherwise it is true. */
template <typename T1, typename T2, typename Helper = void>
struct are_same_rank : public std::false_type {};

template <typename T1, typename T2>
struct are_same_rank<T1, T2,
                     make_void<decltype(std::decay_t<T1>::rank()),
                               decltype(std::decay_t<T2>::rank())>> {
  static constexpr bool value =
      (std::decay_t<T1>::rank() == std::decay_t<T2>::rank());
};
// End: compile time check whether the template parameters have the same rank
} // namespace tensoralgebra

#endif
