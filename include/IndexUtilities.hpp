#ifndef _TENSORALGEBRA_INDEXUTILITIES_HPP
#define _TENSORALGEBRA_INDEXUTILITIES_HPP

#include <cstddef>
#include <utility>

namespace tensoralgebra {

template <typename T> decltype(auto) apply_indices(T &&obj) {
  return std::forward<T>(obj);
}

template <typename T, typename... IndexTs>
decltype(auto) apply_indices(T &&obj, size_t dir, IndexTs... dirs) {
  return apply_indices(std::forward<T>(obj)[dir], dirs...);
}

/// Inserts the index given by insert_dir at the location given by Position
// Note: all the eval functions below will generate a new value - no need to
// worry about allowing return by reference.
template <size_t Position> struct IndexInserter {
  template <typename T, typename... IndexTs>
  static auto eval(const T &obj, size_t insert_dir, size_t dir,
                   IndexTs... dirs) {
    return IndexInserter<Position - 1>::eval(obj[dir], insert_dir, dirs...);
  }
};

template <> struct IndexInserter<1> {
  template <typename T, typename... IndexTs>
  static decltype(auto) eval(const T &obj, size_t insert_dir, IndexTs... dirs) {
    return obj.eval(insert_dir, dirs...);
  }
};

/// Inserts the index given by contract_dir into two locations given
/// by Position1 and Position2.
template <size_t Position1, size_t Position2> struct IndexContracter {
  template <typename T, typename... IndexTs>
  static auto eval(const T &obj, size_t contract_dir, size_t dir,
                   IndexTs... dirs) {
    static_assert(Position1 < Position2,
                  "First index must be smaller than second");
    return IndexContracter<Position1 - 1, Position2 - 1>::eval(
        obj[dir], contract_dir, dirs...);
  }
};

template <size_t Position2> struct IndexContracter<1, Position2> {
  template <typename T, typename... IndexTs>
  static auto eval(const T &obj, size_t contract_dir, IndexTs... dirs) {
    return IndexInserter<Position2 - 1>::eval(obj[contract_dir], contract_dir,
                                              dirs...);
  }
};

} // namespace tensoralgebra

#endif
