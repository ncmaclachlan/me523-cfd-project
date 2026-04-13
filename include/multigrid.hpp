#pragma once
#include <Kokkos_Core.hpp>

namespace mg {

// Wrap periodic index for cell-centered field with ghost layer.
// Owned indices: [1, n]. Ghost: 0 and n+1.
KOKKOS_INLINE_FUNCTION
int pwrap(int idx, int n) {
    if (idx < 1)  return idx + n;
    if (idx > n)  return idx - n;
    return idx;
}

} // namespace mg
