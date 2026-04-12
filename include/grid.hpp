#pragma once
#include <Kokkos_Core.hpp>
#include <stdexcept>

struct MacGrid2D {
    int    nx = 0, ny = 0;
    double lx = 0.0, ly = 0.0;
    double dx = 0.0, dy = 0.0;

    MacGrid2D() = default;

    MacGrid2D(int nx, int ny, double lx, double ly)
        : nx(nx), ny(ny), lx(lx), ly(ly),
          dx(lx / nx), dy(ly / ny)
    {
        if (nx  <= 0)   throw std::invalid_argument("MacGrid2D: nx must be positive");
        if (ny  <= 0)   throw std::invalid_argument("MacGrid2D: ny must be positive");
        if (lx <= 0.0)  throw std::invalid_argument("MacGrid2D: lx must be positive");
        if (ly <= 0.0)  throw std::invalid_argument("MacGrid2D: ly must be positive");
    }

    KOKKOS_INLINE_FUNCTION int dim() const { return 2; }

    // Cell-centred pressure: nx x ny
    KOKKOS_INLINE_FUNCTION int p_nx() const { return nx; }
    KOKKOS_INLINE_FUNCTION int p_ny() const { return ny; }

    // Face-centred u: (nx+1) x ny
    KOKKOS_INLINE_FUNCTION int u_nx() const { return nx + 1; }
    KOKKOS_INLINE_FUNCTION int u_ny() const { return ny; }

    // Face-centred v: nx x (ny+1)
    KOKKOS_INLINE_FUNCTION int v_nx() const { return nx; }
    KOKKOS_INLINE_FUNCTION int v_ny() const { return ny + 1; }
};
