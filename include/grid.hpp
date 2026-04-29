#pragma once
#include <Kokkos_Core.hpp>
#include <stdexcept>
#include "run_config.hpp"

struct MacGrid2D {
    int    nx = 0, ny = 0;
    double lx = 0.0, ly = 0.0;
    double dx = 0.0, dy = 0.0;
    static constexpr int ng = 1;

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

    explicit MacGrid2D(const RunConfig& cfg)
        : MacGrid2D(cfg.nx, cfg.ny, cfg.lx, cfg.ly) {}

    KOKKOS_INLINE_FUNCTION int dim() const { return 2; }

    // Cell-centred pressure: owned nx x ny, allocated (nx+2ng) x (ny+2ng)
    KOKKOS_INLINE_FUNCTION int p_nx_owned() const { return nx; }
    KOKKOS_INLINE_FUNCTION int p_ny_owned() const { return ny; }
    KOKKOS_INLINE_FUNCTION int p_nx_total() const { return nx + 2*ng; }
    KOKKOS_INLINE_FUNCTION int p_ny_total() const { return ny + 2*ng; }
    KOKKOS_INLINE_FUNCTION int p_i_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int p_i_end()    const { return ng + nx; }
    KOKKOS_INLINE_FUNCTION int p_j_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int p_j_end()    const { return ng + ny; }

    // Face-centred u: owned (nx+1) x ny, allocated (nx+1+2ng) x (ny+2ng)
    KOKKOS_INLINE_FUNCTION int u_nx_owned() const { return nx + 1; }
    KOKKOS_INLINE_FUNCTION int u_ny_owned() const { return ny; }
    KOKKOS_INLINE_FUNCTION int u_nx_total() const { return nx + 1 + 2*ng; }
    KOKKOS_INLINE_FUNCTION int u_ny_total() const { return ny + 2*ng; }
    KOKKOS_INLINE_FUNCTION int u_i_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int u_i_end()    const { return ng + nx + 1; }
    KOKKOS_INLINE_FUNCTION int u_j_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int u_j_end()    const { return ng + ny; }

    // Face-centred v: owned nx x (ny+1), allocated (nx+2ng) x (ny+1+2ng)
    KOKKOS_INLINE_FUNCTION int v_nx_owned() const { return nx; }
    KOKKOS_INLINE_FUNCTION int v_ny_owned() const { return ny + 1; }
    KOKKOS_INLINE_FUNCTION int v_nx_total() const { return nx + 2*ng; }
    KOKKOS_INLINE_FUNCTION int v_ny_total() const { return ny + 1 + 2*ng; }
    KOKKOS_INLINE_FUNCTION int v_i_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int v_i_end()    const { return ng + nx; }
    KOKKOS_INLINE_FUNCTION int v_j_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int v_j_end()    const { return ng + ny + 1; }
};

struct MacGrid3D {
    int    nx = 0, ny = 0, nz = 0;
    double lx = 0.0, ly = 0.0, lz = 0.0;
    double dx = 0.0, dy = 0.0, dz = 0.0;
    static constexpr int ng = 1;

    MacGrid3D() = default;

    MacGrid3D(int nx, int ny, double lx, double ly)
        : nx(nx), ny(ny), nz(nz), lx(lx), ly(ly), lz(lz),
          dx(lx / nx), dy(ly / ny), dz(lz / nz)
    {
        if (nx  <= 0)   throw std::invalid_argument("MacGrid3D: nx must be positive");
        if (ny  <= 0)   throw std::invalid_argument("MacGrid3D: ny must be positive");
        if (nz  <= 0)   throw std::invalid_argument("MacGrid3D: nz must be positive");
        if (lx <= 0.0)  throw std::invalid_argument("MacGrid3D: lx must be positive");
        if (ly <= 0.0)  throw std::invalid_argument("MacGrid3D: ly must be positive");
        if (lz <= 0.0)  throw std::invalid_argument("MacGrid3D: lz must be positive");
    }

    explicit MacGrid3D(const RunConfig& cfg)
        : MacGrid3D(cfg.nx, cfg.ny, cfg.nz, cfg.lx, cfg.ly, cfg.lz) {}

    KOKKOS_INLINE_FUNCTION int dim() const { return 3; }

    // Cell-centred pressure: owned nx x ny x nz, allocated (nx+2ng) x (ny+2ng) x (nz + 2ng)
    KOKKOS_INLINE_FUNCTION int p_nx_owned() const { return nx; }
    KOKKOS_INLINE_FUNCTION int p_ny_owned() const { return ny; }
    KOKKOS_INLINE_FUNCTION int p_nz_owned() const { return nz; }

    KOKKOS_INLINE_FUNCTION int p_nx_total() const { return nx + 2*ng; }
    KOKKOS_INLINE_FUNCTION int p_ny_total() const { return ny + 2*ng; }
    KOKKOS_INLINE_FUNCTION int p_nz_total() const { return nz + 2*ng; }

    KOKKOS_INLINE_FUNCTION int p_i_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int p_i_end()    const { return ng + nx; }
    KOKKOS_INLINE_FUNCTION int p_j_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int p_j_end()    const { return ng + ny; }
    KOKKOS_INLINE_FUNCTION int p_k_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int p_k_end()    const { return ng + nz; }

    // Face-centred u: owned (nx+1) x ny x nz, allocated (nx+1+2ng) x (ny+2ng) x (nz+2ng)
    KOKKOS_INLINE_FUNCTION int u_nx_owned() const { return nx + 1; }
    KOKKOS_INLINE_FUNCTION int u_ny_owned() const { return ny; }
    KOKKOS_INLINE_FUNCTION int u_nz_owned() const { return nz; }
    
    KOKKOS_INLINE_FUNCTION int u_nx_total() const { return nx + 1 + 2*ng; }
    KOKKOS_INLINE_FUNCTION int u_ny_total() const { return ny + 2*ng; }
    KOKKOS_INLINE_FUNCTION int u_nz_total() const { return nz + 2*ng; }
    
    KOKKOS_INLINE_FUNCTION int u_i_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int u_i_end()    const { return ng + nx + 1; }
    KOKKOS_INLINE_FUNCTION int u_j_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int u_j_end()    const { return ng + ny; }
    KOKKOS_INLINE_FUNCTION int u_k_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int u_k_end()    const { return ng + nz; }

    // Face-centred v: owned nx x (ny+1) x nz, allocated (nx+2ng) x (ny+1+2ng) x (nz+2ng)
    KOKKOS_INLINE_FUNCTION int v_nx_owned() const { return nx; }
    KOKKOS_INLINE_FUNCTION int v_ny_owned() const { return ny + 1; }
    KOKKOS_INLINE_FUNCTION int v_nz_owned() const { return nz; }

    KOKKOS_INLINE_FUNCTION int v_nx_total() const { return nx + 2*ng; }
    KOKKOS_INLINE_FUNCTION int v_ny_total() const { return ny + 1 + 2*ng; }
    KOKKOS_INLINE_FUNCTION int v_nz_total() const { return nz + 2*ng; }
    
    KOKKOS_INLINE_FUNCTION int v_i_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int v_i_end()    const { return ng + nx; }
    KOKKOS_INLINE_FUNCTION int v_j_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int v_j_end()    const { return ng + ny + 1; }
    KOKKOS_INLINE_FUNCTION int v_k_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int v_k_end()    const { return ng + nz; }
    
    // Face-centred w: owned nx x ny x (nz+1), allocated (nx+2ng) x (ny+2ng) x (nz+1+2ng)
    KOKKOS_INLINE_FUNCTION int w_nx_owned() const { return nx; }
    KOKKOS_INLINE_FUNCTION int w_ny_owned() const { return ny; }
    KOKKOS_INLINE_FUNCTION int w_nz_owned() const { return nz + 1; }

    KOKKOS_INLINE_FUNCTION int v_nx_total() const { return nx + 2*ng; }
    KOKKOS_INLINE_FUNCTION int v_ny_total() const { return ny + 2*ng; }
    KOKKOS_INLINE_FUNCTION int v_nz_total() const { return nz + 1 + 2*ng; }
    
    KOKKOS_INLINE_FUNCTION int v_i_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int v_i_end()    const { return ng + nx; }
    KOKKOS_INLINE_FUNCTION int v_j_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int v_j_end()    const { return ng + ny; }
    KOKKOS_INLINE_FUNCTION int v_k_begin()  const { return ng; }
    KOKKOS_INLINE_FUNCTION int v_k_end()    const { return ng + nz + 1; }
};