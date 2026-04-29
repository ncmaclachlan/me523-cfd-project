#pragma once
#include "sim_state.hpp"
#include "pressure_solver.hpp"
#include <Kokkos_Core.hpp>

// Half-open i/j range over the velocity-face DOFs that the viscous solver
// should iterate on. Excludes any face that the BC pins (Dirichlet) or
// extrapolates (zero-gradient outflow) -- including those faces in GS
// produces a residual floor because BC kernels overwrite the GS update
// between sweeps.
struct UVRange { int i_begin, i_end, j_begin, j_end; };

struct LidDrivenCavityBC {
    double lid_velocity = 1.0;

    LidDrivenCavityBC() = default;
    LidDrivenCavityBC(double u_lid) : lid_velocity(u_lid) {}

    // Pressure has homogeneous Neumann (∂p/∂n = 0) on every wall.
    static PressureBCSides pressure_sides() {
        return { PressureBC::Neumann, PressureBC::Neumann,
                 PressureBC::Neumann, PressureBC::Neumann };
    }

    // u-faces at i=u_i_begin and i=u_i_end-1 sit on the left/right walls
    // (Dirichlet 0); skip them. Top/bottom u-faces are half a cell off the
    // wall, so the half-cell-offset values ARE unknowns -- include them.
    static UVRange u_solve_range(const MacGrid2D& g) {
        return { g.u_i_begin() + 1, g.u_i_end() - 1,
                 g.u_j_begin(),     g.u_j_end()     };
    }
    // v-faces at j=v_j_begin and j=v_j_end-1 sit on bottom/top walls.
    static UVRange v_solve_range(const MacGrid2D& g) {
        return { g.v_i_begin(),     g.v_i_end(),
                 g.v_j_begin() + 1, g.v_j_end() - 1 };
    }

    void apply_u(const MacGrid2D& g, Kokkos::View<double**> u) const {
        const double u_lid = lid_velocity;

        // Left/right walls: u-face sits exactly on the wall, enforce no-slip directly
        Kokkos::parallel_for("ldc_u_lr",
            Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
            KOKKOS_LAMBDA(int j) {
                u(g.u_i_begin(),     j) = 0.0;   // left wall
                u(g.u_i_end() - 1,   j) = 0.0;   // right wall
            });

        // Bottom/top: u-face is half a cell from the wall, enforce via ghost cell
        //   bottom (y=0, no-slip): u_ghost = -u_interior  -> avg = 0 at wall
        //   top    (y=Ly, lid):    u_ghost = 2*U_lid - u_interior
        Kokkos::parallel_for("ldc_u_tb",
            Kokkos::RangePolicy<>(g.u_i_begin(), g.u_i_end()),
            KOKKOS_LAMBDA(int i) {
                u(i, g.u_j_begin() - 1) = -u(i, g.u_j_begin());
                u(i, g.u_j_end())       = 2.0 * u_lid - u(i, g.u_j_end() - 1);
            });
    }

    void apply_v(const MacGrid2D& g, Kokkos::View<double**> v) const {
        // Bottom/top walls: v-face sits exactly on the wall, enforce no-slip directly
        Kokkos::parallel_for("ldc_v_tb",
            Kokkos::RangePolicy<>(g.v_i_begin(), g.v_i_end()),
            KOKKOS_LAMBDA(int i) {
                v(i, g.v_j_begin())     = 0.0;   // bottom wall
                v(i, g.v_j_end() - 1)   = 0.0;   // top wall
            });

        // Left/right: v-face is half a cell from the wall, enforce via ghost cell
        //   left  (x=0,  no-slip): v_ghost = -v_interior
        //   right (x=Lx, no-slip): v_ghost = -v_interior
        Kokkos::parallel_for("ldc_v_lr",
            Kokkos::RangePolicy<>(g.v_j_begin(), g.v_j_end()),
            KOKKOS_LAMBDA(int j) {
                v(g.v_i_begin() - 1, j) = -v(g.v_i_begin(), j);
                v(g.v_i_end(),       j) = -v(g.v_i_end() - 1, j);
            });
    }

    void apply(SimState& s) const {
        apply_u(s.grid, s.u);
        apply_v(s.grid, s.v);
    }
};

struct PeriodicBC {
    // All sides periodic for the pressure Poisson problem.
    static PressureBCSides pressure_sides() {
        return { PressureBC::Periodic, PressureBC::Periodic,
                 PressureBC::Periodic, PressureBC::Periodic };
    }

    // Skip the duplicate periodic face at u_i_end()-1 / v_j_end()-1; every
    // other face is an independent unknown.
    static UVRange u_solve_range(const MacGrid2D& g) {
        return { g.u_i_begin(), g.u_i_end() - 1,
                 g.u_j_begin(), g.u_j_end()     };
    }
    static UVRange v_solve_range(const MacGrid2D& g) {
        return { g.v_i_begin(), g.v_i_end(),
                 g.v_j_begin(), g.v_j_end() - 1 };
    }

    void apply_u(const MacGrid2D& g, Kokkos::View<double**> u) const {
        // Sync duplicate periodic face: u at i=begin and i=end-1 are the same
        // physical face.  Average to keep both sides symmetric, then set ghosts.
        Kokkos::parallel_for("periodic_u_sync",
            Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
            KOKKOS_LAMBDA(const int j) {
                const double avg = 0.5 * (u(g.u_i_begin(), j)
                                        + u(g.u_i_end() - 1, j));
                u(g.u_i_begin(),     j) = avg;
                u(g.u_i_end() - 1,   j) = avg;
            });

        // Ghost must skip past the duplicate periodic face (u_i_end()-1 == u_i_begin()
        // after sync) to the last truly independent face at u_i_end()-2.
        Kokkos::parallel_for("periodic_u_x",
            Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
            KOKKOS_LAMBDA(const int j) {
                u(g.u_i_begin() - 1, j) = u(g.u_i_end() - 2, j);
                u(g.u_i_end(),       j) = u(g.u_i_begin() + 1, j);
            });

        Kokkos::parallel_for("periodic_u_y",
            Kokkos::RangePolicy<>(g.u_i_begin(), g.u_i_end()),
            KOKKOS_LAMBDA(const int i) {
                u(i, g.u_j_begin() - 1) = u(i, g.u_j_end() - 1);
                u(i, g.u_j_end())       = u(i, g.u_j_begin());
            });

        Kokkos::parallel_for("periodic_u_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                u(g.u_i_begin() - 1, g.u_j_begin() - 1) = u(g.u_i_end() - 2, g.u_j_end() - 1);
                u(g.u_i_begin() - 1, g.u_j_end())       = u(g.u_i_end() - 2, g.u_j_begin());
                u(g.u_i_end(),       g.u_j_begin() - 1) = u(g.u_i_begin() + 1, g.u_j_end() - 1);
                u(g.u_i_end(),       g.u_j_end())       = u(g.u_i_begin() + 1, g.u_j_begin());
            });
    }

    void apply_v(const MacGrid2D& g, Kokkos::View<double**> v) const {
        // Sync duplicate periodic face: v at j=begin and j=end-1 are the same
        Kokkos::parallel_for("periodic_v_sync",
            Kokkos::RangePolicy<>(g.v_i_begin(), g.v_i_end()),
            KOKKOS_LAMBDA(const int i) {
                const double avg = 0.5 * (v(i, g.v_j_begin())
                                        + v(i, g.v_j_end() - 1));
                v(i, g.v_j_begin())     = avg;
                v(i, g.v_j_end() - 1)   = avg;
            });

        Kokkos::parallel_for("periodic_v_x",
            Kokkos::RangePolicy<>(g.v_j_begin(), g.v_j_end()),
            KOKKOS_LAMBDA(const int j) {
                v(g.v_i_begin() - 1, j) = v(g.v_i_end() - 1, j);
                v(g.v_i_end(),       j) = v(g.v_i_begin(),   j);
            });

        // Same duplicate-face issue as u in x: v_j_end()-1 == v_j_begin() after sync.
        Kokkos::parallel_for("periodic_v_y",
            Kokkos::RangePolicy<>(g.v_i_begin(), g.v_i_end()),
            KOKKOS_LAMBDA(const int i) {
                v(i, g.v_j_begin() - 1) = v(i, g.v_j_end() - 2);
                v(i, g.v_j_end())       = v(i, g.v_j_begin() + 1);
            });

        Kokkos::parallel_for("periodic_v_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                v(g.v_i_begin() - 1, g.v_j_begin() - 1) = v(g.v_i_end() - 1, g.v_j_end() - 2);
                v(g.v_i_begin() - 1, g.v_j_end())       = v(g.v_i_end() - 1, g.v_j_begin() + 1);
                v(g.v_i_end(),       g.v_j_begin() - 1) = v(g.v_i_begin(),   g.v_j_end() - 2);
                v(g.v_i_end(),       g.v_j_end())       = v(g.v_i_begin(),   g.v_j_begin() + 1);
            });
    }

    void apply(SimState& s) const {
        const MacGrid2D& g = s.grid;

        // pressure
        Kokkos::parallel_for("periodic_p_x",
            Kokkos::RangePolicy<>(g.p_j_begin(), g.p_j_end()),
            KOKKOS_LAMBDA(const int j) {
                s.p(g.p_i_begin() - 1, j) = s.p(g.p_i_end() - 1, j);
                s.p(g.p_i_end(),       j) = s.p(g.p_i_begin(),   j);
            });

        Kokkos::parallel_for("periodic_p_y",
            Kokkos::RangePolicy<>(g.p_i_begin(), g.p_i_end()),
            KOKKOS_LAMBDA(const int i) {
                s.p(i, g.p_j_begin() - 1) = s.p(i, g.p_j_end() - 1);
                s.p(i, g.p_j_end())       = s.p(i, g.p_j_begin());
            });

        Kokkos::parallel_for("periodic_p_corners",
            Kokkos::RangePolicy<>(0, 1),
            KOKKOS_LAMBDA(const int) {
                s.p(g.p_i_begin() - 1, g.p_j_begin() - 1) = s.p(g.p_i_end() - 1, g.p_j_end() - 1);
                s.p(g.p_i_begin() - 1, g.p_j_end())       = s.p(g.p_i_end() - 1, g.p_j_begin());
                s.p(g.p_i_end(),       g.p_j_begin() - 1) = s.p(g.p_i_begin(),   g.p_j_end() - 1);
                s.p(g.p_i_end(),       g.p_j_end())       = s.p(g.p_i_begin(),   g.p_j_begin());
            });

        apply_u(g, s.u);
        apply_v(g, s.v);
    }
};

// -------------------------------------------------------------------------
// InflowOutflowBC
//   Left  (x=0,  inflow):   u = u_inf  (Dirichlet),  v zero-gradient
//   Right (x=Lx, outflow):  u, v zero-gradient (∂/∂x = 0)
//   Top/bottom (symmetry):  ∂u/∂y = 0,  v = 0,  ∂p/∂y = 0
//
// Pressure: Neumann left/top/bottom, Dirichlet (p=0) right. The Dirichlet
// outlet pins the constant, so PressureSolver drops its mean-zero gauge.
// -------------------------------------------------------------------------
struct InflowOutflowBC {
    double u_inf = 1.0;

    InflowOutflowBC() = default;
    InflowOutflowBC(double u_in) : u_inf(u_in) {}

    static PressureBCSides pressure_sides() {
        return { PressureBC::Neumann,    // left   (inflow)
                 PressureBC::Dirichlet,  // right  (outflow)
                 PressureBC::Neumann,    // bottom (symmetry)
                 PressureBC::Neumann };  // top    (symmetry)
    }

    // Inflow face (i=u_i_begin) is Dirichlet u=u_inf -- exclude.
    // Outflow face (i=u_i_end-1) is set by zero-gradient extrapolation in
    // apply_u after each sweep -- exclude. Top/bottom u-faces are
    // half-cell-offset symmetry (Neumann), so they ARE unknowns.
    static UVRange u_solve_range(const MacGrid2D& g) {
        return { g.u_i_begin() + 1, g.u_i_end() - 1,
                 g.u_j_begin(),     g.u_j_end()     };
    }
    // Top/bottom v-faces sit on the symmetry walls with v=0 (Dirichlet).
    static UVRange v_solve_range(const MacGrid2D& g) {
        return { g.v_i_begin(),     g.v_i_end(),
                 g.v_j_begin() + 1, g.v_j_end() - 1 };
    }

    void apply_u(const MacGrid2D& g, Kokkos::View<double**> u) const {
        const double u_in = u_inf;

        // Inflow face (sits exactly on x=0): Dirichlet u = u_inf.
        // Outflow face: zero-gradient extrapolation. The viscous solver
        // excludes u_i_end()-1 from its sweep (legacy periodic-DOF count),
        // so we must set the outflow face value here.
        Kokkos::parallel_for("io_u_lr",
            Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
            KOKKOS_LAMBDA(int j) {
                u(g.u_i_begin(),     j) = u_in;
                u(g.u_i_end() - 1,   j) = u(g.u_i_end() - 2, j);
            });

        // Streamwise ghost cells (one cell outside the boundary u-face)
        Kokkos::parallel_for("io_u_lr_ghost",
            Kokkos::RangePolicy<>(g.u_j_begin(), g.u_j_end()),
            KOKKOS_LAMBDA(int j) {
                u(g.u_i_begin() - 1, j) = u_in;
                u(g.u_i_end(),       j) = u(g.u_i_end() - 1, j);
            });

        // Top/bottom symmetry: u-face is half a cell from the wall, so
        // ghost = interior gives ∂u/∂y = 0 at the wall.
        Kokkos::parallel_for("io_u_tb",
            Kokkos::RangePolicy<>(g.u_i_begin() - 1, g.u_i_end() + 1),
            KOKKOS_LAMBDA(int i) {
                u(i, g.u_j_begin() - 1) = u(i, g.u_j_begin());
                u(i, g.u_j_end())       = u(i, g.u_j_end() - 1);
            });
    }

    void apply_v(const MacGrid2D& g, Kokkos::View<double**> v) const {
        // Top/bottom walls: v-face sits exactly on the wall, set v = 0.
        Kokkos::parallel_for("io_v_tb",
            Kokkos::RangePolicy<>(g.v_i_begin(), g.v_i_end()),
            KOKKOS_LAMBDA(int i) {
                v(i, g.v_j_begin())     = 0.0;
                v(i, g.v_j_end() - 1)   = 0.0;
            });

        // Left/right: v-face is half a cell from the wall; use ghost = interior
        // so that ∂v/∂x = 0 (zero-gradient) on both inflow and outflow.
        Kokkos::parallel_for("io_v_lr",
            Kokkos::RangePolicy<>(g.v_j_begin() - 1, g.v_j_end() + 1),
            KOKKOS_LAMBDA(int j) {
                v(g.v_i_begin() - 1, j) = v(g.v_i_begin(),   j);
                v(g.v_i_end(),       j) = v(g.v_i_end() - 1, j);
            });
    }

    void apply_p(SimState& s) const {
        const MacGrid2D& g = s.grid;

        // Left: Neumann, ghost = interior
        // Right: homogeneous Dirichlet at the half-cell-offset face,
        //        ghost = -interior so the average across the face is 0.
        Kokkos::parallel_for("io_p_lr",
            Kokkos::RangePolicy<>(g.p_j_begin(), g.p_j_end()),
            KOKKOS_LAMBDA(int j) {
                s.p(g.p_i_begin() - 1, j) =  s.p(g.p_i_begin(),   j);
                s.p(g.p_i_end(),       j) = -s.p(g.p_i_end() - 1, j);
            });

        // Top/bottom symmetry: Neumann, ghost = interior
        Kokkos::parallel_for("io_p_tb",
            Kokkos::RangePolicy<>(g.p_i_begin(), g.p_i_end()),
            KOKKOS_LAMBDA(int i) {
                s.p(i, g.p_j_begin() - 1) = s.p(i, g.p_j_begin());
                s.p(i, g.p_j_end())       = s.p(i, g.p_j_end() - 1);
            });
    }

    void apply(SimState& s) const {
        apply_p(s);
        apply_u(s.grid, s.u);
        apply_v(s.grid, s.v);
    }
};
