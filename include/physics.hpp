#pragma once
#include <Kokkos_Core.hpp>
#include "sim_state.hpp"

namespace physics {

// --- Pointwise operators (device-inline, used by kernels in physics.cpp) ---

KOKKOS_INLINE_FUNCTION
double laplacian(const Kokkos::View<double**>& f, int i, int j,
                 double inv_dx2, double inv_dy2) {
    return (f(i+1, j) - 2.0*f(i, j) + f(i-1, j)) * inv_dx2
         + (f(i, j+1) - 2.0*f(i, j) + f(i, j-1)) * inv_dy2;
}

// add divergence here

KOKKOS_INLINE_FUNCTION
double avg_x(const Kokkos::View<double**>& f, int i, int j) {
    return 0.5 * (f(i, j) + f(i+1, j));
}

KOKKOS_INLINE_FUNCTION
double avg_y(const Kokkos::View<double**>& f, int i, int j) {
    return 0.5 * (f(i, j) + f(i, j+1));
}

// --- Stage kernels (defined in physics.cpp) ---

void compute_u_rhs(const SimState& s, double re,
                   Kokkos::View<double**> rhs_u);

void compute_v_rhs(const SimState& s, double re,
                   Kokkos::View<double**> rhs_v);

void compute_pressure_rhs(const SimState& s,
                           Kokkos::View<double**> u_star,
                           Kokkos::View<double**> v_star,
                           double dt,
                           Kokkos::View<double**> rhs);

void correct_velocity(SimState& s,
                       Kokkos::View<double**> u_star,
                       Kokkos::View<double**> v_star,
                       double dt);

} // namespace physics
