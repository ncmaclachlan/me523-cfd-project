#pragma once
#include <Kokkos_Core.hpp>
#include "sim_state.hpp"

namespace physics {

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
