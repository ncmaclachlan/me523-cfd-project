#pragma once
#include <Kokkos_Core.hpp>
#include "sim_state.hpp"

namespace physics {

inline void compute_u_rhs(const SimState& s, double re,
                          Kokkos::View<double**> rhs_u) {
    // TODO: convective + diffusive terms for u-momentum
    //   rhs_u(i,j) = -d(uu)/dx - d(uv)/dy + (1/Re)(d2u/dx2 + d2u/dy2)
}

inline void compute_v_rhs(const SimState& s, double re,
                          Kokkos::View<double**> rhs_v) {
    // TODO: convective + diffusive terms for v-momentum
    //   rhs_v(i,j) = -d(uv)/dx - d(vv)/dy + (1/Re)(d2v/dx2 + d2v/dy2)
}

inline void compute_pressure_rhs(const SimState& s,
                                 Kokkos::View<double**> u_star,
                                 Kokkos::View<double**> v_star,
                                 double dt,
                                 Kokkos::View<double**> rhs) {
    // TODO: divergence of predicted velocity / dt
    //   rhs(i,j) = (1/dt)( du_star/dx + dv_star/dy )
}

inline void correct_velocity(SimState& s,
                              Kokkos::View<double**> u_star,
                              Kokkos::View<double**> v_star,
                              double dt) {
    // TODO: subtract pressure gradient from predicted velocity
    //   u = u_star - dt * dp/dx
    //   v = v_star - dt * dp/dy
}

} // namespace physics
