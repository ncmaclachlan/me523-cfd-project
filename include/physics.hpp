#pragma once
#include <Kokkos_Core.hpp>
#include "sim_state.hpp"

namespace physics {
using SimState = SimState2D;
// --- Pointwise operators (device-inline, used by kernels in physics.cpp) ---

KOKKOS_INLINE_FUNCTION
double laplacian(const Kokkos::View<double**>& f, int i, int j,
                 double inv_dx2, double inv_dy2) {
    return (f(i+1, j) - 2.0*f(i, j) + f(i-1, j)) * inv_dx2
         + (f(i, j+1) - 2.0*f(i, j) + f(i, j-1)) * inv_dy2;
}

KOKKOS_INLINE_FUNCTION
double divergence(const Kokkos::View<double**>& u, 
                 const Kokkos::View<double**>& v,
                 int i, int j,
                 double inv_dx, double inv_dy) {
    return (u(i+1, j) - u(i, j)) * inv_dx
         + (v(i, j+1) - v(i, j)) * inv_dy;
}

KOKKOS_INLINE_FUNCTION
double avg_x(const Kokkos::View<double**>& f, int i, int j) {
    return 0.5 * (f(i, j) + f(i+1, j));
}

KOKKOS_INLINE_FUNCTION
double avg_y(const Kokkos::View<double**>& f, int i, int j) {
    return 0.5 * (f(i, j) + f(i, j+1));
}

// --- Stage kernels (defined in physics.cpp) ---

// Full explicit RHS: convection + diffusion (used by ForwardEuler)
void compute_u_rhs(const SimState& s, double re,
                   Kokkos::View<double**> rhs_u);

void compute_v_rhs(const SimState& s, double re,
                   Kokkos::View<double**> rhs_v);

// Split RHS for Crank-Nicolson: convection and diffusion separately
void compute_u_conv_rhs(const SimState& s,
                        Kokkos::View<double**> rhs_u);

void compute_u_diff_rhs(const SimState& s, double re,
                        Kokkos::View<double**> rhs_u);

void compute_v_conv_rhs(const SimState& s,
                        Kokkos::View<double**> rhs_v);

void compute_v_diff_rhs(const SimState& s, double re,
                        Kokkos::View<double**> rhs_v);

double compute_kinetic_energy(const SimState& s);

// Returns dt = cfl * min(dx, dy) / u_max based on current velocity field.
// u_max is the maximum absolute velocity over all u and v face values.
double compute_cfl_dt(const SimState& s, double cfl);

double compute_l2_divergence(const SimState& s);

struct ErrorNorms { double l2; double linf; };
ErrorNorms compute_error_norms(const SimState& s, double re);

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

namespace physics3D {
// --- Pointwise operators (device-inline, used by kernels in physics.cpp) ---
using SimState = SimState3D;

KOKKOS_INLINE_FUNCTION
double laplacian(const Kokkos::View<double***>& f, int i, int j, int k,
                 double inv_dx2, double inv_dy2, double inv_dz2) {
    return (f(i+1, j, k) - 2.0*f(i, j, k) + f(i-1, j, k)) * inv_dx2
         + (f(i, j+1, k) - 2.0*f(i, j, k) + f(i, j-1, k)) * inv_dy2
         + (f(i, j, k+1)- 2.0*f(i,j,k) + f(i,j,k-1)) * inv_dx2;
}

KOKKOS_INLINE_FUNCTION
double divergence(const Kokkos::View<double***>& u, 
                 const Kokkos::View<double***>& v,
                 const Kokkos::View<double***>& w,
                 int i, int j, int k,
                 double inv_dx, double inv_dy, double inv_dz) {
    return (u(i+1, j, k) - u(i, j, k)) * inv_dx
         + (v(i, j+1, k) - v(i, j, k)) * inv_dy
         + (w(i, j, k+1) - w(i, j, k)) * inv_dz;
}

KOKKOS_INLINE_FUNCTION
double avg_x(const Kokkos::View<double***>& f, int i, int j, int k) {
    return 0.5 * (f(i, j, k) + f(i+1, j, k));
}

KOKKOS_INLINE_FUNCTION
double avg_y(const Kokkos::View<double***>& f, int i, int j, int k) {
    return 0.5 * (f(i, j, k) + f(i, j+1, k));
}

KOKKOS_INLINE_FUNCTION
double avg_z(const Kokkos::View<double***>& f, int i, int j, int k) {
    return 0.5 * (f(i, j, k) + f(i, j, k+1));
}
// --- Stage kernels (defined in physics.cpp) ---

// Full explicit RHS: convection + diffusion (used by ForwardEuler)
void compute_u_rhs(const SimState& s, double re,
                   Kokkos::View<double***> rhs_u);

void compute_v_rhs(const SimState& s, double re,
                   Kokkos::View<double***> rhs_v);

void compute_w_rhs(const SimState& s, double re,
                   Kokkos::View<double***> rhs_w);

// Split RHS for Crank-Nicolson: convection and diffusion separately
void compute_u_conv_rhs(const SimState& s,
                        Kokkos::View<double***> rhs_u);

void compute_u_diff_rhs(const SimState& s, double re,
                        Kokkos::View<double***> rhs_u);

void compute_v_conv_rhs(const SimState& s,
                        Kokkos::View<double***> rhs_v);

void compute_v_diff_rhs(const SimState& s, double re,
                        Kokkos::View<double***> rhs_v);

void compute_w_conv_rhs(const SimState& s,
                        Kokkos::View<double***> rhs_v);

void compute_w_diff_rhs(const SimState& s, double re,
                        Kokkos::View<double***> rhs_w);

double compute_kinetic_energy(const SimState& s);

// Returns dt = cfl * min(dx, dy) / u_max based on current velocity field.
// u_max is the maximum absolute velocity over all u and v face values.
double compute_cfl_dt(const SimState& s, double cfl);

double compute_l2_divergence(const SimState& s);

struct ErrorNorms { double l2; double linf; };
ErrorNorms compute_error_norms(const SimState& s, double re);

void compute_pressure_rhs(const SimState& s,
                           Kokkos::View<double***> u_star,
                           Kokkos::View<double***> v_star,
                           Kokkos::View<double***> w_star,
                           double dt,
                           Kokkos::View<double***> rhs);

void correct_velocity(SimState& s,
                       Kokkos::View<double***> u_star,
                       Kokkos::View<double***> v_star,
                       Kokkos::View<double***> w_star,
                       double dt);
} // namespace physics3D