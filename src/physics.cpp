#include "physics.hpp"

namespace physics {


void compute_u_rhs(const SimState& s, double re,
                   Kokkos::View<double**> rhs_u) {
    // convective + diffusive terms for u-momentum
    // rhs_u(i,j) = -d(uu)/dx - d(uv)/dy + (1/Re)(d2u/dx2 + d2u/dy2)
    const double inv_dx = 1.0 / s.grid.dx;
    const double inv_dy = 1.0 / s.grid.dy;
    const double inv_dx2 = 1.0 / (s.grid.dx*s.grid.dx);
    const double inv_dy2 = 1.0 / (s.grid.dy*s.grid.dy);
    const double inv_re  = 1.0 / re;
    
    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    Kokkos::parallel_for("compute_u_rhs",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {s.grid.u_i_begin(), s.grid.u_j_begin()},
        {s.grid.u_i_end(), s.grid.u_j_end()}),
    KOKKOS_LAMBDA(int i, int j) {
        const double u_east = avg_x(u, i, j);
        const double u_west = avg_x(u, i-1, j);
        const double duudx = (u_east*u_east - u_west*u_west) * inv_dx;

        const double u_north = avg_y(u, i, j);
        const double u_south = avg_y(u, i, j-1);
        const double v_north = avg_x(v, i-1, j+1);
        const double v_south = avg_x(v, i-1, j);
        const double duvdy = (u_north*v_north - u_south * v_south) *inv_dy;

        const double viscous_term = inv_re * laplacian(u, i, j, inv_dx2, inv_dy2);

        rhs_u(i,j) = - duudx - duvdy + viscous_term;
    });
}

void compute_v_rhs(const SimState& s, double re,
                   Kokkos::View<double**> rhs_v) {
    // convective + diffusive terms for v-momentum
    // rhs_v(i,j) = -d(uv)/dx - d(vv)/dy + (1/Re)(d2v/dx2 + d2v/dy2)
    const double inv_dx  = 1.0 / s.grid.dx;
    const double inv_dy  = 1.0 / s.grid.dy;
    const double inv_dx2 = 1.0 / (s.grid.dx*s.grid.dx);
    const double inv_dy2 = 1.0 / (s.grid.dy*s.grid.dy);
    const double inv_re  = 1.0 / re;

    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    Kokkos::parallel_for("compute_v_rhs",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {s.grid.v_i_begin(), s.grid.v_j_begin()},
        {s.grid.v_i_end(),   s.grid.v_j_end()}),
    KOKKOS_LAMBDA(int i, int j) {
        const double v_north = avg_y(v, i, j);
        const double v_south = avg_y(v, i, j-1);
        const double dvvdy = (v_north*v_north - v_south*v_south) * inv_dy;

        const double v_east = avg_x(v, i,   j);
        const double v_west = avg_x(v, i-1, j);
        const double u_east = avg_y(u, i+1, j-1);
        const double u_west = avg_y(u, i,   j-1);
        const double duvdx = (u_east*v_east - u_west*v_west) * inv_dx;

        const double viscous_term = inv_re * laplacian(v, i, j, inv_dx2, inv_dy2);

        rhs_v(i,j) = -duvdx - dvvdy + viscous_term;
    });
}

void compute_u_conv_rhs(const SimState& s,
                        Kokkos::View<double**> rhs_u) {
    const double inv_dx = 1.0 / s.grid.dx;
    const double inv_dy = 1.0 / s.grid.dy;

    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    Kokkos::parallel_for("compute_u_conv_rhs",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {s.grid.u_i_begin(), s.grid.u_j_begin()},
        {s.grid.u_i_end(),   s.grid.u_j_end()}),
    KOKKOS_LAMBDA(int i, int j) {
        const double u_east  = avg_x(u, i,   j);
        const double u_west  = avg_x(u, i-1, j);
        const double duudx   = (u_east*u_east - u_west*u_west) * inv_dx;

        const double u_north = avg_y(u, i,   j);
        const double u_south = avg_y(u, i,   j-1);
        const double v_north = avg_x(v, i-1, j+1);
        const double v_south = avg_x(v, i-1, j);
        const double duvdy   = (u_north*v_north - u_south*v_south) * inv_dy;

        rhs_u(i, j) = -duudx - duvdy;
    });
}

void compute_u_diff_rhs(const SimState& s, double re,
                        Kokkos::View<double**> rhs_u) {
    const double inv_dx2 = 1.0 / (s.grid.dx * s.grid.dx);
    const double inv_dy2 = 1.0 / (s.grid.dy * s.grid.dy);
    const double inv_re  = 1.0 / re;

    SimState::View2D u = s.u;

    Kokkos::parallel_for("compute_u_diff_rhs",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {s.grid.u_i_begin(), s.grid.u_j_begin()},
        {s.grid.u_i_end(),   s.grid.u_j_end()}),
    KOKKOS_LAMBDA(int i, int j) {
        rhs_u(i, j) = inv_re * laplacian(u, i, j, inv_dx2, inv_dy2);
    });
}

void compute_v_conv_rhs(const SimState& s,
                        Kokkos::View<double**> rhs_v) {
    const double inv_dx = 1.0 / s.grid.dx;
    const double inv_dy = 1.0 / s.grid.dy;

    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    Kokkos::parallel_for("compute_v_conv_rhs",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {s.grid.v_i_begin(), s.grid.v_j_begin()},
        {s.grid.v_i_end(),   s.grid.v_j_end()}),
    KOKKOS_LAMBDA(int i, int j) {
        const double v_north = avg_y(v, i,   j);
        const double v_south = avg_y(v, i,   j-1);
        const double dvvdy   = (v_north*v_north - v_south*v_south) * inv_dy;

        const double v_east  = avg_x(v, i,   j);
        const double v_west  = avg_x(v, i-1, j);
        const double u_east  = avg_y(u, i+1, j-1);
        const double u_west  = avg_y(u, i,   j-1);
        const double duvdx   = (u_east*v_east - u_west*v_west) * inv_dx;

        rhs_v(i, j) = -duvdx - dvvdy;
    });
}

void compute_v_diff_rhs(const SimState& s, double re,
                        Kokkos::View<double**> rhs_v) {
    const double inv_dx2 = 1.0 / (s.grid.dx * s.grid.dx);
    const double inv_dy2 = 1.0 / (s.grid.dy * s.grid.dy);
    const double inv_re  = 1.0 / re;

    SimState::View2D v = s.v;

    Kokkos::parallel_for("compute_v_diff_rhs",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {s.grid.v_i_begin(), s.grid.v_j_begin()},
        {s.grid.v_i_end(),   s.grid.v_j_end()}),
    KOKKOS_LAMBDA(int i, int j) {
        rhs_v(i, j) = inv_re * laplacian(v, i, j, inv_dx2, inv_dy2);
    });
}

double compute_kinetic_energy(const SimState& s){
    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    double sum_u2 = 0.0;
    Kokkos::parallel_reduce("KE_u", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.u_i_begin(), s.grid.u_j_begin()},
            {s.grid.u_i_end(), s.grid.u_j_end()}),
        KOKKOS_LAMBDA(int i, int j, double& lsum){
            lsum += u(i, j) * u(i, j);
        }, sum_u2);

    
    double sum_v2 = 0.0;
    Kokkos::parallel_reduce("KE_v", 
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.v_i_begin(), s.grid.v_j_begin()},
            {s.grid.v_i_end(), s.grid.v_j_end()}),
        KOKKOS_LAMBDA(int i, int j, double& lsum){
            lsum += v(i, j) * v(i, j);
        }, sum_v2);
    
    return 0.5 * (sum_u2 + sum_v2) * s.grid.dx * s.grid.dy;
}

double compute_cfl_dt(const SimState& s, double cfl) {
    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    double u_max = 0.0;
    Kokkos::parallel_reduce("cfl_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.u_i_begin(), s.grid.u_j_begin()},
            {s.grid.u_i_end(),   s.grid.u_j_end()}),
        KOKKOS_LAMBDA(int i, int j, double& lmax) {
            const double val = Kokkos::fabs(u(i, j));
            if (val > lmax) lmax = val;
        },
        Kokkos::Max<double>(u_max));

    double v_max = 0.0;
    Kokkos::parallel_reduce("cfl_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.v_i_begin(), s.grid.v_j_begin()},
            {s.grid.v_i_end(),   s.grid.v_j_end()}),
        KOKKOS_LAMBDA(int i, int j, double& lmax) {
            const double val = Kokkos::fabs(v(i, j));
            if (val > lmax) lmax = val;
        },
        Kokkos::Max<double>(v_max));

    const double vel_max = Kokkos::max(u_max, v_max);
    const double h       = Kokkos::min(s.grid.dx, s.grid.dy);

    // Guard against zero velocity (t=0 or fully damped flow)
    if (vel_max < 1e-14) return cfl * h;
    return cfl * h / vel_max;
}

double compute_l2_divergence(const SimState& s) {
    const double inv_dx = 1.0 / s.grid.dx;
    const double inv_dy = 1.0 / s.grid.dy;

    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    double sum_div2 = 0.0;
    Kokkos::parallel_reduce("L2_div",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.p_i_begin(), s.grid.p_j_begin()},
            {s.grid.p_i_end(),   s.grid.p_j_end()}),
        KOKKOS_LAMBDA(int i, int j, double& lsum) {
            double d = divergence(u, v, i, j, inv_dx, inv_dy);
            lsum += d * d;
        }, sum_div2);

    // RMS divergence: divide by cell count before taking root so the metric is
    // grid-size-independent and meaningful for mesh refinement studies.
    return Kokkos::sqrt(sum_div2 / (s.grid.nx * s.grid.ny));
}

void compute_pressure_rhs(const SimState& s,
                           Kokkos::View<double**> u_star,
                           Kokkos::View<double**> v_star,
                           double dt,
                           Kokkos::View<double**> rhs) {
    // divergence of predicted velocity / dt
    //   rhs(i,j) = (1/dt)( du_star/dx + dv_star/dy )
    const double inv_dx = 1.0 / s.grid.dx;
    const double inv_dy = 1.0 / s.grid.dy;
    const double inv_dt = 1.0 / dt;

    Kokkos::parallel_for("compute_pressure_rhs",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
        {s.grid.p_i_begin(), s.grid.p_j_begin()},
        {s.grid.p_i_end(), s.grid.p_j_end()}),
    KOKKOS_LAMBDA(int i, int j) {
        const double dudx_star = (u_star(i+1,j) - u_star(i,j)) * inv_dx;
        const double dvdy_star = (v_star(i,j+1) - v_star(i,j)) * inv_dy;

        rhs(i,j) = (dudx_star + dvdy_star) * inv_dt;
    }
    );
}

void correct_velocity(SimState& s,
                      Kokkos::View<double**> u_star,
                      Kokkos::View<double**> v_star,
                      double dt) {
    // subtract pressure gradient from predicted velocity
    //   u = u_star - dt * dp/dx
    //   v = v_star - dt * dp/dy

    const double inv_dx = 1.0 / s.grid.dx;
    const double inv_dy = 1.0 / s.grid.dy;

    SimState::View2D p = s.p;
    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    Kokkos::parallel_for(
        "correct_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.u_i_begin(), s.grid.u_j_begin()},
            {s.grid.u_i_end(),   s.grid.u_j_end()}),
        KOKKOS_LAMBDA(int i, int j) {
            const double dpdx = (p(i, j) - p(i - 1, j)) * inv_dx;
            u(i, j) = u_star(i, j) - dt * dpdx;
        });

    Kokkos::parallel_for(
        "correct_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.v_i_begin(), s.grid.v_j_begin()},
            {s.grid.v_i_end(),   s.grid.v_j_end()}),
        KOKKOS_LAMBDA(int i, int j) {
            const double dpdy = (p(i, j) - p(i, j - 1)) * inv_dy;
            v(i, j) = v_star(i, j) - dt * dpdy;
        });
}

ErrorNorms compute_error_norms(const SimState& s, double re) {
    const double nu    = 1.0 / re;
    const double decay = Kokkos::exp(-2.0 * nu * s.time);
    const double dx    = s.grid.dx;
    const double dy    = s.grid.dy;
    const int    ng    = MacGrid2D::ng;

    SimState::View2D u = s.u;
    SimState::View2D v = s.v;

    // u-velocity error
    double u_sum2 = 0.0, u_max = 0.0;
    int u_count = (s.grid.u_i_end() - s.grid.u_i_begin())
                * (s.grid.u_j_end() - s.grid.u_j_begin());

    Kokkos::parallel_reduce("err_u",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.u_i_begin(), s.grid.u_j_begin()},
            {s.grid.u_i_end(),   s.grid.u_j_end()}),
        KOKKOS_LAMBDA(int i, int j, double& lsum2, double& lmax) {
            double x = (i - ng) * dx;
            double y = (j - ng + 0.5) * dy;
            double u_exact = Kokkos::sin(x) * Kokkos::cos(y) * decay;
            double err = u(i, j) - u_exact;
            lsum2 += err * err;
            if (Kokkos::abs(err) > lmax) lmax = Kokkos::abs(err);
        },
        Kokkos::Sum<double>(u_sum2),
        Kokkos::Max<double>(u_max));

    // v-velocity error
    double v_sum2 = 0.0, v_max = 0.0;
    int v_count = (s.grid.v_i_end() - s.grid.v_i_begin())
                * (s.grid.v_j_end() - s.grid.v_j_begin());

    Kokkos::parallel_reduce("err_v",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {s.grid.v_i_begin(), s.grid.v_j_begin()},
            {s.grid.v_i_end(),   s.grid.v_j_end()}),
        KOKKOS_LAMBDA(int i, int j, double& lsum2, double& lmax) {
            double x = (i - ng + 0.5) * dx;
            double y = (j - ng) * dy;
            double v_exact = -Kokkos::cos(x) * Kokkos::sin(y) * decay;
            double err = v(i, j) - v_exact;
            lsum2 += err * err;
            if (Kokkos::abs(err) > lmax) lmax = Kokkos::abs(err);
        },
        Kokkos::Sum<double>(v_sum2),
        Kokkos::Max<double>(v_max));

    double l2   = Kokkos::sqrt((u_sum2 + v_sum2) / (u_count + v_count));
    double linf = Kokkos::max(u_max, v_max);

    return {l2, linf};
}

} // namespace physics
