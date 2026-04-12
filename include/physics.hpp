#pragma once
#include <Kokkos_Core.hpp>

struct IncompressibleNS {
    double re = 100.0;

    explicit IncompressibleNS(double reynolds = 100.0) : re(reynolds) {}

    // TODO: momentum residual
    // TODO: pressure Poisson RHS
};
