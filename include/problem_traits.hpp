#pragma once
#include <Kokkos_Core.hpp>
#include "grid.hpp"
#include "physics.hpp"
#include "boundary_conditions.hpp"
#include "integrator.hpp"
#include "initial_conditions.hpp"
#include "output.hpp"

struct HW7 {
    using Grid             = MacGrid2D;
    using Scalar           = double;
    using ExecSpace        = Kokkos::DefaultExecutionSpace;
    using Physics          = IncompressibleNS;
    using BC               = LidDrivenCavityBC;
    using Integrator       = ForwardEuler;
    using InitialCondition = ZeroIC;
    using Output           = CSVOutput;
};
