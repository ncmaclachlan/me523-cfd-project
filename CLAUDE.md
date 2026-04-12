# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2D incompressible Navier-Stokes solver for ME 523 (Advanced Heat Transfer). Built with C++17, Kokkos for performance portability (CPU/GPU), and Trilinos for linear algebra. This is an active course project with skeleton modules that are being filled in.

## Build Commands

Dependencies are managed via Spack with per-machine environment definitions in `spack/<machine>/spack.yaml` (machines: `ncam`, `hmorenom`, `greatlakes`).

```bash
# One-time setup (concretize + install all deps)
./scripts/setup_spack.sh <machine>

# Build (requires spack sourced in shell)
./scripts/build_with_spack.sh           # Release
./scripts/build_with_spack.sh Debug     # Debug

# Binary output
build/apps/ns_solver

# Build docs locally
doxygen docs/Doxyfile
```

Spack must be sourced (`source ~/spack/share/spack/setup-env.sh`) before running any script. The build script auto-detects the installed spack environment from `spack.lock` files.

## Architecture

The solver uses a **module pipeline** pattern. Each stage of the simulation is a `SimModule` subclass that implements `execute(const RunConfig& cfg, SimState& state)`:

1. `GridModule` — generates the computational mesh
2. `InitialConditionsModule` — sets initial u, v, p fields
3. `SolverModule` — time-steps the Navier-Stokes equations (main logic, currently a stub)
4. `OutputModule` — writes results to file

The main executable (`apps/ns_solver.cpp`) instantiates and runs these modules sequentially.

**Key data structures** (in `include/structs.hpp`):
- `RunConfig` — simulation parameters (grid size, domain, dt, Re, etc.)
- `SimState` — all runtime data as `Kokkos::View<double*>` arrays (1D, row-major for nx*ny grids)

**Library structure**: All module sources compile into a static library `ns2d` (defined in `src/CMakeLists.txt`). The executable and tests link against `ns2d`.

## Adding New Modules

1. Create header in `include/` inheriting from `SimModule`
2. Implement in `src/` and add the `.cpp` to `src/CMakeLists.txt`
3. Instantiate and call in `apps/ns_solver.cpp` at the appropriate pipeline stage

## Tests

Test infrastructure is set up in `tests/CMakeLists.txt` but no tests exist yet. To add a test:
```cmake
add_executable(test_grid test_grid.cpp)
target_link_libraries(test_grid PRIVATE ns2d)
add_test(NAME GridTest COMMAND test_grid)
```
Run tests after building: `cd build && ctest`

## CI/CD

GitHub Actions workflow (`.github/workflows/docs.yml`) auto-deploys Doxygen docs to GitHub Pages on push to `main`.

## Compile Definitions

`USE_MPI` and `USE_OPENMP` are conditionally defined based on whether MPI and OpenMP are found at configure time. Check for these with `#ifdef` in source files.
