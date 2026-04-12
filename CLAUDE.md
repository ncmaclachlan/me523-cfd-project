# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2D incompressible Navier-Stokes solver for ME 523 (Advanced Heat Transfer). Built with C++17, Kokkos for performance portability (CPU/GPU), and Trilinos for linear algebra. Uses the Chorin projection method on a staggered (MAC) grid. This is an active course project with skeleton stubs being filled in.

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

Hybrid library: templated/Kokkos code is header-only in `include/`, while non-templated host-side components (`PressureSolver`, `CSVOutput`) are compiled from `src/`. The `ns2d` CMake target is a compiled `STATIC` library that links Kokkos, Trilinos, and optional MPI/OpenMP.

### Key data structures

- `RunConfig` (`include/run_config.hpp`) — simulation parameters: grid size, domain, dt, t_end, Re, output settings
- `MacGrid2D` (`include/grid.hpp`) — staggered grid with cell-centered pressure and face-centered velocities. Constructible from `RunConfig`
- `SimState` (`include/sim_state.hpp`) — all runtime data as `Kokkos::View<double**>` 2D views: `u`, `v`, `p`, `u_star`, `v_star`, `rhs`

### Solver pipeline (Chorin projection method)

`Solver` (`include/solver.hpp`) is templated on `<BC, Integrator>` with defaults `<LidDrivenCavityBC, ForwardEuler>`. Each time step in `Solver::advance()` runs:

1. **BC application** — `bc.apply(state)` sets wall/lid boundary conditions
2. **Velocity prediction** — `integrator.predict(state, re, dt)` computes `u_star`, `v_star` from momentum equations (calls `physics::compute_u_rhs`, `physics::compute_v_rhs`)
3. **Pressure Poisson RHS** — `physics::compute_pressure_rhs(state, dt, rhs)` computes divergence of predicted velocity
4. **Pressure solve** — `pressure.solve(state)` solves Laplacian(p) = rhs via Trilinos
5. **Velocity correction** — `physics::correct_velocity(state, dt)` subtracts pressure gradient from predicted velocity

`Solver::run()` drives the time loop and writes output at intervals via `CSVOutput`.

### Header inventory

| Header | Purpose |
|---|---|
| `run_config.hpp` | Parameter struct (nx, ny, lx, ly, dt, t_end, re, output settings) |
| `grid.hpp` | `MacGrid2D` — staggered grid geometry and index helpers |
| `sim_state.hpp` | `SimState` — Kokkos 2D views for all field variables |
| `boundary_conditions.hpp` | `LidDrivenCavityBC`, `PeriodicBC` — wall/lid/periodic BCs |
| `initial_conditions.hpp` | `ZeroIC`, `TaylorGreenIC` — field initialization |
| `physics.hpp` | `physics::` namespace — momentum RHS, pressure RHS, velocity correction |
| `integrator.hpp` | `ForwardEuler`, `RK2` — time integration (predict step only) |
| `pressure_solver.hpp` | `PressureSolver` — Trilinos-based pressure Poisson solve |
| `output.hpp` | `CSVOutput` — write fields to CSV |
| `solver.hpp` | `Solver<BC, Integrator>` — orchestrates the projection method |

### Extension points

- **New BC type**: define a struct with `void apply(SimState&)`, pass as first template arg to `Solver`
- **New integrator**: define a struct with `void predict(SimState&, double re, double dt)`, pass as second template arg
- **Physics functions** are free functions in `physics::` namespace; if a second set of equations is ever needed, promote to a template parameter

### Kokkos usage

- All compute kernels must use `Kokkos::parallel_for` with `Kokkos::MDRangePolicy<Kokkos::Rank<2>>` and `KOKKOS_LAMBDA`
- Views use default execution/memory space (set at Kokkos configure time via spack)
- Host mirrors required for I/O: `Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view)`
- `Kokkos::fence()` before timing or Trilinos calls

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

## AI Use Documentation

All AI-assisted interactions for this project must be documented in `hw_7_tex/prompts.tex`. This is **required by ME 523 course policy**.

### How to document interactions

When Claude is used during development, append a new interaction to `prompts.tex` following this format:

```latex
\subsection*{Interaction N}

\textbf{Prompt:}
\begin{quote}
[Your prompt text here]
\end{quote}

\textbf{Response summary:}
[Brief description of what Claude generated or modified]

\textbf{Date and Model:} [e.g., Claude Opus 4.6, 2026-04-12]
```

When asked to do work on this project, Claude should proactively offer to document the interaction in `prompts.tex` after completing the task.
