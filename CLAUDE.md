# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2D incompressible Navier-Stokes solver for ME 523 (Advanced Heat Transfer). Built with C++17, Kokkos for performance portability (CPU/GPU), and optional Trilinos for linear algebra. Uses the Chorin projection method on a staggered (MAC) grid with a built-in geometric multigrid pressure solver. This is an active course project.

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

# Run solver
build/apps/ns_solver --nx 64 --ny 64 --re 100 --cfl 0.5 --t_end 10

# Run tests
cd build && ctest

# Build docs locally
doxygen docs/Doxyfile
```

Spack must be sourced (`source ~/spack/share/spack/setup-env.sh`) before running any script. The build script auto-detects the installed spack environment from `spack.lock` files.

Trilinos is optional and off by default. Enable with `-DUSE_TRILINOS=ON` at CMake configure time (the build script does not pass this; add it manually if needed). The built-in multigrid solver (`include/multigrid.hpp`) handles the pressure Poisson equation without Trilinos.

## Architecture

Hybrid library: templated/Kokkos code is header-only in `include/`, while non-templated host-side components (`PressureSolver`, `CSVOutput`, `physics.cpp`) are compiled from `src/`. The `ns2d` CMake target is a compiled `STATIC` library that links Kokkos and optional MPI/OpenMP/Trilinos.

### Key data structures

- `RunConfig` (`include/run_config.hpp`) — simulation parameters: grid size, domain (default 2π×2π), dt, CFL, t_end, Re, output settings. Supports adaptive timestepping via `cfl > 0` or fixed dt when `cfl <= 0`. Generates run directory paths automatically: `data/run_{nx}_{ny}_{Re}_{cfl}`
- `MacGrid2D` (`include/grid.hpp`) — staggered grid with cell-centered pressure and face-centered velocities. Constructible from `RunConfig`
- `SimState` (`include/sim_state.hpp`) — all runtime data as `Kokkos::View<double**>` 2D views: `u`, `v`, `p`, `u_star`, `v_star`, `rhs`

### Solver pipeline (Chorin projection method)

`Solver` (`include/solver.hpp`) is templated on `<BC, Integrator>` with defaults `<LidDrivenCavityBC, ForwardEuler>`. The current `ns_solver.cpp` uses `<PeriodicBC, CrankThatNicolson>`. Each time step in `Solver::advance()` runs:

1. **BC application** — `bc.apply(state)` sets boundary conditions (wall/lid or periodic)
2. **Velocity prediction** — `integrator.predict(state, bc, u_star, v_star, re, dt)` computes predicted velocities from momentum equations
3. **Pressure Poisson RHS** — `physics::compute_pressure_rhs(state, dt, rhs)` computes divergence of predicted velocity
4. **Pressure solve** — `pressure.solve(state)` solves Laplacian(p) = rhs (multigrid or Trilinos)
5. **Velocity correction** — `physics::correct_velocity(state, dt)` subtracts pressure gradient from predicted velocity

`Solver::run()` drives the time loop, tracks `RunStats`, and writes output at intervals via `CSVOutput`.

### Header inventory

| Header | Purpose |
|---|---|
| `run_config.hpp` | Parameter struct with adaptive CFL or fixed dt timestepping |
| `grid.hpp` | `MacGrid2D` — staggered grid geometry and index helpers |
| `sim_state.hpp` | `SimState` — Kokkos 2D views for all field variables |
| `boundary_conditions.hpp` | `LidDrivenCavityBC`, `PeriodicBC` — wall/lid/periodic BCs |
| `initial_conditions.hpp` | `ZeroIC`, `TaylorGreenIC` — field initialization |
| `physics.hpp` | `physics::` namespace — momentum RHS, pressure RHS, velocity correction |
| `integrator.hpp` | `ForwardEuler`, `RK2`, `CrankThatNicolson` — time integration schemes |
| `viscous_solver.hpp` | `ViscousSolver` — Red-Black Gauss-Seidel for implicit viscous terms (used by Crank-Nicolson) |
| `multigrid.hpp` | `mg::` namespace — geometric multigrid helpers for pressure solve |
| `pressure_solver.hpp` | `PressureSolver` — pressure Poisson solve (multigrid built-in, Trilinos optional) |
| `output.hpp` | `CSVOutput` — write fields to CSV |
| `solver.hpp` | `Solver<BC, Integrator>` — orchestrates the projection method, tracks `RunStats` |

### Extension points

- **New BC type**: define a struct with `void apply(SimState&)`, pass as first template arg to `Solver`
- **New integrator**: define a struct with `void predict(SimState&, BC&, Views..., double re, double dt)`, pass as second template arg
- **Physics functions** are free functions in `physics::` namespace; if a second set of equations is ever needed, promote to a template parameter

### Kokkos usage

- All compute kernels must use `Kokkos::parallel_for` with `Kokkos::MDRangePolicy<Kokkos::Rank<2>>` and `KOKKOS_LAMBDA`
- Views use default execution/memory space (set at Kokkos configure time via spack)
- Host mirrors required for I/O: `Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, view)`
- `Kokkos::fence()` before timing or Trilinos calls

## Tests

Six tests exist in `tests/`: OutputTest, PressureSolverTest, KineticEnergyTest, L2DivergenceTest, ViscousSolverTest, IntegratorsTest. All link against `ns2d`. Run after building:
```bash
cd build && ctest
```

## Post-processing

The `plotting/` directory is a Python package for visualizing solver output. Run with `python -m plotting` from the project root. Key modules: `loader.py` (read CSV output), `fields.py` (field plots), `diagnostics.py` (convergence/error analysis), `style.py` (matplotlib config).

## Compile Definitions

- `USE_MPI` / `USE_OPENMP` — conditionally defined based on whether MPI/OpenMP are found at configure time. Check with `#ifdef`
- `HAS_TRILINOS` — defined when `-DUSE_TRILINOS=ON` is passed to CMake

## CI/CD

GitHub Actions workflow (`.github/workflows/docs.yml`) auto-deploys Doxygen docs to GitHub Pages on push to `main`.

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
