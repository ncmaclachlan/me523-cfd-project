# me523-cfd-project

2D incompressible Navier-Stokes solver built with Kokkos for performance portability
across CPU and GPU architectures.

**Documentation:** https://ncmaclachlan.github.io/me523-cfd-project/

---

## Directory Structure

```
me523-cfd-project/
├── include/        # Header files
├── src/            # Library source
├── apps/           # Solver executable
├── tests/          # Test suite
├── docs/           # Doxyfile and documentation pages
├── scripts/        # Build and environment setup scripts
├── spack/          # Spack environment definitions per machine
│   ├── ncam/
│   ├── hmorenom/
│   └── greatlakes/
├── analysis/       # Post-processing scripts
├── runs/           # Solver output (gitignored)
├── hw_7_tex/       # Homework writeups
└── final_report_tex/
```

---

## Prerequisites

- [Spack](https://spack.io) installed and sourced in your shell
- GCC 14 (including gfortran) available on your machine
- CUDA toolkit installed if targeting GPU (see `spack/<machine>/spack.yaml`)

Install and source Spack if not already set up:

```bash
git clone --depth=1 https://github.com/spack/spack.git ~/spack
source ~/spack/share/spack/setup-env.sh
```

Register your compilers:

```bash
spack compiler find
```

---

## Setup

Run once per machine to concretize and install all dependencies:

```bash
./scripts/setup_spack.sh <machine>
```

Where `<machine>` is one of: `ncam`, `hmorenom`, `greatlakes`.

See `spack/<machine>/spack.yaml` for machine-specific configuration — comments mark
every value that may need to change (CUDA arch, compiler version, MPI implementation).

---

## Building

```bash
./scripts/build_with_spack.sh           # Release (default)
./scripts/build_with_spack.sh Debug     # Debug
```

The script detects the installed Spack environment automatically and invokes CMake.
The binary is placed at `build/apps/ns_solver`.

---

## Dependencies

| Package | Role |
|---------|------|
| [Kokkos](https://kokkos.org/kokkos-core-wiki/) | Performance portability — CPU/GPU execution |
| [Kokkos Kernels](https://kokkos.org/kokkos-kernels-wiki/) | Portable BLAS and sparse linear algebra |
| [Trilinos](https://trilinos.github.io) | Distributed linear algebra, solvers, preconditioners |
| [OpenMPI](https://www.open-mpi.org) | Distributed-memory communication |
| [SuperLU-DIST](https://portal.nersc.gov/project/sparse/superlu/) | Distributed sparse direct solver |
| [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview) | Parallel graph partitioning |

---

## Documentation

Docs are built with Doxygen and deployed automatically to GitHub Pages on every push
to `main`. To build locally:

```bash
doxygen docs/Doxyfile
```

Output is written to `docs/html/`.
