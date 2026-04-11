# Build Environment and Dependencies {#spack_guide}

## What is Spack?

[Spack](https://spack.io) is a package manager designed for HPC systems. Unlike system
package managers, Spack can install multiple versions of the same library side by side,
manage complex dependency trees, and build packages with specific compiler versions and
compile-time options (called *variants*). This makes it well suited for scientific
software that has strict dependency requirements and needs to run on many different
machines — from a developer laptop to a large cluster.

## Installing Spack

Spack is installed by cloning its repository and sourcing a shell setup script.
It does not require root access.

```bash
git clone --depth=1 https://github.com/spack/spack.git ~/spack
source ~/spack/share/spack/setup-env.sh
```

To make Spack available in every new shell session, add the `source` line to your
`~/.bashrc` or `~/.zshrc`.

Once Spack is installed, register the compilers available on your machine:

```bash
spack compiler find
```

Verify that the expected compilers appear:

```bash
spack compiler list
```

This project requires **gcc@14** including gfortran. If it is not listed, install it
through your system package manager before running `spack compiler find` again.

---

## Setting Up the Project Environment

Each supported machine has its own Spack environment under `spack/<machine>/spack.yaml`.
The environment pins all package versions and variants needed to build the project.

**Step 1 — one time per machine:** concretize and install all dependencies.

```bash
./scripts/setup_spack.sh <machine>
```

Where `<machine>` is one of:

| Name | Description |
|------|-------------|
| `ncam` | ncam local workstation (CUDA arch 120, gcc@14.2.0) |
| `hmorenom` | hmorenom local workstation (see comments in spack.yaml for changes needed) |
| `greatlakes` | UMich Great Lakes HPC cluster (see comments in spack.yaml for changes needed) |

**Step 2 — every build:**

```bash
./scripts/build_with_spack.sh           # Release (default)
./scripts/build_with_spack.sh Debug     # Debug
```

The build script automatically detects which environment has been installed on the
current machine and activates it before invoking CMake.

---

## Dependencies

### Kokkos

[Kokkos](https://kokkos.org/kokkos-core-wiki/) is a C++ performance portability
framework that abstracts over CPU threads (OpenMP), CUDA, HIP, and other backends
through a single programming model. This project uses `Kokkos::View` as the primary
data structure for field arrays, allowing the solver to target both CPU and GPU
execution without changing the numerical code.

- Documentation: [https://kokkos.org/kokkos-core-wiki/](https://kokkos.org/kokkos-core-wiki/)
- GitHub: [https://github.com/kokkos/kokkos](https://github.com/kokkos/kokkos)

Spack variant used in this project: `+openmp +cuda +cuda_lambda +serial cuda_arch=<arch>`

### Trilinos

[Trilinos](https://trilinos.github.io) is a collection of scientific libraries providing
linear algebra, solvers, preconditioners, and mesh support. This project uses the
Tpetra/Belos/MueLu/Ifpack2 stack for distributed linear algebra and iterative solvers,
and Zoltan2 for load balancing. Trilinos is built against Kokkos so all operations
are performance portable.

- Documentation: [https://trilinos.github.io/documentation.html](https://trilinos.github.io/documentation.html)
- GitHub: [https://github.com/trilinos/Trilinos](https://github.com/trilinos/Trilinos)

Spack variants used: `+kokkos +tpetra +belos +muelu +ifpack2 +mpi +openmp +cuda +amesos2 +zoltan2`

### Kokkos Kernels

[Kokkos Kernels](https://github.com/kokkos/kokkos-kernels) provides performance-portable
BLAS and sparse linear algebra routines built on top of Kokkos, including cuBLAS,
cuSPARSE, and cuSolver backends when CUDA is enabled.

- Documentation: [https://kokkos.org/kokkos-kernels-wiki/](https://kokkos.org/kokkos-kernels-wiki/)
- GitHub: [https://github.com/kokkos/kokkos-kernels](https://github.com/kokkos/kokkos-kernels)

### MPI

[OpenMPI](https://www.open-mpi.org) provides the distributed-memory communication layer.
It is built with UCX transport and CUDA support so that GPU buffers can be communicated
directly without staging through host memory.

- Documentation: [https://www.open-mpi.org/doc/](https://www.open-mpi.org/doc/)

### SuperLU-DIST and ParMETIS

[SuperLU-DIST](https://portal.nersc.gov/project/sparse/superlu/) provides distributed
sparse direct solvers. [ParMETIS](http://glaros.dtc.umn.edu/gkhome/metis/parmetis/overview)
provides parallel graph partitioning used by Trilinos and SuperLU-DIST for load balancing.
