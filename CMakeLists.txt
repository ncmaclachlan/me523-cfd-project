cmake_minimum_required(VERSION 3.16)
project(NavierStokes2D VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_DEBUG   "-g -O0")

# Prevent conda/miniconda lib dirs from polluting RPATH — they contain
# an incompatible OpenMPI that shadows the spack-installed one at runtime.
# CMAKE_BUILD_RPATH_USE_LINK_PATH=OFF stops CMake from auto-adding every
# linked library's directory to the build RPATH; we rely on CMAKE_BUILD_RPATH
# (populated below from spack) instead.
# Prevent conda/miniconda lib dirs from polluting RPATH — they contain
# an incompatible OpenMPI that shadows the spack-installed one at runtime.
# We disable automatic RPATH from link paths and instead build an explicit
# RPATH from CMAKE_PREFIX_PATH (set by spack), filtering out conda entries.
set(CMAKE_BUILD_RPATH_USE_LINK_PATH OFF)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF)
list(FILTER CMAKE_PREFIX_PATH EXCLUDE REGEX "miniconda|anaconda|conda")

# Build explicit RPATH from spack prefix paths (skip any conda leftovers)
set(_spack_rpath "")
foreach(_prefix ${CMAKE_PREFIX_PATH})
  if(NOT _prefix MATCHES "miniconda|anaconda|conda" AND IS_DIRECTORY "${_prefix}/lib")
    list(APPEND _spack_rpath "${_prefix}/lib")
  endif()
endforeach()
set(CMAKE_BUILD_RPATH "${_spack_rpath}")
set(CMAKE_INSTALL_RPATH "${_spack_rpath}")

# ---------------------------------------------------------------------------
# Spack integration
# Activate a spack environment before configuring so that find_package calls
# below resolve against spack-installed packages automatically:
#
#   spack env activate spack/ncam          (ncam local machine)
#   spack env activate spack/hmorenom      (hmorenom local machine)
#   spack env activate spack/greatlakes    (UMich Great Lakes HPC)
#
# Spack sets CMAKE_PREFIX_PATH on activation — no manual path config needed.
# Then configure and build as normal:
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j$(nproc)
# ---------------------------------------------------------------------------

# Optional: MPI — resolved from spack env when openmpi is installed
find_package(MPI COMPONENTS CXX)

# Optional: OpenMP — enabled via +openmp in kokkos/trilinos spack variants
find_package(OpenMP COMPONENTS CXX)

# Required: Kokkos — activate a spack environment before configuring
find_package(Kokkos REQUIRED)

# Required: Trilinos — activate a spack environment before configuring
find_package(Trilinos REQUIRED)

# config.h written to build directory, include it via target_include_directories
configure_file(config.h.in ${CMAKE_BINARY_DIR}/config.h)

add_subdirectory(src)
add_subdirectory(apps)
enable_testing()
add_subdirectory(tests)

find_package(Doxygen QUIET)
if(DOXYGEN_FOUND)
  add_subdirectory(docs)
endif()
