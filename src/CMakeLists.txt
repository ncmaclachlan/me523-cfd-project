# All solver code is templated and header-only.
# ns2d is an INTERFACE library: it propagates includes and link deps
# to any target that links against it, with no compiled sources of its own.
add_library(ns2d INTERFACE)

target_include_directories(ns2d
    INTERFACE
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}
)

if(MPI_FOUND)
    target_link_libraries(ns2d INTERFACE MPI::MPI_CXX)
    target_compile_definitions(ns2d INTERFACE USE_MPI)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(ns2d INTERFACE OpenMP::OpenMP_CXX)
    target_compile_definitions(ns2d INTERFACE USE_OPENMP)
endif()

target_link_libraries(ns2d INTERFACE Kokkos::kokkos)

target_include_directories(ns2d INTERFACE ${Trilinos_INCLUDE_DIRS})
target_link_libraries(ns2d INTERFACE ${Trilinos_LIBRARIES})
