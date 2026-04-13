# ns2d is a compiled library for non-templated components
# (PressureSolver, CSVOutput) plus an INTERFACE layer for
# header-only templated code (Solver, physics, BCs, integrators).
add_library(ns2d
    physics.cpp
    pressure_solver.cpp
    output.cpp
)

target_include_directories(ns2d
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${CMAKE_BINARY_DIR}
)

if(MPI_FOUND)
    target_link_libraries(ns2d PUBLIC MPI::MPI_CXX)
    target_compile_definitions(ns2d PUBLIC USE_MPI)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(ns2d PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(ns2d PUBLIC USE_OPENMP)
endif()

target_link_libraries(ns2d PUBLIC Kokkos::kokkos)

if(USE_TRILINOS)
    target_include_directories(ns2d PUBLIC ${Trilinos_INCLUDE_DIRS})
    target_link_libraries(ns2d PUBLIC ${Trilinos_LIBRARIES})
    target_compile_definitions(ns2d PUBLIC HAS_TRILINOS)
endif()
