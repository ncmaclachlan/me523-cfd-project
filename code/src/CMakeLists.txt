add_library(NavierStokes2D
    modules/grid_module.cpp
    modules/initial_conditions_module.cpp
    modules/solver_module.cpp
    modules/output_module.cpp
)

target_include_directories(NavierStokes2D PUBLIC
    ${CMAKE_SOURCE_DIR}/src/include
    ${CMAKE_BINARY_DIR}
)
