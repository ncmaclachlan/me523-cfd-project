add_executable(ns_solver ns_solver.cpp)
target_link_libraries(ns_solver PRIVATE NavierStokes2D)
target_include_directories(ns_solver PRIVATE ${CMAKE_SOURCE_DIR}/src/include)
