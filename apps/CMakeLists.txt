add_executable(ns_solver ns_solver.cpp)
target_link_libraries(ns_solver PRIVATE ns2d)

add_executable(ib_solver ib_solver.cpp)
target_link_libraries(ib_solver PRIVATE ns2d)
