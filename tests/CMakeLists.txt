add_executable(test_output test_output.cpp)
target_link_libraries(test_output PRIVATE ns2d)
add_test(NAME OutputTest COMMAND test_output)

add_executable(test_pressure_solver test_pressure_solver.cpp)
target_link_libraries(test_pressure_solver PRIVATE ns2d)
add_test(NAME PressureSolverTest COMMAND test_pressure_solver)
