add_executable(test_output test_output.cpp)
target_link_libraries(test_output PRIVATE ns2d)
add_test(NAME OutputTest COMMAND test_output)
