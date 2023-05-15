find_package(GTest REQUIRED)

add_executable(test_yaml_reader test_yaml_reader.cpp)
target_link_libraries(test_yaml_reader 
                    yaml_reader
                    GTest::GTest 
                    GTest::Main)