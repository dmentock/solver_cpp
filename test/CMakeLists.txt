find_package(GTest REQUIRED)

add_executable(test_config test_config.cpp)
target_link_libraries(test_config
                      config
                      GTest::GTest 
                      GTest::Main)