project(myproject)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 11)
find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/include)

enable_language(Fortran)
add_library(stub_fortran_interface SHARED stub_fortran_interface.f90)
target_compile_options(stub_fortran_interface PUBLIC "-fvisibility=default")

# add_executable(test_cpp_fortran_interface test_cpp_fortran_interface.cpp)
# target_link_libraries(test_cpp_fortran_interface
#                       PUBLIC
#                       stub_fortran_interface
#                       GTest::GTest
#                       GTest::Main
#                       gmock gmock_main)
# add_executable(test_vti_reader test_vti_reader.cpp)
# target_link_libraries(test_vti_reader 
#                       vti_reader 
#                       GTest::GTest 
#                       GTest::Main)
add_executable(test_discretization_grid test_discretization_grid.cpp)
target_link_libraries(test_discretization_grid 
                      discretization_grid 
                      vti_reader 
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)

add_executable(test_spectral_mech_basic test_spectral_mech_basic.cpp)
target_link_libraries(test_spectral_mech_basic 
                      grid_mech_spectral_basic 
                      discretization_grid
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)
add_executable(test_spectral test_spectral.cpp)
target_link_libraries(test_spectral 
                    ${PETSC_ARCH_DIR}/lib/libpetsc.so
                    grid_spectral
                    discretization_grid
                    GTest::GTest 
                    GTest::Main 
                    gmock gmock_main)
