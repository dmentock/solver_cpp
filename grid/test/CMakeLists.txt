find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/include)

enable_language(Fortran)
add_library(stub_fortran_interface SHARED stub_fortran_interface.f90)
target_compile_options(stub_fortran_interface PUBLIC "-fvisibility=default")


add_executable(test_cpp_fortran_interface test_cpp_fortran_interface.cpp)
target_link_libraries(test_cpp_fortran_interface
                      PUBLIC
                      stub_fortran_interface
                      GTest::GTest
                      GTest::Main
                      gmock gmock_main)

add_executable(test_discretization_grid test_discretization_grid.cpp)
target_link_libraries(test_discretization_grid 
                      discretization_grid 
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)
                      
add_executable(test_spectral test_spectral.cpp)
target_link_libraries(test_spectral 
                      spectral
                      discretization_grid
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)
                      
add_executable(test_spectral_fields_mech test_spectral_fields_mech.cpp)
target_link_libraries(test_spectral_fields_mech 
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      spectral_fields_mech
                      spectral
                      discretization_grid
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)

add_executable(test_spectral_solver_basic test_spectral_solver_basic.cpp)
target_link_libraries(test_spectral_solver_basic 
                      PUBLIC
                      spectral_solver_basic 
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)
