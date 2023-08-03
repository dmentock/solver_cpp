find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS}
                    ${CMAKE_SOURCE_DIR}/include)

enable_language(Fortran)
add_library(stub_fortran_interface SHARED stub_fortran_interface.f90)

add_executable(test_cpp_fortran_interface test_cpp_fortran_interface.cpp)
target_link_libraries(test_cpp_fortran_interface
                      PUBLIC
                      stub_fortran_interface
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      GTest::GTest
                      GTest::Main
                      gmock gmock_main)

add_executable(test_discretization_grid test_discretization_grid.cpp)
target_link_libraries(test_discretization_grid 
                      discretization_grid 
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)

add_executable(test_fft test_fft.cpp)
target_link_libraries(test_fft 
                      fft
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      GTest::GTest
                      GTest::Main)                 
                      
add_executable(test_spectral test_spectral.cpp)
target_link_libraries(test_spectral 
                      spectral
                      discretization_grid
                      f_clean_up_fortran
                      GTest::GTest
                      GTest::Main 
                      gmock gmock_main)
                      
add_executable(test_mech_base test_mech_base.cpp)
target_link_libraries(test_mech_base 
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      mech_base
                      spectral
                      discretization_grid
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)

add_executable(test_mech_solver_basic test_mech_solver_basic.cpp)
target_link_libraries(test_mech_solver_basic
                      PUBLIC
                      mech_solver_basic
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)

add_executable(test_damask_grid test_damask_grid.cpp)
target_link_libraries(test_damask_grid
                      PUBLIC
                      damask_grid_lib
                      f_homogenization
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)
