find_package(GTest REQUIRED)

set(TEST_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/bin/solver_unittests")

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
set_target_properties(test_cpp_fortran_interface PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)

add_executable(test_discretization_grid test_discretization_grid.cpp)
target_link_libraries(test_discretization_grid 
                      discretization_grid
                      fortran_library
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      ${ZLIB_LIBRARIES}
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)
set_target_properties(test_discretization_grid PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)

add_executable(test_config test_config.cpp)
target_link_libraries(test_config
                      config_c
                      GTest::GTest 
                      GTest::Main)
set_target_properties(test_config PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)

add_executable(test_fft test_fft.cpp)
target_link_libraries(test_fft 
                      fft
                      GTest::GTest
                      GTest::Main)                 
set_target_properties(test_fft PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)       

add_executable(test_spectral test_spectral.cpp)
target_link_libraries(test_spectral
                      spectral
                      fortran_library
                      discretization_grid
                      GTest::GTest
                      GTest::Main 
                      gmock gmock_main)
set_target_properties(test_spectral PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)

add_executable(test_mech_base test_mech_base.cpp)
target_link_libraries(test_mech_base 
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      mech_base
                      fortran_library
                      spectral
                      discretization_grid
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)
set_target_properties(test_mech_base PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)

add_executable(test_mech_solver_basic test_mech_solver_basic.cpp)
target_link_libraries(test_mech_solver_basic
                      PUBLIC
                      mech_solver_basic
                      fortran_library
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      GTest::GTest 
                      GTest::Main 
                      gmock gmock_main)
set_target_properties(test_mech_solver_basic PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)

add_executable(test_damask_grid test_damask_grid.cpp)
target_link_libraries(test_damask_grid
                      PUBLIC
                      fortran_library
                      damask_grid_lib
                      GTest::GTest
                      GTest::Main 
                      gmock gmock_main)
set_target_properties(test_damask_grid PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${TEST_OUTPUT_DIR}
)
