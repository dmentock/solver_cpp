project(grid)
cmake_minimum_required(VERSION 3.10)

include_directories(${CMAKE_Fortran_MODULE_DIRECTORY})

get_target_property(FORTRAN_LIBRARY_DIR fortran_library BINARY_DIR)

add_library(base64 base64.f90)
target_include_directories(base64 PUBLIC ${FORTRAN_LIBRARY_DIR})
target_link_libraries(base64 PUBLIC fortran_library)

add_library(zlib zlib.f90)
target_include_directories(zlib PUBLIC ${FORTRAN_LIBRARY_DIR})

message("fortran_library is built in ${FORTRAN_LIBRARY_DIR}")

get_target_property(ZLIB_INCLUDE_DIRS zlib INCLUDE_DIRECTORIES)
message("Include directories for zlib: ${ZLIB_INCLUDE_DIRS}")

target_link_libraries(zlib PUBLIC fortran_library -lz)

add_library(VTI VTI.f90)
target_link_libraries(VTI 
                      PUBLIC
                      base64
                      zlib
                      -lubsan)

add_library(discretization_grid discretization_grid.cpp)
target_link_libraries(discretization_grid 
                      PUBLIC
                      VTI
                      -lmpi
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      -lfftw3)

add_library(fft fft.cpp)
target_link_libraries(fft 
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      -lmpi
                      -lfftw3)

add_library(spectral spectral.cpp)
target_link_libraries(spectral
                      PUBLIC
                      fft
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so)

add_library(config_c config.cpp)
target_link_libraries(config_c -lyaml)

add_library(mech_base mech_base.cpp)
target_link_libraries(mech_base
                      PUBLIC
                      spectral
                      discretization_grid
                      config_c)

add_library(mech_solver_basic mech_solver_basic.cpp)
target_link_libraries(mech_solver_basic 
                      PUBLIC
                      mech_base)

add_library(damask_grid_lib damask_grid.cpp)
target_link_libraries(damask_grid_lib 
                      PUBLIC
                      fft
                      spectral  
                      discretization_grid
                      config_c
                      mech_base
                      mech_solver_basic
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      OpenMP::OpenMP_CXX)

add_subdirectory(test)
