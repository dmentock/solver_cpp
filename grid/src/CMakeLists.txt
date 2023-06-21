add_library(discretization_grid discretization_grid.cpp)
target_link_libraries(discretization_grid 
                      PUBLIC
                      f_discretization
                      f_VTI
                      ${MPI_C_LIBRARIES}
                      ${MPI_CXX_LIBRARIES}
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      -lfftw3)

add_library(fft fft.cpp)
target_link_libraries(fft 
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      -lfftw3)

add_library(spectral spectral.cpp)
target_link_libraries(spectral
                      fft
                      f_prec
                      f_math
                      f_rotations
                      f_homogenization
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      ${PETSC_ARCH_DIR}/lib/libmpi.a)

add_library(mech_base mech_base.cpp)
target_link_libraries(mech_base
                      spectral
                      discretization_grid
                      config
                      f_math
                      f_rotations
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      -lfftw3)

add_library(mech_solver_basic mech_solver_basic.cpp)
target_link_libraries(mech_solver_basic 
                      PUBLIC
                      mech_base)
