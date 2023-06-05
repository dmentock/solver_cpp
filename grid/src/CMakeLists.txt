add_library(discretization_grid discretization_grid.cpp)
target_link_libraries(discretization_grid 
                      PUBLIC
                      f_discretization
                      # CLI
                      f_VTI
                      ${MPI_C_LIBRARIES}
                      ${MPI_CXX_LIBRARIES}
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      -lfftw3)
                      
add_library(spectral spectral/spectral.cpp)
target_link_libraries(spectral 
                      f_prec
                      f_math
                      f_rotations
                      f_homogenization
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      -lfftw3)

add_library(spectral_mech_utilities spectral/mech/utilities.cpp)
target_link_libraries(spectral_mech_utilities
                      spectral
                      discretization_grid
                      config
                      f_math
                      f_rotations
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      -lfftw3)

add_library(spectral_mech_basic spectral/mech/basic.cpp)
target_link_libraries(spectral_mech_basic 
                      PUBLIC
                      spectral_mech_utilities)
