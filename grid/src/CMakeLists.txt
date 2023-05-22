add_library(discretization_grid discretization_grid.cpp)
target_link_libraries(discretization_grid 
                      PUBLIC
                      discretization
                      ${MPI_C_LIBRARIES}
                      ${MPI_CXX_LIBRARIES}
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      -lfftw3)
                      
add_library(spectral spectral/spectral.cpp)
target_link_libraries(spectral 
                      f_math
                      f_rotations
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      -lfftw3)

add_library(spectral_fields_mech spectral/fields/mech.cpp)
target_link_libraries(spectral_fields_mech 
                      spectral
                      discretization_grid
                      config
                      f_math
                      f_rotations
                      ${PETSC_ARCH_DIR}/lib/libpetsc.so
                      ${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_ARCH_DIR}/lib/libmpi.a
                      -lfftw3)

add_library(spectral_solver_basic spectral/solver/basic.cpp)
target_link_libraries(spectral_solver_basic 
                      PUBLIC
                      spectral_fields_mech)
