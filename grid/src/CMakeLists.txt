add_library(vti_reader vti_reader.cpp)
target_link_libraries(vti_reader PUBLIC 
                      ${VTK_LIBRARIES}
                      ${MPI_C_LIBRARIES}
                      ${MPI_CXX_LIBRARIES})

add_library(discretization_grid discretization_grid.cpp)
target_link_libraries(discretization_grid 
                      PUBLIC
                      discretization
                      ${MPI_C_LIBRARIES}
                      ${MPI_CXX_LIBRARIES}
                      ${VTK_LIBRARIES}
                      ${PETSC_DIR}/lib/libfftw3_mpi.a
                      ${PETSC_DIR}/lib/libmpi.a
                      -lfftw3)

add_library(grid_spectral SHARED spectral/spectral.cpp)
add_library(grid_spectral_mech_basic spectral/mech/basic.cpp)
target_link_libraries(grid_spectral_mech_basic 
                    PUBLIC
                    grid_spectral
                    ${PETSC_ARCH_DIR}/lib/libpetsc.so
                    ${MPI_C_LIBRARIES}
                    ${MPI_CXX_LIBRARIES})
