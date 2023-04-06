enable_language(Fortran)

add_library(discretization SHARED discretization.f90)

target_compile_options(discretization PUBLIC "-fvisibility=default")

add_library(vti_reader vti_reader.cpp)
target_link_libraries(vti_reader PUBLIC ${VTK_LIBRARIES} ${MPI_LIBRARIES})

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

add_library(grid_mech_spectral_basic spectral/mech/basic.cpp)
target_link_libraries(grid_mech_spectral_basic 
                    PUBLIC
                    ${PETSC_ARCH_DIR}/lib/libpetsc.so
                    ${MPI_C_LIBRARIES}
                    ${MPI_CXX_LIBRARIES})
