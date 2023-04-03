cmake_minimum_required(VERSION 3.10)

project(myproject)

set(CMAKE_CXX_STANDARD 11)

find_package(MPI REQUIRED)
find_package(VTK REQUIRED)

set(PETSC_DIR "/usr/local/petsc")
set(FFTW_LIBRARIES "${PETSC_DIR}/lib/libfftw3_mpi.a" "${PETSC_DIR}/lib/libfftw3.a")

include_directories(${CMAKE_SOURCE_DIR}/include
                    ${MPI_INCLUDE_PATH}
                    ${PETSC_DIR}/include)

add_subdirectory(src)
add_subdirectory(test)