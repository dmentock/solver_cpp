cmake_minimum_required(VERSION 3.10)

project(myproject)

set(CMAKE_CXX_STANDARD 11)

set(PETSC_DIR $ENV{PETSC_DIR})
set(PETSC_ARCH_DIR ${PETSC_DIR}/arch-linux-c-debug)
list(APPEND CMAKE_PREFIX_PATH "${PETSC_ARCH_DIR}")

unset(MPI_C_LIBRARIES CACHE)
unset(MPI_CXX_LIBRARIES CACHE)
set(MPI_C_COMPILER "${PETSC_ARCH_DIR}/bin/mpicc")
set(MPI_CXX_COMPILER "${PETSC_ARCH_DIR}/bin/mpicxx")
set(MPIEXEC_EXECUTABLE "${PETSC_ARCH_DIR}/bin/mpiexec")
find_path(MPI_INCLUDE_PATH mpi.h
          HINTS ${PETSC_ARCH_DIR}/include)
find_library(MPI_C_LIBRARIES NAMES mpich PATHS "${PETSC_ARCH_DIR}/lib" NO_DEFAULT_PATH)
find_library(MPI_CXX_LIBRARIES NAMES mpichcxx PATHS "${PETSC_ARCH_DIR}/lib" NO_DEFAULT_PATH)
message(STATUS "MPI_C_COMPILER: ${MPI_C_COMPILER}")
message(STATUS "MPI_CXX_COMPILER: ${MPI_CXX_COMPILER}")
message(STATUS "MPIEXEC_EXECUTABLE: ${MPIEXEC_EXECUTABLE}")
message(STATUS "MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")


find_package(VTK REQUIRED)
find_package(Eigen3 REQUIRED)

# set(FFTW_LIBRARIES "${PETSC_DIR}/lib/libfftw3_mpi.a" "${PETSC_DIR}/lib/libfftw3.a")


include_directories(${CMAKE_SOURCE_DIR}/include
                    ${MPI_INCLUDE_PATH}
                    ${PETSC_DIR}/include
                    ${PETSC_ARCH_DIR}/include
                    ${EIGEN3_INCLUDE_DIR})
link_directories(${PETSC_ARCH_DIR}/lib)

add_subdirectory(src)
add_subdirectory(test)