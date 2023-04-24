project(myproject)
cmake_minimum_required(VERSION 3.10)

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

find_package(Eigen3 REQUIRED)
find_package(VTK REQUIRED)
# # possibly use FetchContent to make version control of modules easier
# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/refs/tags/release-1.11.0.tar.gz
# )
# FetchContent_MakeAvailable(googletest)

set(FFTW_LIBRARIES "${PETSC_ARCH_DIR}/lib/libfftw3_mpi.a" "${PETSC_ARCH_DIR}/lib/libfftw3.a")

include_directories(${CMAKE_SOURCE_DIR}/include
                    ${MPI_INCLUDE_PATH}
                    ${PETSC_DIR}/include
                    ${PETSC_ARCH_DIR}/include
                    ${EIGEN3_INCLUDE_DIR})

add_subdirectory(src)
add_subdirectory(grid)
