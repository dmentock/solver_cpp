project(myproject)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 20)
# set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_CXX_EXTENSIONS OFF)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fmodules-ts")

add_compile_options(-Wno-ignored-attributes)

set(PETSC_DIR $ENV{PETSC_DIR})
set(PETSC_ARCH_DIR ${PETSC_DIR}/arch-linux-c-debug)
list(APPEND CMAKE_PREFIX_PATH "${PETSC_ARCH_DIR}")

set(MPI_INCLUDE_PATH "${PETSC_DIR}/${PETSC_ARCH}/include")
set(MPI_LIBRARIES "${PETSC_DIR}/${PETSC_ARCH}/lib/libmpi.so")
set(MPI_C_COMPILER "${PETSC_ARCH_DIR}/bin/mpicc")
set(MPI_CXX_COMPILER "${PETSC_ARCH_DIR}/bin/mpicxx")
set(MPIEXEC_EXECUTABLE "${PETSC_ARCH_DIR}/bin/mpiexec")
find_library(MPI_C_LIBRARIES NAMES mpi PATHS "${PETSC_ARCH_DIR}/lib" NO_DEFAULT_PATH)
find_library(MPI_CXX_LIBRARIES NAMES mpicxx PATHS "${PETSC_ARCH_DIR}/lib" NO_DEFAULT_PATH)

include_directories(${MPI_INCLUDE_PATH})

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

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
add_subdirectory(test)
add_subdirectory(grid)
