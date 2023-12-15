file(GLOB damask-sources CONFIGURE_DEPENDS *.f90 *.c)

if(PROJECT_NAME STREQUAL "damask-grid")
    set(executable-name "DAMASK_grid")
    set(PETSC_ARCH_DIR $ENV{PETSC_DIR}/$ENV{PETSC_ARCH})
    set(LAPACK_LIBRARIES "${PETSC_ARCH_DIR}/lib/libflapack.a")
    set(BLAS_LIBRARIES "${PETSC_ARCH_DIR}/lib/libfblas.a")
    set(HDF5_ROOT "${PETSC_ARCH_DIR}")

    find_library(FYAML_LIBRARY NAMES fyaml)
    find_package(OpenMP REQUIRED)
    find_package(HDF5 REQUIRED COMPONENTS C Fortran HL)
    find_package(ZLIB REQUIRED)
    find_package(Eigen3 REQUIRED)
    
    message(STATUS "PETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIRPETSC_ARCH_DIR: ${PETSC_ARCH_DIR}")

    link_directories(${PETSC_ARCH_DIR}/lib)
    include_directories(
      $ENV{PETSC_DIR}/include
      ${PETSC_ARCH_DIR}/include
      ${EIGEN3_INCLUDE_DIR}
      grid)

    add_library(fortran_library OBJECT ${damask-sources} grid/base64.f90)
    target_link_libraries(fortran_library PUBLIC
                          ${LAPACK_LIBRARIES}
                          ${BLAS_LIBRARIES}
                          ${PETSC_ARCH_DIR}/lib/libpetsc.so
                          HDF5::HDF5
                          ${FYAML_LIBRARY}
                          OpenMP::OpenMP_CXX)

    add_subdirectory(grid)
    add_executable(${executable-name} grid/main.cpp)
    target_link_libraries(${executable-name}
                          PUBLIC
                          fortran_library
                          damask_grid_lib)
    install(TARGETS ${executable-name} RUNTIME DESTINATION bin)

else()
  if(PROJECT_NAME STREQUAL "damask-mesh")
    set(executable-name "DAMASK_mesh")
    file(GLOB solver-sources CONFIGURE_DEPENDS mesh/*.f90)
  elseif(PROJECT_NAME STREQUAL "damask-test")
    set(executable-name "DAMASK_test")
    file(GLOB solver-sources CONFIGURE_DEPENDS test/*.f90)
  endif()

  if(NOT CMAKE_BUILD_TYPE STREQUAL "SYNTAXONLY")
    add_executable(${executable-name} ${damask-sources} ${solver-sources})
    install(TARGETS ${executable-name} RUNTIME DESTINATION bin)
  else()
    add_library(${executable-name} OBJECT ${damask-sources} ${solver-sources})
    exec_program(mktemp OUTPUT_VARIABLE nothing)
    exec_program(mktemp ARGS -d OUTPUT_VARIABLE black_hole)
    install(PROGRAMS ${nothing} DESTINATION ${black_hole})
  endif()
endif()

string(REPLACE ";" "\n" sources "${damask-sources};${solver-sources}")
message(${CMAKE_BINARY_DIR})
file(WRITE ${CMAKE_BINARY_DIR}/sources.txt ${sources})
