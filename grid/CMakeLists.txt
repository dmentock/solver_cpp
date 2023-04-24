project(grid)

cmake_minimum_required(VERSION 3.10)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${PETSC_ARCH_DIR}/lib)

add_subdirectory(src)
add_subdirectory(test)