# enable_language(Fortran)
# add_library(discretization SHARED discretization.f90)
# target_compile_options(discretization PUBLIC "-fvisibility=default")
add_library(discretization discretization.cpp)
