enable_language(Fortran)
# add_library(discretization SHARED discretization.f90)
# target_compile_options(discretization PUBLIC "-fvisibility=default")

add_library(f_prec SHARED prec.f90)
add_library(f_lapack SHARED lapack.f90)
target_link_libraries(f_lapack 
                      PUBLIC
                      f_prec)

add_library(f_misc SHARED misc.f90)
target_link_libraries(f_misc 
                      PUBLIC
                      f_prec)

add_library(f_math SHARED math.f90)
target_link_libraries(f_math 
                      PUBLIC
                      f_misc
                      f_lapack)


add_library(f_rotations SHARED rotations.f90)
target_link_libraries(f_rotations 
add_library(VTI VTI.f90)
target_link_libraries(VTI
                      PUBLIC
                      f_prec)


add_library(discretization discretization.cpp)

add_library(config config.cpp)
target_link_libraries(config yaml-cpp)
