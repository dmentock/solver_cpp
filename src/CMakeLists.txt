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

add_library(f_discretization SHARED discretization.f90)
target_link_libraries(f_discretization
                      PUBLIC
                      f_VTI)

                                            
add_library(f_homogenization_mechanical SHARED homogenization_mechanical.f90)
target_link_libraries(f_homogenization_mechanical 
                      PUBLIC
                      f_prec
                      f_discretization
                      f_math)

add_library(f_homogenization SHARED homogenization.f90)
target_link_libraries(f_homogenization 
                      PUBLIC
                      f_homogenization_mechanical
                      f_math)

add_library(f_rotations SHARED rotations.f90)
target_link_libraries(f_rotations 
                      PUBLIC
                      f_math)

add_library(f_VTI SHARED VTI.f90)
target_link_libraries(f_VTI
                      PUBLIC
                      f_prec)

add_library(f_clean_up_fortran SHARED clean_up_fortran.f90)
target_link_libraries(f_clean_up_fortran
                      PUBLIC
                      f_homogenization)


# add_library(discretization discretization.cpp)

add_library(config config.cpp)
target_link_libraries(config yaml-cpp)


