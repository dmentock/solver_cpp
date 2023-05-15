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
# target_compile_options(f_math PUBLIC "-fvisibility=default")

add_library(f_rotations SHARED rotations.f90)
target_link_libraries(f_rotations 
                    PUBLIC
                    f_math)
# target_compile_options(f_rotations PUBLIC "-fvisibility=default")

add_library(discretization discretization.cpp)

add_library(yaml_reader yaml_reader.cpp)
target_link_libraries(yaml_reader yaml-cpp)
