###################################################################################################
# GNU Compiler
###################################################################################################
if (CMAKE_Fortran_COMPILER_VERSION VERSION_LESS 9.0)
  message (FATAL_ERROR "GCC Compiler version: ${CMAKE_Fortran_COMPILER_VERSION} not supported")
endif ()

if (OPENMP)
  set (OPENMP_FLAGS "-fopenmp")
endif ()

if (OPTIMIZATION STREQUAL "DEBUG")
  set (OPTIMIZATION_FLAGS "-Og")
elseif (OPTIMIZATION STREQUAL "OFF")
  set (OPTIMIZATION_FLAGS "-O0")
elseif (OPTIMIZATION STREQUAL "DEFENSIVE")
  set (OPTIMIZATION_FLAGS "-O2 -mtune=native")
elseif (OPTIMIZATION STREQUAL "AGGRESSIVE")
  set (OPTIMIZATION_FLAGS "-O3 -march=native -funroll-loops -ftree-vectorize -flto")
endif ()

set (STANDARD_CHECK "-std=f2018 -pedantic-errors" )

#------------------------------------------------------------------------------------------------
# Fine tuning compilation options
set (COMPILE_FLAGS "${COMPILE_FLAGS} -cpp")
# preprocessor

set (COMPILE_FLAGS "${COMPILE_FLAGS} -fPIE")
# position independent code

set (COMPILE_FLAGS "${COMPILE_FLAGS} -ffree-line-length-none")
# PETSc macros are long, line length is enforced in pre-receive hook

set (COMPILE_FLAGS "${COMPILE_FLAGS} -fimplicit-none")
# assume "implicit none" even if not present in source

set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wall")
# sets the following Fortran options:
#   -Waliasing:                   warn about possible aliasing of dummy arguments. Specifically, it warns if the same actual argument is associated with a dummy argument with "INTENT(IN)" and a dummy argument with "INTENT(OUT)" in a call with an explicit interface.
#   -Wampersand:                  checks if a character expression is continued proberly by an ampersand at the end of the line and at the beginning of the new line
#   -Warray-bounds:               checks if array reference is out of bounds at compile time. use -fcheck-bounds to also check during runtime
#   -Wconversion:                 warn about implicit conversions between different type
#   -Wsurprising:                 warn when "suspicious" code constructs are encountered. While technically legal these usually indicate that an error has been made.
#   -Wc-binding-type:
#   -Wintrinsics-std:             only standard intrisics are available, e.g. "call flush(6)" will cause an error
#   -Wno-tabs:                    do not allow tabs in source
#   -Wintrinsic-shadow:           warn if a user-defined procedure or module procedure has the same name as an intrinsic
#   -Wline-truncation:
#   -Wtarget-lifetime:
#   -Wreal-q-constant:            warn about real-literal-constants with 'q'  exponent-letter
#   -Wunused:                     a number of unused-xxx warnings
# and sets the general (non-Fortran options) options:
#   -Waddress
#   -Warray-bounds (only with -O2)
#   -Wc++11-compat
#   -Wchar-subscripts
#   -Wcomment
#   -Wformat
#   -Wmaybe-uninitialized
#   -Wnonnull
#   -Wparentheses
#   -Wpointer-sign
#   -Wreorder
#   -Wreturn-type
#   -Wsequence-point
#   -Wstrict-aliasing
#   -Wstrict-overflow=1
#   -Wswitch
#   -Wtrigraphs
#   -Wuninitialized
#   -Wunknown-pragmas
#   -Wunused-function
#   -Wunused-label
#   -Wunused-value
#   -Wunused-variable
#   -Wvolatile-register-var

set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wextra")
# sets the following Fortran options:
#   -Wunuses-parameter:
#   -Wcompare-reals:
# and sets the general (non-Fortran options) options:
#   -Wclobbered
#   -Wempty-body
#   -Wignored-qualifiers
#   -Wmissing-field-initializers
#   -Woverride-init
#   -Wsign-compare
#   -Wtype-limits
#   -Wuninitialized
#   -Wunused-but-set-parameter (only with -Wunused or -Wall)
#   -Wno-globals

set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wcharacter-truncation")
# warn if character expressions (strings) are truncated

set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wunderflow")
# produce a warning when numerical constant expressions are encountered, which yield an UNDERFLOW during compilation

set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wsuggest-attribute=pure")
set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wsuggest-attribute=noreturn")
set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wconversion-extra")
set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wimplicit-procedure")
set (COMPILE_FLAGS "${COMPILE_FLAGS} -Wunused-parameter")
set (COMPILE_FLAGS "${COMPILE_FLAGS} -ffpe-summary=all")
# print summary of floating point exeptions (invalid,zero,overflow,underflow,inexact,denormal)

# Additional options
# -Wimplicit-interface:          no interfaces for lapack/MPI routines
# -Wunsafe-loop-optimizations:   warn if the loop cannot be optimized due to nontrivial assumptions

#------------------------------------------------------------------------------------------------
# Runtime debugging
set (DEBUG_FLAGS "${DEBUG_FLAGS} -ffpe-trap=invalid,zero,overflow")
# stop execution if floating point exception is detected (NaN is silent)
# Additional options
# -ffpe-trap=precision,denormal,underflow

set (DEBUG_FLAGS "${DEBUG_FLAGS} -g")
# Generate symbolic debugging information in the object file

set (DEBUG_FLAGS "${DEBUG_FLAGS} -Og")
# Optimize debugging experience

set (DEBUG_FLAGS "${DEBUG_FLAGS} -fbacktrace")
set (DEBUG_FLAGS "${DEBUG_FLAGS} -fdump-core")
set (DEBUG_FLAGS "${DEBUG_FLAGS} -fcheck=all")
# checks for (array-temps,bounds,do,mem,pointer,recursion)

set (DEBUG_FLAGS "${DEBUG_FLAGS} -fstack-protector-all")
# Inserts a guard variable onto the stack frame for all functions

set (DEBUG_FLAGS "${DEBUG_FLAGS} -finit-real=snan -finit-integer=-2147483648")
# "strange" values to simplify debugging

set (DEBUG_FLAGS "${DEBUG_FLAGS} -fsanitize=undefined")
# detect undefined behavior
# Additional options
# -fsanitize=address,leak,thread
