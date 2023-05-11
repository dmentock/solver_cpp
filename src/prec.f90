module prec
  use, intrinsic :: IEEE_arithmetic

  implicit none(type,external)
  public

  ! https://stevelionel.com/drfortran/2017/03/27/doctor-fortran-in-it-takes-all-kinds
  integer,     parameter :: pReal      = IEEE_selected_real_kind(15,307)                            !< number with 15 significant digits, up to 1e+-307 (typically 64 bit)
  real(pReal), private, parameter :: PREAL_EPSILON = epsilon(0.0_pReal)                             !< minimum positive number such that 1.0 + EPSILON /= 1.0.
  real(pReal), private, parameter :: PREAL_MIN     = tiny(0.0_pReal)                                !< smallest normalized floating point number

  contains

  logical elemental pure function dEq0(a,tol)

    real(pReal), intent(in)           :: a
    real(pReal), intent(in), optional :: tol


    if (present(tol)) then
      dEq0 = abs(a) <= tol
    else
      dEq0 = abs(a) <= PREAL_MIN * 10.0_pReal
    end if
  end function dEq0

end module prec
