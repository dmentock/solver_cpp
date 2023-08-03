!--------------------------------------------------------------------------------------------------
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @brief needs a good name and description
!--------------------------------------------------------------------------------------------------
module materialpoint
  use homogenization

  implicit none(type,external)
  public

contains


!--------------------------------------------------------------------------------------------------
!> @brief Initialize all modules.
!--------------------------------------------------------------------------------------------------
subroutine materialpoint_initAll() bind(C, name="f_materialpoint_initAll")
  print *, "MAT"
  call homogenization_init()

end subroutine materialpoint_initAll

end module materialpoint
