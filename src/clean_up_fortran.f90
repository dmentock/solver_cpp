module clean_up_fortran
  use homogenization

  implicit none

contains  

  subroutine deallocate_resources() bind(C, name="f_deallocate_resources")
    if (allocated(homogenization_dPdF)) deallocate(homogenization_dPdF)
    if (allocated(homogenization_F0)) deallocate(homogenization_F0)
    if (allocated(homogenization_F)) deallocate(homogenization_F)
    if (allocated(homogenization_P)) deallocate(homogenization_P)
  end subroutine deallocate_resources

end module clean_up_fortran