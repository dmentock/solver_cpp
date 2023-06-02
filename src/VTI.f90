!--------------------------------------------------------------------------------------------------
!> @author Martin Diehl, KU Leuven
!> @brief Read data from image files of the visualization toolkit.
!--------------------------------------------------------------------------------------------------
module VTI
  use prec
  use iso_c_binding

  implicit none(type,external)
  private

  public :: &
    VTI_readDataset_int

contains

!--------------------------------------------------------------------------------------------------
!> @brief Read integer dataset from a VTK image data (*.vti) file.
!> @details https://vtk.org/Wiki/VTK_XML_Formats
!--------------------------------------------------------------------------------------------------
subroutine VTI_readDataset_int(materialAt, array_size) bind(C, name="f_VTI_readDataset_int")
  integer(c_int), intent(in) :: array_size
  integer(c_int), intent(inout) :: materialAt(array_size)

end subroutine VTI_readDataset_int



end module VTI