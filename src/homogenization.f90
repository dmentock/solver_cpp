module homogenization
  use prec
  use math
  use iso_c_binding
  use discretization

  implicit none(type,external)

  public ::  &
    homogenization_init, &
    homogenization_mechanical_response, &
    homogenization_mechanical_response2, &
    homogenization_thermal_response

  logical, public, target :: terminallyIll = .false.
! General variables for the homogenization at a  material point
  real(pReal),   dimension(:,:,:),   allocatable,   target, public :: &
    homogenization_F0, &                                                                            !< def grad of IP at start of FE increment
    homogenization_F                                                                                !< def grad of IP to be reached at end of FE increment
  real(pReal),   dimension(:,:,:),    allocatable,  target, public :: & !, protected :: &                   Issue with ifort
    homogenization_P                                                                                !< first P--K stress of IP
  real(pReal),   dimension(:,:,:,:,:), allocatable, target, public :: & !, protected ::  &
    homogenization_dPdF

contains

subroutine homogenization_init() bind(C, name="f_homogenization_init")
  call mechanical_init()
end subroutine homogenization_init

subroutine mechanical_init()
  print'(/,1x,a)', '<<<+-  homogenization:mechanical init  -+>>>'
  allocate(homogenization_dPdF(3,3,3,3,discretization_Ncells), source=0.0_pReal)
  print *, "discretization_Ncells", discretization_Ncells
  homogenization_F0 = spread(math_I3, 3,discretization_Ncells)
  homogenization_F = homogenization_F0
  allocate(homogenization_P(3,3,discretization_Ncells),source=0.0_pReal)
end subroutine mechanical_init

subroutine homogenization_fetch_tensor_pointers(n_cells_global, &
                                                c_homog_F0, &
                                                c_homog_F, &
                                                c_homog_P, &
                                                c_homog_dPdF, &
                                                c_terminallyIll) &
  bind(C, name="f_homogenization_fetch_tensor_pointers")

  integer, intent(in) :: n_cells_global
  type(c_ptr), intent(out) :: &
    c_homog_F0, c_homog_F, &
    c_homog_P, &
    c_homog_dPdF, &
    c_terminallyIll

  ! allocate(homogenization_F0(3,3,n_cells_global),source=0.0_pReal)
  ! allocate(homogenization_F(3,3,n_cells_global),source=0.0_pReal)
  ! allocate(homogenization_P(3,3,n_cells_global),source=0.0_pReal)
  ! allocate(homogenization_dPdF(3,3,3,3,n_cells_global),source=0.0_pReal)

  c_homog_F0 = c_loc(homogenization_F0)
  c_homog_F = c_loc(homogenization_F)
  c_homog_P = c_loc(homogenization_P)
  c_homog_dPdF = c_loc(homogenization_dPdF)
  c_terminallyIll = c_loc(terminallyIll)

end subroutine homogenization_fetch_tensor_pointers


subroutine homogenization_mechanical_response(Delta_t,cell_start,cell_end) & 
  bind(C, name="f_homogenization_mechanical_response")

  real(pReal), intent(in) :: Delta_t                                                                !< time increment
  integer, intent(in) :: &
    cell_start, cell_end

end subroutine homogenization_mechanical_response


!--------------------------------------------------------------------------------------------------
!> @brief
!--------------------------------------------------------------------------------------------------
subroutine homogenization_thermal_response(Delta_t,cell_start,cell_end) &
  bind(C, name="f_homogenization_thermal_response")

  real(pReal), intent(in) :: Delta_t                                                                !< time increment
  integer, intent(in) :: &
    cell_start, cell_end


end subroutine homogenization_thermal_response


!--------------------------------------------------------------------------------------------------
!> @brief
!--------------------------------------------------------------------------------------------------
subroutine homogenization_mechanical_response2(Delta_t,FEsolving_execIP,FEsolving_execElem) &
  bind(C, name="f_homogenization_mechanical_response2")

  real(pReal), intent(in) :: Delta_t                                                                !< time increment
  integer, dimension(2), intent(in) :: FEsolving_execElem, FEsolving_execIP


end subroutine homogenization_mechanical_response2


end module homogenization
