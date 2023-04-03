
module discretization
  use :: ISO_C_BINDING

  implicit none(type,external)
  private

  integer(c_int),     public, protected :: &
    discretization_nIPs, &
    discretization_Nelems, &
    discretization_Ncells

  integer(c_int),     public, protected, dimension(:),   allocatable :: &
    discretization_materialAt

  real(c_double), public, protected, dimension(:,:), allocatable :: &
    discretization_IPcoords0, &
    discretization_IPcoords, &
    discretization_NodeCoords0, &
    discretization_NodeCoords

  integer(c_int) :: &
    discretization_sharedNodesBegin

  public :: &
    discretization_init

contains

subroutine discretization_init(materialAt,&
                               discretization_Nelems,&
                               IPcoords0,NodeCoords0,&
                               sharedNodesBegin)
  use :: ISO_C_BINDING

  integer(c_int),     dimension(discretization_Nelems),   intent(in) :: &
    materialAt
  integer(c_int), intent(in) :: &
    discretization_Nelems
  real(c_double), dimension(discretization_Nelems,3), intent(in) :: &
    IPcoords0, &
    NodeCoords0
  integer(c_int), optional,           intent(in) :: &
    sharedNodesBegin

  print'(/,1x,a)', '<<<+-  discretization init  -+>>>'; flush(6)

  discretization_nIPs   = size(IPcoords0,2)/discretization_Nelems
  discretization_Ncells = discretization_Nelems*discretization_nIPs

  discretization_materialAt = materialAt

  discretization_IPcoords0   = IPcoords0
  discretization_IPcoords    = IPcoords0

  discretization_NodeCoords0 = NodeCoords0
  discretization_NodeCoords  = NodeCoords0

  if (present(sharedNodesBegin)) then
    discretization_sharedNodesBegin = sharedNodesBegin
  else
    discretization_sharedNodesBegin = size(discretization_NodeCoords0,2)
  end if

end subroutine discretization_init

end module discretization