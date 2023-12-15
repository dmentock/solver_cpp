!--------------------------------------------------------------------------------------------------
!> @brief spatial discretization
!> @details serves as an abstraction layer between the different solvers and DAMASK
!--------------------------------------------------------------------------------------------------
module discretization

  use prec
  use result
  use iso_c_binding

  implicit none(type,external)
  private

  integer,     public, protected :: &
    discretization_nIPs, &
    discretization_Nelems, &
    discretization_Ncells

  integer,     public, protected, dimension(:),   allocatable :: &
    discretization_materialAt                                                                       !ToDo: discretization_ID_material

  real(pREAL), public, protected, target, dimension(:,:), allocatable :: &
    discretization_IPcoords0, &
    discretization_IPcoords, &
    discretization_NodeCoords0, &
    discretization_NodeCoords

  integer :: &
    discretization_sharedNodesBegin

  public :: &
    discretization_init, &
    discretization_result, &
    discretization_setIPcoords, &
    discretization_setNodeCoords

contains

!--------------------------------------------------------------------------------------------------
!> @brief stores the relevant information in globally accesible variables
!--------------------------------------------------------------------------------------------------
subroutine discretization_init(materialAt,&
                               IPcoords0,NodeCoords0,&
                               sharedNodesBegin)

  integer,     dimension(:),   intent(in) :: &
    materialAt
  real(pREAL), dimension(:,:), intent(in) :: &
    IPcoords0, &
    NodeCoords0
  integer, optional,           intent(in) :: &
    sharedNodesBegin                                                                                !< index of first node shared among different processes (MPI)

  print'(/,1x,a)', '<<<+-  discretization init  -+>>>'; flush(6)

  discretization_Nelems = size(materialAt,1)
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

!--------------------------------------------------------------------------------------------------
!> @brief stores the relevant information from the c++ grid solver in globally accesible variables
!--------------------------------------------------------------------------------------------------
subroutine discretization_init_c(materialAt, discretization_Nelems, &
                                 IPcoords0, n_ips, &
                                 NodeCoords0, n_nodes, &
                                 sharedNodesBegin) bind(C, name="f_discretization_init")

  integer,          intent(in) :: &
    discretization_Nelems, n_ips, n_nodes
  integer,     dimension(discretization_Nelems),   intent(in) :: &
    materialAt
  real(pReal), dimension(3,n_ips), intent(in) :: &
    IPcoords0
  real(pReal), dimension(3,n_nodes), intent(in) :: &
    NodeCoords0
  integer,          intent(in) :: &
    sharedNodesBegin                                                                                            !< index of first node shared among different processes (MPI)

  call discretization_init(materialAt, IPcoords0, NodeCoords0, sharedNodesBegin)

end subroutine discretization_init_c

!--------------------------------------------------------------------------------------------------
!> @brief write the displacements
!--------------------------------------------------------------------------------------------------
subroutine discretization_result()

  real(pREAL), dimension(:,:), allocatable :: u

  call result_closeGroup(result_addGroup('current/geometry'))

  u = discretization_NodeCoords (:,:discretization_sharedNodesBegin) &
    - discretization_NodeCoords0(:,:discretization_sharedNodesBegin)
  call result_writeDataset(u,'current/geometry','u_n','displacements of the nodes','m')

  u = discretization_IPcoords &
    - discretization_IPcoords0
  call result_writeDataset(u,'current/geometry','u_p','displacements of the materialpoints (cell centers)','m')

end subroutine discretization_result


!--------------------------------------------------------------------------------------------------
!> @brief stores current IP coordinates
!--------------------------------------------------------------------------------------------------
subroutine discretization_setIPcoords(IPcoords)

  real(pREAL), dimension(:,:), intent(in) :: IPcoords

  discretization_IPcoords = IPcoords

end subroutine discretization_setIPcoords


!--------------------------------------------------------------------------------------------------
!> @brief stores current IP coordinates
!--------------------------------------------------------------------------------------------------
subroutine discretization_setNodeCoords(NodeCoords)

  real(pREAL), dimension(:,:), intent(in) :: NodeCoords

  discretization_NodeCoords = NodeCoords

end subroutine discretization_setNodeCoords

!--------------------------------------------------------------------------------------------------
!> @brief pass pointers of fortran-defined IPcoords and NodeCoords arrays to cpp solver
!--------------------------------------------------------------------------------------------------
subroutine discretization_fetch_ip_node_coord_pointers(c_ip_coords_ptr, &
                                                       c_node_coords_ptr) &
  bind(C, name="f_discretization_fetch_ip_node_coord_pointers")

  type(c_ptr), intent(out) :: &
    c_ip_coords_ptr, &
    c_node_coords_ptr

  c_ip_coords_ptr = c_loc(discretization_IPcoords)
  c_node_coords_ptr = c_loc(discretization_NodeCoords)

end subroutine discretization_fetch_ip_node_coord_pointers


end module discretization
