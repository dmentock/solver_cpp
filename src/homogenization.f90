!--------------------------------------------------------------------------------------------------
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @author Denny Tjahjanto, Max-Planck-Institut für Eisenforschung GmbH
!> @brief homogenization manager, organizing deformation partitioning and stress homogenization
!--------------------------------------------------------------------------------------------------
module homogenization
  use prec
  use math
  use constants
  use IO
  use config
  use material
  use phase
  use discretization
  use HDF5
  use HDF5_utilities
  use result
  use crystal
  use iso_c_binding

  implicit none(type,external)
  private

  type :: tState
    integer :: &
      sizeState        = 0                                                                          !< size of state
    ! http://stackoverflow.com/questions/3948210
    real(pREAL), pointer,     dimension(:,:), contiguous :: &                                       !< is basically an allocatable+target, but in a type needs to be pointer
      state0, &
      state
  end type

  enum, bind(c); enumerator :: &
    THERMAL_UNDEFINED_ID, &
    THERMAL_PASS_ID, &
    THERMAL_ISOTEMPERATURE_ID
  end enum
  integer(kind(THERMAL_UNDEFINED_ID)), dimension(:),   allocatable :: &
    thermal_type                                                                                    !< type of each homogenization

  type(tState),        allocatable, dimension(:), public :: &
    homogState, &
    damageState_h

  logical,             allocatable, dimension(:) :: &
    thermal_active, &
    damage_active

  logical, public, target :: &
    terminallyIll = .false.                                                                         !< at least one material point is terminally ill

!--------------------------------------------------------------------------------------------------
! General variables for the homogenization at a  material point
  real(pREAL),   dimension(:,:,:), target, allocatable, public :: &
    homogenization_F0, &                                                                            !< def grad of IP at start of FE increment
    homogenization_F                                                                                !< def grad of IP to be reached at end of FE increment
  real(pREAL),   dimension(:,:,:), target, allocatable, public :: & !, protected :: &                   Issue with ifort
    homogenization_P                                                                                !< first P--K stress of IP
  real(pREAL),   dimension(:,:,:,:,:), target, allocatable, public :: & !, protected ::  &
    homogenization_dPdF                                                                             !< tangent of first P--K stress at IP

!--------------------------------------------------------------------------------------------------
  interface

    module subroutine mechanical_init()
    end subroutine mechanical_init

    module subroutine thermal_init()
    end subroutine thermal_init

    module subroutine damage_init()
    end subroutine damage_init

    module subroutine mechanical_partition(subF,ce)
      real(pREAL), intent(in), dimension(3,3) :: &
        subF
      integer,     intent(in) :: &
        ce
    end subroutine mechanical_partition

    module subroutine thermal_partition(ce)
      integer,     intent(in) :: ce
    end subroutine thermal_partition

    module subroutine damage_partition(ce)
      integer,     intent(in) :: ce
    end subroutine damage_partition

    module subroutine mechanical_homogenize(Delta_t,ce)
     real(pREAL), intent(in) :: Delta_t
     integer, intent(in) :: &
       ce                                                                                           !< cell
    end subroutine mechanical_homogenize

    module subroutine mechanical_result(group_base,ho)
      character(len=*), intent(in) :: group_base
      integer, intent(in)          :: ho
    end subroutine mechanical_result

    module subroutine damage_result(ho,group)
      integer,          intent(in) :: ho
      character(len=*), intent(in) :: group
    end subroutine damage_result

    module subroutine thermal_result(ho,group)
      integer,          intent(in) :: ho
      character(len=*), intent(in) :: group
    end subroutine thermal_result

    module function mechanical_updateState(subdt,subF,ce) result(doneAndHappy)
      real(pREAL), intent(in) :: &
        subdt                                                                                       !< current time step
      real(pREAL), intent(in), dimension(3,3) :: &
        subF
      integer,     intent(in) :: &
        ce                                                                                          !< cell
      logical, dimension(2) :: doneAndHappy
    end function mechanical_updateState

    module function homogenization_thermal_active() result(active)
      logical :: active
    end function homogenization_thermal_active

    module function homogenization_mu_T(ce) result(mu)
      integer, intent(in) :: ce
      real(pREAL) :: mu
    end function homogenization_mu_T

    module function homogenization_K_T(ce) result(K)
      integer, intent(in) :: ce
      real(pREAL), dimension(3,3) :: K
    end function homogenization_K_T

    module function homogenization_f_T(ce) result(f)
      integer, intent(in) :: ce
      real(pREAL) :: f
    end function homogenization_f_T

    module subroutine homogenization_thermal_setField(T,dot_T)
      real(pREAL), dimension(:),  intent(in) :: T, dot_T
    end subroutine homogenization_thermal_setField

    module function homogenization_damage_active() result(active)
      logical :: active
    end function homogenization_damage_active

    module function homogenization_mu_phi(ce) result(mu)
      integer, intent(in) :: ce
      real(pREAL) :: mu
    end function homogenization_mu_phi

    module function homogenization_K_phi(ce) result(K)
      integer, intent(in) :: ce
      real(pREAL), dimension(3,3) :: K
    end function homogenization_K_phi

    module function homogenization_f_phi(phi,ce) result(f)
      integer, intent(in) :: ce
      real(pREAL), intent(in) :: phi
      real(pREAL) :: f
    end function homogenization_f_phi

    module subroutine homogenization_set_phi(phi)
      real(pREAL), dimension(:), intent(in) :: phi
    end subroutine homogenization_set_phi

  end interface

  public ::  &
    homogenization_init, &
    homogenization_mechanical_response, &
    homogenization_mechanical_response2, &
    homogenization_thermal_response, &
    homogenization_thermal_active, &
    homogenization_mu_T, &
    homogenization_K_T, &
    homogenization_f_T, &
    homogenization_thermal_setfield, &
    homogenization_damage_active, &
    homogenization_mu_phi, &
    homogenization_K_phi, &
    homogenization_f_phi, &
    homogenization_set_phi, &
    homogenization_forward, &
    homogenization_result, &
    homogenization_restartRead, &
    homogenization_restartWrite

contains


!--------------------------------------------------------------------------------------------------
!> @brief module initialization
!--------------------------------------------------------------------------------------------------
subroutine homogenization_init() bind(C, name="f_homogenization_init")

  type(tDict) , pointer :: &
    num_homog, &
    num_homogGeneric

  print'(/,1x,a)', '<<<+-  homogenization init  -+>>>'; flush(IO_STDOUT)


  allocate(homogState      (size(material_name_homogenization)))
  allocate(damageState_h   (size(material_name_homogenization)))
  call parseHomogenization()

  call mechanical_init()
  call thermal_init()
  call damage_init()

end subroutine homogenization_init


!--------------------------------------------------------------------------------------------------
!> @brief
!--------------------------------------------------------------------------------------------------
subroutine homogenization_mechanical_response(Delta_t,cell_start,cell_end) &
  bind(C, name="f_homogenization_mechanical_response")

  real(pREAL), intent(in) :: Delta_t                                                                !< time increment
  integer, intent(in) :: &
    cell_start, cell_end
  integer :: &
    co, ce, ho, en
  logical :: &
    converged
  logical, dimension(2) :: &
    doneAndHappy

  print *, "<< homogenization_mechanical_response"
  do ce = cell_start, cell_end

    en = material_entry_homogenization(ce)
    ho = material_ID_homogenization(ce)

    call phase_restore(ce,.false.) ! wrong name (is more a forward function)

    if (homogState(ho)%sizeState > 0)  homogState(ho)%state(:,en) = homogState(ho)%state0(:,en)
    if (damageState_h(ho)%sizeState > 0) damageState_h(ho)%state(:,en) = damageState_h(ho)%state0(:,en)
    call damage_partition(ce)

    doneAndHappy = [.false.,.true.]

    convergenceLooping: do while (.not. (terminallyIll .or. doneAndHappy(1)))

      call mechanical_partition(homogenization_F(1:3,1:3,ce),ce)
      converged = all([(phase_mechanical_constitutive(Delta_t,co,ce),co=1,homogenization_Nconstituents(ho))])
      if (converged) then
        doneAndHappy = mechanical_updateState(Delta_t,homogenization_F(1:3,1:3,ce),ce)
        converged = all(doneAndHappy)
      else
        doneAndHappy = [.true.,.false.]
      end if
    end do convergenceLooping

    converged = converged .and. all([(phase_damage_constitutive(Delta_t,co,ce),co=1,homogenization_Nconstituents(ho))])

    if (.not. converged) then
      if (.not. terminallyIll) print*, ' Cell ', ce, ' terminally ill'
      terminallyIll = .true.
    end if
  end do

end subroutine homogenization_mechanical_response


!--------------------------------------------------------------------------------------------------
!> @brief
!--------------------------------------------------------------------------------------------------
subroutine homogenization_thermal_response(Delta_t,cell_start,cell_end) &
  bind(C, name="f_homogenization_thermal_response")

  real(pREAL), intent(in) :: Delta_t                                                                !< time increment
  integer, intent(in) :: &
    cell_start, cell_end
  integer :: &
    co, ce, ho


  do ce = cell_start, cell_end
    if (terminallyIll) continue
    ho = material_ID_homogenization(ce)
    do co = 1, homogenization_Nconstituents(ho)
      if (.not. phase_thermal_constitutive(Delta_t,material_ID_phase(co,ce),material_entry_phase(co,ce))) then
        if (.not. terminallyIll) print*, ' Cell ', ce, ' terminally ill'
        terminallyIll = .true.
      end if
    end do
  end do

end subroutine homogenization_thermal_response


!--------------------------------------------------------------------------------------------------
!> @brief
!--------------------------------------------------------------------------------------------------
subroutine homogenization_mechanical_response2(Delta_t,FEsolving_execIP,FEsolving_execElem) &
bind(C, name="f_homogenization_mechanical_response2")

  real(pREAL), intent(in) :: Delta_t                                                                !< time increment
  integer, dimension(2), intent(in) :: FEsolving_execElem, FEsolving_execIP
  integer :: &
    ip, &                                                                                           !< integration point number
    el, &                                                                                           !< element number
    co, ce, ho
  ! print *, "PP1", homogenization_P
  elementLooping3: do el = FEsolving_execElem(1),FEsolving_execElem(2)
    IpLooping3: do ip = FEsolving_execIP(1),FEsolving_execIP(2)
      ce = (el-1)*discretization_nIPs + ip
      ho = material_ID_homogenization(ce)
      do co = 1, homogenization_Nconstituents(ho)
        call crystallite_orientations(co,ip,el)
      end do
      call mechanical_homogenize(Delta_t,ce)

    end do IpLooping3

  end do elementLooping3
  print *, "FFF", homogenization_F


end subroutine homogenization_mechanical_response2


!--------------------------------------------------------------------------------------------------
!> @brief writes homogenization results to HDF5 output file
!--------------------------------------------------------------------------------------------------
subroutine homogenization_result

  integer :: ho
  character(len=:), allocatable :: group_base,group


  call result_closeGroup(result_addGroup('current/homogenization/'))

  do ho=1,size(material_name_homogenization)
    group_base = 'current/homogenization/'//trim(material_name_homogenization(ho))
    call result_closeGroup(result_addGroup(group_base))

    call mechanical_result(group_base,ho)

    if (damage_active(ho)) then
      group = trim(group_base)//'/damage'
      call result_closeGroup(result_addGroup(group))
      call damage_result(ho,group)
    end if

    if (thermal_active(ho)) then
      group = trim(group_base)//'/thermal'
      call result_closeGroup(result_addGroup(group))
      call thermal_result(ho,group)
    end if

 end do

end subroutine homogenization_result


!--------------------------------------------------------------------------------------------------
!> @brief Forward data after successful increment.
! ToDo: Any guessing for the current states possible?
!--------------------------------------------------------------------------------------------------
subroutine homogenization_forward

  integer :: ho


  do ho = 1, size(material_name_homogenization)
    homogState (ho)%state0 = homogState (ho)%state
    if (damageState_h(ho)%sizeState > 0) &
      damageState_h(ho)%state0 = damageState_h(ho)%state
  end do

end subroutine homogenization_forward


!--------------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------------
subroutine homogenization_restartWrite(fileHandle)

  integer(HID_T), intent(in) :: fileHandle

  integer(HID_T), dimension(2) :: groupHandle
  integer :: ho


  groupHandle(1) = HDF5_addGroup(fileHandle,'homogenization')

  do ho = 1, size(material_name_homogenization)

    groupHandle(2) = HDF5_addGroup(groupHandle(1),material_name_homogenization(ho))

    call HDF5_write(homogState(ho)%state,groupHandle(2),'omega_mechanical') ! ToDo: should be done by mech

    if (damageState_h(ho)%sizeState > 0) &
      call HDF5_write(damageState_h(ho)%state,groupHandle(2),'omega_damage') ! ToDo: should be done by mech

    call HDF5_closeGroup(groupHandle(2))

  end do

  call HDF5_closeGroup(groupHandle(1))

end subroutine homogenization_restartWrite


!--------------------------------------------------------------------------------------------------
!--------------------------------------------------------------------------------------------------
subroutine homogenization_restartRead(fileHandle)

  integer(HID_T), intent(in) :: fileHandle

  integer(HID_T), dimension(2) :: groupHandle
  integer :: ho


  groupHandle(1) = HDF5_openGroup(fileHandle,'homogenization')

  do ho = 1, size(material_name_homogenization)

    groupHandle(2) = HDF5_openGroup(groupHandle(1),material_name_homogenization(ho))

    call HDF5_read(homogState(ho)%state0,groupHandle(2),'omega_mechanical') ! ToDo: should be done by mech

    if (damageState_h(ho)%sizeState > 0) &
      call HDF5_read(damageState_h(ho)%state0,groupHandle(2),'omega_damage') ! ToDo: should be done by mech

    call HDF5_closeGroup(groupHandle(2))

  end do

  call HDF5_closeGroup(groupHandle(1))

end subroutine homogenization_restartRead


!--------------------------------------------------------------------------------------------------
!> @brief parses the homogenization part from the material configuration
!--------------------------------------------------------------------------------------------------
subroutine parseHomogenization

  type(tDict), pointer :: &
    material_homogenization, &
    homog, &
    homogThermal, &
    homogDamage

  integer :: h

  material_homogenization => config_material%get_dict('homogenization')

  allocate(thermal_type(size(material_name_homogenization)),source=THERMAL_UNDEFINED_ID)
  allocate(thermal_active(size(material_name_homogenization)),source=.false.)
  allocate(damage_active(size(material_name_homogenization)),source=.false.)

  do h=1, size(material_name_homogenization)
    homog => material_homogenization%get_dict(h)

    if (homog%contains('thermal')) then
      homogThermal => homog%get_dict('thermal')
        select case (homogThermal%get_asStr('type'))
          case('pass')
            thermal_type(h) = THERMAL_PASS_ID
            thermal_active(h) = .true.
          case('isotemperature')
            thermal_type(h) = THERMAL_ISOTEMPERATURE_ID
            thermal_active(h) = .true.
          case default
            call IO_error(500,ext_msg=homogThermal%get_asStr('type'))
        end select
    end if

    if (homog%contains('damage')) then
      homogDamage => homog%get_dict('damage')
        select case (homogDamage%get_asStr('type'))
          case('pass')
            damage_active(h) = .true.
          case default
            call IO_error(500,ext_msg=homogDamage%get_asStr('type'))
        end select
    end if
  end do

end subroutine parseHomogenization

!--------------------------------------------------------------------------------------------------
!> @brief pass pointers of fortran-defined homogenization arrays to cpp solver
!--------------------------------------------------------------------------------------------------
subroutine homogenization_fetch_tensor_pointers(c_homog_F0, &
                                                c_homog_F, &
                                                c_homog_P, &
                                                c_homog_dPdF, &
                                                c_terminallyIll) &
  bind(C, name="f_homogenization_fetch_tensor_pointers")

  type(c_ptr), intent(out) :: &
    c_homog_F0, c_homog_F, &
    c_homog_P, &
    c_homog_dPdF, &
    c_terminallyIll

  c_homog_F0 = c_loc(homogenization_F0)
  c_homog_F = c_loc(homogenization_F)
  c_homog_P = c_loc(homogenization_P)
  c_homog_dPdF = c_loc(homogenization_dPdF)
  c_terminallyIll = c_loc(terminallyIll)

  end subroutine homogenization_fetch_tensor_pointers

end module homogenization
