!--------------------------------------------------------------------------------------------------
!> @author Martin Diehl, KU Leuven
!> @brief Partition F and homogenize P/dPdF
!--------------------------------------------------------------------------------------------------
submodule(homogenization) mechanical


  interface

    module subroutine pass_init()
    end subroutine pass_init

    module subroutine isostrain_init()
    end subroutine isostrain_init

    module subroutine RGC_init()
    end subroutine RGC_init


    module subroutine isostrain_partitionDeformation(F,avgF)
      real(pREAL),   dimension (:,:,:), intent(out) :: F                                            !< partitioned deformation gradient
      real(pREAL),   dimension (3,3),   intent(in)  :: avgF                                         !< average deformation gradient at material point
    end subroutine isostrain_partitionDeformation

    module subroutine RGC_partitionDeformation(F,avgF,ce)
      real(pREAL),   dimension (:,:,:), intent(out) :: F                                            !< partitioned deformation gradient
      real(pREAL),   dimension (3,3),   intent(in)  :: avgF                                         !< average deformation gradient at material point
      integer,                          intent(in)  :: &
        ce
    end subroutine RGC_partitionDeformation


    module function RGC_updateState(P,F,avgF,dt,dPdF,ce) result(doneAndHappy)
      logical, dimension(2) :: doneAndHappy
      real(pREAL), dimension(:,:,:),     intent(in)    :: &
        P,&                                                                                         !< partitioned stresses
        F                                                                                           !< partitioned deformation gradients
      real(pREAL), dimension(:,:,:,:,:), intent(in) :: dPdF                                         !< partitioned stiffnesses
      real(pREAL), dimension(3,3),       intent(in) :: avgF                                         !< average F
      real(pREAL),                       intent(in) :: dt                                           !< time increment
      integer,                           intent(in) :: &
        ce                                                                                          !< cell
    end function RGC_updateState


    module subroutine RGC_result(ho,group)
      integer,          intent(in) :: ho                                                            !< homogenization type
      character(len=*), intent(in) :: group                                                         !< group name in HDF5 file
    end subroutine RGC_result

  end interface

  type :: tOutput                                                                                   !< requested output (per phase)
    character(len=pSTRLEN), allocatable, dimension(:) :: &
      label
  end type tOutput
  type(tOutput), allocatable, dimension(:) :: output_mechanical

  enum, bind(c); enumerator :: &
    MECHANICAL_UNDEFINED_ID, &
    MECHANICAL_PASS_ID, &
    MECHANICAL_ISOSTRAIN_ID, &
    MECHANICAL_RGC_ID
  end enum
  integer(kind(MECHANICAL_UNDEFINED_ID)), dimension(:),   allocatable :: &
    mechanical_type                                                                                 !< type of each homogenization

contains

!--------------------------------------------------------------------------------------------------
!> @brief Allocate variables and set parameters.
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_init()

  print'(/,1x,a)', '<<<+-  homogenization:mechanical init  -+>>>'

  call parseMechanical()

  allocate(homogenization_dPdF(3,3,3,3,discretization_Ncells), source=0.0_pREAL)
  homogenization_F0 = spread(math_I3,3,discretization_Ncells)
  homogenization_F = homogenization_F0
  allocate(homogenization_P(3,3,discretization_Ncells),source=0.0_pREAL)

  if (any(mechanical_type == MECHANICAL_PASS_ID))      call pass_init()
  if (any(mechanical_type == MECHANICAL_ISOSTRAIN_ID)) call isostrain_init()
  if (any(mechanical_type == MECHANICAL_RGC_ID))       call RGC_init()

end subroutine mechanical_init


!--------------------------------------------------------------------------------------------------
!> @brief Partition F onto the individual constituents.
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_partition(subF,ce)

  real(pREAL), intent(in), dimension(3,3) :: &
    subF
  integer,     intent(in) :: &
    ce

  integer :: co
  real(pREAL), dimension (3,3,homogenization_Nconstituents(material_ID_homogenization(ce))) :: Fs

  chosenHomogenization: select case(mechanical_type(material_ID_homogenization(ce)))

    case (MECHANICAL_PASS_ID) chosenHomogenization
      Fs(1:3,1:3,1) = subF

    case (MECHANICAL_ISOSTRAIN_ID) chosenHomogenization
      call isostrain_partitionDeformation(Fs,subF)

    case (MECHANICAL_RGC_ID) chosenHomogenization
      call RGC_partitionDeformation(Fs,subF,ce)

  end select chosenHomogenization

  do co = 1,homogenization_Nconstituents(material_ID_homogenization(ce))
    call phase_set_F(Fs(1:3,1:3,co),co,ce)
  end do


end subroutine mechanical_partition


!--------------------------------------------------------------------------------------------------
!> @brief Average P and dPdF from the individual constituents.
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_homogenize(Delta_t,ce)

  real(pREAL), intent(in) :: Delta_t
  integer, intent(in) :: ce

  integer :: co

  homogenization_P(1:3,1:3,ce)            = phase_P(1,ce)*material_v(1,ce)
  ! print *, "phase_P(1,ce)1", phase_P(1,ce)
  ! print *, "material_v(1,ce)1", material_v(1,ce)

  homogenization_dPdF(1:3,1:3,1:3,1:3,ce) = phase_mechanical_dPdF(Delta_t,1,ce)*material_v(1,ce)
  do co = 2, homogenization_Nconstituents(material_ID_homogenization(ce))
    homogenization_P(1:3,1:3,ce)            = homogenization_P(1:3,1:3,ce) &
                                            + phase_P(co,ce)*material_v(co,ce)
    homogenization_dPdF(1:3,1:3,1:3,1:3,ce) = homogenization_dPdF(1:3,1:3,1:3,1:3,ce) &
                                            + phase_mechanical_dPdF(Delta_t,co,ce)*material_v(co,ce)
  end do
  ! if (homogenization_P(1,1,ce)/=0) then
  !   print *, ""
  ! end if
  ! print *, "phase_P(1,ce)2", phase_P(1,ce)
  ! print *, "material_v(1,ce)2", material_v(1,ce)
end subroutine mechanical_homogenize


!--------------------------------------------------------------------------------------------------
!> @brief update the internal state of the homogenization scheme and tell whether "done" and
!> "happy" with result
!--------------------------------------------------------------------------------------------------
module function mechanical_updateState(subdt,subF,ce) result(doneAndHappy)

  real(pREAL), intent(in) :: &
    subdt                                                                                           !< current time step
  real(pREAL), intent(in), dimension(3,3) :: &
    subF
  integer,     intent(in) :: &
    ce
  logical, dimension(2) :: doneAndHappy

  integer :: co
  real(pREAL) :: dPdFs(3,3,3,3,homogenization_Nconstituents(material_ID_homogenization(ce)))
  real(pREAL) :: Fs(3,3,homogenization_Nconstituents(material_ID_homogenization(ce)))
  real(pREAL) :: Ps(3,3,homogenization_Nconstituents(material_ID_homogenization(ce)))


  if (mechanical_type(material_ID_homogenization(ce)) == MECHANICAL_RGC_ID) then
      do co = 1, homogenization_Nconstituents(material_ID_homogenization(ce))
        dPdFs(:,:,:,:,co) = phase_mechanical_dPdF(subdt,co,ce)
        Fs(:,:,co)        = phase_F(co,ce)
        Ps(:,:,co)        = phase_P(co,ce)
      end do
      doneAndHappy = RGC_updateState(Ps,Fs,subF,subdt,dPdFs,ce)
  else
    doneAndHappy = .true.
  end if

end function mechanical_updateState


!--------------------------------------------------------------------------------------------------
!> @brief Write results to file.
!--------------------------------------------------------------------------------------------------
module subroutine mechanical_result(group_base,ho)

  character(len=*), intent(in) :: group_base
  integer, intent(in)          :: ho

  integer :: ou
  character(len=:), allocatable :: group


  group = trim(group_base)//'/mechanical'
  call result_closeGroup(result_addGroup(group))

  select case(mechanical_type(ho))

    case(MECHANICAL_RGC_ID)
      call RGC_result(ho,group)

  end select

  do ou = 1, size(output_mechanical(1)%label)

    select case (output_mechanical(ho)%label(ou))
      case('F')
        call result_writeDataset(reshape(homogenization_F,[3,3,discretization_nCells]),group,'F', &
                                 'deformation gradient','1')
      case('P')
        call result_writeDataset(reshape(homogenization_P,[3,3,discretization_nCells]),group,'P', &
                                 'first Piola-Kirchhoff stress','Pa')
    end select
  end do

end subroutine mechanical_result


!--------------------------------------------------------------------------------------------------
!> @brief parses the homogenization part from the material configuration
!--------------------------------------------------------------------------------------------------
subroutine parseMechanical()

  type(tDict), pointer :: &
    material_homogenization, &
    homog, &
    mechanical

  integer :: ho


  material_homogenization => config_material%get_dict('homogenization')

  allocate(mechanical_type(size(material_name_homogenization)), source=MECHANICAL_UNDEFINED_ID)
  allocate(output_mechanical(size(material_name_homogenization)))

  do ho=1, size(material_name_homogenization)
    homog => material_homogenization%get_dict(ho)
    mechanical => homog%get_dict('mechanical')
#if defined(__GFORTRAN__)
    output_mechanical(ho)%label = output_as1dStr(mechanical)
#else
    output_mechanical(ho)%label = mechanical%get_as1dStr('output',defaultVal=emptyStrArray)
#endif
    select case (mechanical%get_asStr('type'))
      case('pass')
        mechanical_type(ho) = MECHANICAL_PASS_ID
      case('isostrain')
        mechanical_type(ho) = MECHANICAL_ISOSTRAIN_ID
      case('RGC')
        mechanical_type(ho) = MECHANICAL_RGC_ID
      case default
        call IO_error(500,ext_msg=mechanical%get_asStr('type'))
    end select
  end do

end subroutine parseMechanical


end submodule mechanical
