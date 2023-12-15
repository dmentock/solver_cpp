!--------------------------------------------------------------------------------------------------
!> @author Franz Roters, Max-Planck-Institut für Eisenforschung GmbH
!> @author Philip Eisenlohr, Max-Planck-Institut für Eisenforschung GmbH
!> @brief needs a good name and description
!--------------------------------------------------------------------------------------------------
module materialpoint
  use parallelization
  use signal
  use CLI
  use prec
  use misc
  use IO
  use YAML_types
  use YAML_parse
  use HDF5
  use HDF5_utilities
  use result
  use config
  use math
  use rotations
  use polynomials
  use tables
  use crystal
  use material
  use phase
  use homogenization
  use discretization
#if   defined(MESH)
  use FEM_quadrature
  use discretization_mesh
#elif defined(GRID)
  use base64
  use iso_c_binding
#endif

  implicit none(type,external)
  public

contains


!--------------------------------------------------------------------------------------------------
!> @brief Initialize all modules.
!--------------------------------------------------------------------------------------------------
subroutine materialpoint_initAll()

  call materialpoint_initBase()
  call materialpoint_initDamask()

end subroutine materialpoint_initAll

function c_str_to_f_str(c_string, string_length) result(f_string)
  integer(c_int), value :: string_length
  character(kind=c_char), dimension(string_length), intent(in) :: c_string
  character(len=string_length) :: f_string

  integer :: i, str_length

  str_length = 0
  do i = 1, string_length
    if (c_string(i) == c_null_char) exit
    f_string(i:i) = c_string(i)
    str_length = str_length + 1
  end do

end function c_str_to_f_str

subroutine materialpoint_initBase_c(material_path_c, material_path_len, &
                                    load_path_c, load_path_len, &
                                    grid_path_c, grid_path_len, &
                                    numerics_path_c, numerics_path_len) &
  bind(C, name="f_materialpoint_initBase")
  character(kind=c_char), intent(in) :: &
    material_path_c, &
    load_path_c, &
    grid_path_c, &
    numerics_path_c
  integer(c_int), value, intent(in) :: &
    material_path_len, &
    load_path_len, &
    grid_path_len, &
    numerics_path_len

  character(len=material_path_len) :: &
    material_path
  character(len=load_path_len) :: &
    load_path
  character(len=grid_path_len) :: &
    grid_path
  character(len=numerics_path_len) :: &
    numerics_path

  call parallelization_init()
  material_path = c_str_to_f_str(material_path_c, material_path_len)
  load_path = c_str_to_f_str(load_path_c, load_path_len)
  grid_path = c_str_to_f_str(grid_path_c, grid_path_len)

  if (numerics_path_len>0) then
    numerics_path = c_str_to_f_str(numerics_path_c, numerics_path_len)
   call CLI_init(material_path, load_path, grid_path, numerics_path)                                                                     ! grid and mesh commandline interface
  else
    call CLI_init(material_path, load_path, grid_path)                                                                                   ! grid and mesh commandline interface
  end if
  call signal_init()
  call prec_init()
  call misc_init()
  call IO_init()
#if   defined(MESH)
  call FEM_quadrature_init()
#elif defined(GRID)
   call base64_init()
#endif
  call YAML_types_init()
  call YAML_parse_init()
  call HDF5_utilities_init()

end subroutine materialpoint_initBase_c

subroutine materialpoint_initBase()

  call parallelization_init()
  call CLI_init()                                                                                   ! grid and mesh commandline interface
  call signal_init()
  call prec_init()
  call misc_init()
  call IO_init()
#if   defined(MESH)
  call FEM_quadrature_init()
#elif defined(GRID)
   call base64_init()
#endif
  call YAML_types_init()
  call YAML_parse_init()
  call HDF5_utilities_init()

end subroutine materialpoint_initBase

subroutine materialpoint_initDamask() bind(C, name="f_materialpoint_initDamask")

  call result_init(restart=CLI_restartInc>0)
  call config_init()
  call math_init()
  call rotations_init()
  call polynomials_init()
  call tables_init()
  call crystal_init()
#if   defined(MESH)
  call discretization_mesh_init(restart=CLI_restartInc>0)
#endif
  call material_init(restart=CLI_restartInc>0)
  call phase_init()
  call homogenization_init()
  call materialpoint_init()
  call config_material_deallocate()

end subroutine materialpoint_initDamask



!--------------------------------------------------------------------------------------------------
!> @brief Read restart information if needed.
!--------------------------------------------------------------------------------------------------
subroutine materialpoint_init()

  integer(HID_T) :: fileHandle


  print'(/,1x,a)', '<<<+-  materialpoint init  -+>>>'; flush(IO_STDOUT)


  if (CLI_restartInc > 0) then
    print'(/,1x,a,1x,i0)', 'loading restart information of increment',CLI_restartInc; flush(IO_STDOUT)

    fileHandle = HDF5_openFile(getSolverJobName()//'_restart.hdf5','r')

    call homogenization_restartRead(fileHandle)
    call phase_restartRead(fileHandle)

    call HDF5_closeFile(fileHandle)
  end if

end subroutine materialpoint_init


!--------------------------------------------------------------------------------------------------
!> @brief Write restart information.
!--------------------------------------------------------------------------------------------------
subroutine materialpoint_restartWrite()

  integer(HID_T) :: fileHandle


  print'(1x,a)', 'saving field and constitutive data required for restart';flush(IO_STDOUT)

  fileHandle = HDF5_openFile(getSolverJobName()//'_restart.hdf5','a')

  call homogenization_restartWrite(fileHandle)
  call phase_restartWrite(fileHandle)

  call HDF5_closeFile(fileHandle)

end subroutine materialpoint_restartWrite


!--------------------------------------------------------------------------------------------------
!> @brief Forward data for new time increment.
!--------------------------------------------------------------------------------------------------
subroutine materialpoint_forward() bind(C, name="f_materialpoint_forward")

  ! print *, "mfhomogenization_F1", homogenization_F
  call homogenization_forward()
  call phase_forward()
  ! print *, "mfhomogenization_F2", homogenization_F

end subroutine materialpoint_forward


!--------------------------------------------------------------------------------------------------
!> @brief Trigger writing of results.
!--------------------------------------------------------------------------------------------------
subroutine materialpoint_result(inc,time) bind(C, name="f_materialpoint_result")

  integer,     intent(in) :: inc
  real(pREAL), intent(in) :: time

  call result_openJobFile()
  call result_addIncrement(inc,time)
  call phase_result()
  call homogenization_result()
  call discretization_result()
  call result_finalizeIncrement()
  call result_closeJobFile()

end subroutine materialpoint_result

end module materialpoint
