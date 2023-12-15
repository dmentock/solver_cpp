module fortran_testmodule
  use iso_c_binding
  use, intrinsic :: IEEE_arithmetic

  implicit none

  double precision, pointer :: my_global_ndarray(:,:,:)
  logical, target :: my_global_boolean = .false.
  double precision, dimension(:,:,:), target, allocatable :: tensormap_array_

  integer,     parameter :: pREAL      = IEEE_selected_real_kind(15,307)
  real(pReal),   dimension(:,:,:),   allocatable,   target, public :: &
    tensor
contains

subroutine datatypes_test(int_num,&
                            double_num,&
                            arr1d,&
                            arr2d,&
                            arr3d) bind(C, name="f_datatypes_test")
  use :: ISO_C_BINDING
  integer :: a, b, c
  integer(c_int) :: &
    int_num
  real(c_double) :: &
    double_num
  integer(c_int), dimension(3) :: &
    arr1d
  integer(c_int), dimension(3,3) :: &
    arr2d
  integer(c_int), dimension(3,3,3) :: &
    arr3d

  int_num = 1
  double_num = 2.3

  do a = 1, 3;
    arr1d(a) = a
  end do;

  do a = 1, 3;
    do b = 1, 3
      arr2d(a,b) = b+3*(a-1)
    end do;
  end do;

  do a = 1, 3;
    do b = 1, 3
      do c = 1, 3
        arr3d(a,b,c) = c+3*(b-1)+3*3*(a-1)
      end do;
    end do;
  end do;

end subroutine datatypes_test

subroutine interface_test_func(output_val) bind(C, name="f_interface_test_func")
  integer, intent(out) :: output_val
  output_val = 123
end subroutine interface_test_func

subroutine link_global_tensor(c_pointer, dims, n_dims) bind(C, name="f_link_global_tensor")
  type(c_ptr), intent(in), value :: c_pointer
  integer(c_int), intent(in) :: dims(*)
  integer(c_int), intent(in) :: n_dims
  integer, allocatable :: array_shape(:)
  integer :: i

  allocate(array_shape(n_dims))
  do i = 1, n_dims
    array_shape(i) = dims(i)
  end do
  call c_f_pointer(c_pointer, my_global_ndarray, array_shape)
  if (my_global_ndarray(1,1,1) /= 1) then
    print *, "wrong array passed to fortran:", my_global_ndarray
    stop 0
  end if
  my_global_ndarray(1,1,2) = 2
end subroutine link_global_tensor

subroutine link_global_boolean(c_bool) bind(C, name="f_link_global_boolean")
  type(c_ptr), intent(out) :: c_bool

  c_bool = c_loc(my_global_boolean)
  my_global_boolean = .true.
end subroutine link_global_boolean

subroutine verify_bool_modification() bind(C, name="f_verify_bool_modification")
  if (my_global_boolean .neqv. .false.) then
    print *, "c++ boolean modification failed"
    stop 0
  end if
  my_global_boolean = .true.
end subroutine verify_bool_modification

subroutine allocate_tensormap_array() bind(C, name="allocate_tensormap_array")
  allocate(tensormap_array_(3,3,3))
  tensormap_array_ = 0
  tensormap_array_(1,1,1) = 1
end subroutine allocate_tensormap_array

subroutine get_tensormap_array_ptr(tensormap_array) bind(C, name="get_tensormap_array_ptr")
  type(c_ptr), intent(out) :: tensormap_array

  allocate(tensormap_array_(3,3,3))
  tensormap_array = c_loc(tensormap_array_)

end subroutine get_tensormap_array_ptr

  subroutine fetch_tensor_pointers(tensor_ptr) &
    bind(C, name="f_fetch_tensor_pointers")

    type(c_ptr), intent(out) :: &
      tensor_ptr

    allocate(tensor(3,3,3))
    tensor_ptr = c_loc(tensor)

  end subroutine fetch_tensor_pointers

subroutine return_string(c_string, string_length) bind(C, name="f_return_string")
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
  print *, "Fortran String: ", f_string
end subroutine return_string

end module fortran_testmodule
