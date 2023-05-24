module fortran_testmodule
  use iso_c_binding
  implicit none
  double precision, pointer :: my_global_ndarray(:,:,:) ! Change according to your need

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

subroutine link_global_variable(c_pointer, dims, n_dims) bind(C, name="f_link_global_variable")
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
end subroutine link_global_variable

end module fortran_testmodule