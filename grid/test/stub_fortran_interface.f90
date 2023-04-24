module fortran_testmodule

contains

subroutine fortran_datatypes_test(int_num,&
                            double_num,&
                            arr1d,&
                            arr2d,&
                            arr3d)
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

end subroutine fortran_datatypes_test

subroutine fortran_interface_test_func(output_val)
  integer, intent(out) :: output_val
  output_val = 123
end subroutine fortran_interface_test_func

end module fortran_testmodule