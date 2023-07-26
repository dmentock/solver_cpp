module math
  use prec
  use LAPACK_interface
  use misc
  implicit none(type,external)
  public

  real(pReal), parameter :: &
    PI = acos(-1.0_pReal), &                                                                        !< ratio of a circle's circumference to its diameter
    TAU = 2.0_pReal*PI, &                                                                           !< ratio of a circle's circumference to its radius
    INDEG = 360.0_pReal/TAU, &                                                                      !< conversion from radian to degree
    INRAD = TAU/360.0_pReal   
    
  integer, dimension (2,9), parameter, private :: &
    MAPPLAIN = reshape([&
      1,1, &
      1,2, &
      1,3, &
      2,1, &
      2,2, &
      2,3, &
      3,1, &
      3,2, &
      3,3  &
      ],shape(MAPPLAIN))                                                                        !< conversion from degree to radian

  real(pReal), dimension(3,3), parameter :: &
    math_I3 = reshape([&
      1.0_pReal,0.0_pReal,0.0_pReal, &
      0.0_pReal,1.0_pReal,0.0_pReal, &
      0.0_pReal,0.0_pReal,1.0_pReal  &
      ],shape(math_I3))                                                                             !< 3x3 Identity

contains

pure subroutine math_invert(InvA, error, A, n) bind(C, name="f_math_invert")

  integer, intent(in)  :: n
  real(pReal), dimension(n,n), intent(in)  :: A
  real(pReal), dimension(n,n), intent(out) :: invA
  integer,                     intent(out) :: error

  integer,     dimension(n)    :: ipiv
  real(pReal), dimension(n*n)  :: work
  integer                      :: ierr

  invA = A
  call dgetrf(n, n, invA, n, ipiv, ierr)
  error = (ierr /= 0)
  call dgetri(n, invA, n, ipiv, work, n*n, ierr)
  error = error /= 0 .or. (ierr /= 0)

end subroutine math_invert

subroutine math_3333to99_(m3333, m99) bind(C, name="f_math_3333to99")

  real(pReal), dimension(3,3,3,3), intent(in)  :: m3333
  real(pReal), dimension(9,9)       :: m99

  m99 = math_3333to99(m3333)
end subroutine math_3333to99_

pure function math_3333to99(m3333)

  real(pReal), dimension(9,9)                 :: math_3333to99
  real(pReal), dimension(3,3,3,3), intent(in) :: m3333

  integer :: i,j


#ifndef __INTEL_COMPILER
  do concurrent(i=1:9, j=1:9)
    math_3333to99(i,j) = m3333(MAPPLAIN(1,i),MAPPLAIN(2,i),MAPPLAIN(1,j),MAPPLAIN(2,j))
  end do
#else
  forall(i=1:9, j=1:9) math_3333to99(i,j) = m3333(MAPPLAIN(1,i),MAPPLAIN(2,i),MAPPLAIN(1,j),MAPPLAIN(2,j))
#endif

end function math_3333to99

subroutine math_99to3333_(m99, m3333) bind(C, name="f_math_99to3333")

  real(pReal), dimension(3,3,3,3)   :: m3333
  real(pReal), dimension(9,9), intent(in)       :: m99

  m3333 = math_99to3333(m99)
end subroutine math_99to3333_
!--------------------------------------------------------------------------------------------------
!> @brief convert 9x9 matrix into 3x3x3x3 matrix
!--------------------------------------------------------------------------------------------------
pure function math_99to3333(m99)

  real(pReal), dimension(3,3,3,3)         :: math_99to3333
  real(pReal), dimension(9,9), intent(in) :: m99

  integer :: i,j


#ifndef __INTEL_COMPILER
  do concurrent(i=1:9, j=1:9)
    math_99to3333(MAPPLAIN(1,i),MAPPLAIN(2,i),MAPPLAIN(1,j),MAPPLAIN(2,j)) = m99(i,j)
  end do
#else
  forall(i=1:9, j=1:9) math_99to3333(MAPPLAIN(1,i),MAPPLAIN(2,i),MAPPLAIN(1,j),MAPPLAIN(2,j)) = m99(i,j)
#endif

end function math_99to3333

real(pReal) pure function math_det33(m)

  real(pReal), dimension(3,3), intent(in) :: m


  math_det33 = m(1,1)* (m(2,2)*m(3,3)-m(2,3)*m(3,2)) &
             - m(1,2)* (m(2,1)*m(3,3)-m(2,3)*m(3,1)) &
             + m(1,3)* (m(2,1)*m(3,2)-m(2,2)*m(3,1))

end function math_det33

subroutine math_mul3333xx33_(A, B, res) bind(C, name="f_math_mul3333xx33")

  real(pReal), dimension(3,3,3,3), intent(in) :: A
  real(pReal), dimension(3,3),     intent(in) :: B
  real(pReal), dimension(3,3) :: res

  res = math_mul3333xx33(A, B)
end subroutine math_mul3333xx33_

!--------------------------------------------------------------------------------------------------
!> @brief matrix double contraction 3333x33 = 33 (ijkl,kl)
!--------------------------------------------------------------------------------------------------
pure function math_mul3333xx33(A,B)

  real(pReal), dimension(3,3,3,3), intent(in) :: A
  real(pReal), dimension(3,3),     intent(in) :: B
  real(pReal), dimension(3,3) :: math_mul3333xx33

  integer :: i,j

#ifndef __INTEL_COMPILER
  do concurrent(i=1:3, j=1:3)
    math_mul3333xx33(i,j) = sum(A(i,j,1:3,1:3)*B(1:3,1:3))
  end do
#else
  forall (i=1:3, j=1:3) math_mul3333xx33(i,j) = sum(A(i,j,1:3,1:3)*B(1:3,1:3))
#endif

end function math_mul3333xx33


end module math
