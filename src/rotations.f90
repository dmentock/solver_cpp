module rotations
  use iso_c_binding, only: c_double, c_int, c_ptr
  use math

  implicit none(type,external)

  
  private


  real(pReal), parameter :: P = -1.0_pReal                                                          !< parameter for orientation conversion.

  type, public :: tRotation
    real(pReal), dimension(4) :: q
    contains
      procedure, public :: asMatrix
      procedure, private :: rotRot__
      generic,   public  :: rotate => rotVector,rotTensor2,rotTensor4
      procedure, public  :: rotVector
      procedure, public  :: rotTensor2
      procedure, public  :: rotTensor4
      procedure, public  :: standardize
  end type tRotation

  ! interface
  !     subroutine c_asMatrix(rot, result) bind(C, name="c_asMatrix")
  !         import tRotation
  !         type(tRotation), intent(in) :: rot
  !         real(c_double), intent(out), dimension(3,3) :: result
  !     end subroutine c_asMatrix
  ! end interface

  real(pReal), parameter :: &
    PREF = sqrt(6.0_pReal/PI), &
    A    = PI**(5.0_pReal/6.0_pReal)/6.0_pReal**(1.0_pReal/6.0_pReal), &
    AP   = PI**(2.0_pReal/3.0_pReal), &
    SC   = A/AP, &
    BETA = A/2.0_pReal, &
    R1   = (3.0_pReal*PI/4.0_pReal)**(1.0_pReal/3.0_pReal), &
    R2   = sqrt(2.0_pReal), &
    PI12 = PI/12.0_pReal, &
    PREK = R1 * 2.0_pReal**(1.0_pReal/4.0_pReal)/BETA

contains

subroutine rotate_tensor4(qu, T, rotated) bind(C, name="f_rotate_tensor4")
  real(c_double), intent(in), dimension(4) :: qu
  real(c_double),     intent(in),  dimension(3,3,3,3) :: T
  real(c_double),     intent(out), dimension(3,3,3,3) :: rotated
  type(tRotation) :: rot
  rot%q = qu
  rotated = rot%rotTensor4(T)
end subroutine rotate_tensor4

subroutine rotate_tensor2(qu, T, rotated) bind(C, name="f_rotate_tensor2")
  real(c_double), intent(in), dimension(4) :: qu
  real(c_double),     intent(in),  dimension(3,3) :: T
  real(c_double),     intent(out), dimension(3,3) :: rotated
  type(tRotation) :: rot
  rot%q = qu
  rotated = rot%rotTensor2(T)
end subroutine rotate_tensor2

pure function asMatrix(self)

  class(tRotation), intent(in) :: self
  real(pReal), dimension(3,3)  :: asMatrix


  asMatrix = qu2om(self%q)

end function asMatrix

subroutine qu2om_(qu, om) bind(C, name="f_qu2om")
  use, intrinsic :: iso_c_binding
  real(c_double), intent(in), dimension(4) :: qu
  real(c_double), intent(out), dimension(3,3) :: om

  om = qu2om(qu)
end subroutine qu2om_


pure function qu2om(qu) result(om)

  real(pReal), intent(in), dimension(4)   :: qu
  real(pReal),             dimension(3,3) :: om

  real(pReal)                             :: qq


  qq = qu(1)**2-sum(qu(2:4)**2)

  om(1,1) = qq+2.0_pReal*qu(2)**2
  om(2,2) = qq+2.0_pReal*qu(3)**2
  om(3,3) = qq+2.0_pReal*qu(4)**2

  om(1,2) = 2.0_pReal*(qu(2)*qu(3)-qu(1)*qu(4))
  om(2,3) = 2.0_pReal*(qu(3)*qu(4)-qu(1)*qu(2))
  om(3,1) = 2.0_pReal*(qu(4)*qu(2)-qu(1)*qu(3))
  om(2,1) = 2.0_pReal*(qu(3)*qu(2)+qu(1)*qu(4))
  om(3,2) = 2.0_pReal*(qu(4)*qu(3)+qu(1)*qu(2))
  om(1,3) = 2.0_pReal*(qu(2)*qu(4)+qu(1)*qu(3))

  if (sign(1.0_pReal,P) < 0.0_pReal) om = transpose(om)
  om = om/math_det33(om)**(1.0_pReal/3.0_pReal)

end function qu2om

! subroutine c_asMatrix(rot, result)
!   type(tRotation), intent(in) :: rot
!   real(c_double), intent(out), dimension(3,3) :: result
!   real(pReal), dimension(3,3) :: asMatrix_result
!   asMatrix_result = asMatrix(rot)
!   result = asMatrix_result
! end subroutine c_asMatrix

pure elemental function rotRot__(self,R) result(rRot)

  type(tRotation)              :: rRot
  class(tRotation), intent(in) :: self,R

  rRot = tRotation(multiplyQuaternion(self%q,R%q))
  call rRot%standardize()
end function rotRot__

pure elemental subroutine standardize(self)
  class(tRotation), intent(inout) :: self
  
  if (sign(1.0_pReal,self%q(1)) < 0.0_pReal) self%q = - self%q
end subroutine standardize

pure function conjugateQuaternion(qu)

  real(pReal), dimension(4), intent(in) :: qu
  real(pReal), dimension(4) :: conjugateQuaternion


  conjugateQuaternion = [qu(1), -qu(2), -qu(3), -qu(4)]

end function conjugateQuaternion

pure function multiplyQuaternion(qu1,qu2)

  real(pReal), dimension(4), intent(in) :: qu1, qu2
  real(pReal), dimension(4) :: multiplyQuaternion

  multiplyQuaternion(1) = qu1(1)*qu2(1) - qu1(2)*qu2(2) -      qu1(3)*qu2(3) - qu1(4)*qu2(4)
  multiplyQuaternion(2) = qu1(1)*qu2(2) + qu1(2)*qu2(1) + P * (qu1(3)*qu2(4) - qu1(4)*qu2(3))
  multiplyQuaternion(3) = qu1(1)*qu2(3) + qu1(3)*qu2(1) + P * (qu1(4)*qu2(2) - qu1(2)*qu2(4))
  multiplyQuaternion(4) = qu1(1)*qu2(4) + qu1(4)*qu2(1) + P * (qu1(2)*qu2(3) - qu1(3)*qu2(2))
end function multiplyQuaternion

pure function rotVector(self,v,active) result(vRot)

  real(pReal),                 dimension(3) :: vRot
  class(tRotation), intent(in)              :: self
  real(pReal),     intent(in), dimension(3) :: v
  logical,         intent(in), optional     :: active

  real(pReal), dimension(4) :: v_normed, q


  if (dEq0(norm2(v))) then
    vRot = v
  else
    v_normed = [0.0_pReal,v]/norm2(v)
    q = merge(multiplyQuaternion(conjugateQuaternion(self%q), multiplyQuaternion(v_normed, self%q)), &
              multiplyQuaternion(self%q, multiplyQuaternion(v_normed, conjugateQuaternion(self%q))), &
              misc_optional(active,.false.))
    vRot = q(2:4)*norm2(v)
  end if

end function rotVector


!--------------------------------------------------------------------------------------------------
!> @author Marc De Graef, Carnegie Mellon University
!> @brief Rotate a rank-2 tensor passively (default) or actively.
!> @details: Rotation is based on rotation matrix
!--------------------------------------------------------------------------------------------------
pure function rotTensor2(self,T,active) result(tRot)

  real(pReal),                 dimension(3,3) :: tRot
  class(tRotation), intent(in)                :: self
  real(pReal),     intent(in), dimension(3,3) :: T
  logical,         intent(in), optional       :: active


  tRot = merge(matmul(matmul(transpose(self%asMatrix()),T),self%asMatrix()), &

               matmul(matmul(self%asMatrix(),T),transpose(self%asMatrix())), &
               misc_optional(active,.false.))

end function rotTensor2


!--------------------------------------------------------------------------------------------------
!> @brief Rotate a rank-4 tensor passively (default) or actively.
!> @details: rotation is based on rotation matrix
!! ToDo: Need to check active/passive !!!
!--------------------------------------------------------------------------------------------------
pure function rotTensor4(self,T,active) result(tRot)

  real(pReal),                 dimension(3,3,3,3) :: tRot
  class(tRotation), intent(in)                    :: self
  real(pReal),     intent(in), dimension(3,3,3,3) :: T
  logical,         intent(in), optional           :: active

  real(pReal), dimension(3,3) :: R
  integer :: i,j,k,l,m,n,o,p

  R = merge(transpose(self%asMatrix()),self%asMatrix(),misc_optional(active,.false.))

  tRot = 0.0_pReal
  do i = 1,3;do j = 1,3;do k = 1,3;do l = 1,3
  do m = 1,3;do n = 1,3;do o = 1,3;do p = 1,3
    tRot(i,j,k,l) = tRot(i,j,k,l) &
                  + R(i,m) * R(j,n) * R(k,o) * R(l,p) * T(m,n,o,p)
  end do; end do; end do; end do; end do; end do; end do; end do

end function rotTensor4



end module rotations
