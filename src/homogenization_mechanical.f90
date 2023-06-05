!--------------------------------------------------------------------------------------------------
!> @author Martin Diehl, KU Leuven
!> @brief Partition F and homogenize P/dPdF
!--------------------------------------------------------------------------------------------------
submodule(homogenization) mechanical
  use prec
  use discretization

contains


module subroutine mechanical_init()

  print'(/,1x,a)', '<<<+-  homogenization:mechanical init  -+>>>'

  ! call parseMechanical()

  allocate(homogenization_dPdF(3,3,3,3,discretization_Ncells), source=0.0_pReal)
  print *, "discretization_Ncells", discretization_Ncells
  homogenization_F0 = spread(math_I3      ,         3,discretization_Ncells)
  homogenization_F = homogenization_F0
  allocate(homogenization_P(3,3,discretization_Ncells),source=0.0_pReal)

end subroutine mechanical_init

end submodule mechanical
