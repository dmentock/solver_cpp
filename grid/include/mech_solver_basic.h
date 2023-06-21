#ifndef MECH_BASIC_H
#define MECH_BASIC_H

#include <mech_base.h>
#include <petsc.h>
#include <petscsnes.h>
#include <string>

class MechSolverBasic : public MechBase {
public:
  MechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : MechBase(config_, grid_, spectral_) {}
  void init();
  static PetscErrorCode formResidual(DMDALocalInfo* residual_subdomain, void* F_raw, void* r_raw, void *ctx);

  Eigen::Tensor<double, 2> P_av;
  Eigen::Tensor<double, 4> C_volAvg;
  Eigen::Tensor<double, 4> C_volAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvg;
  Eigen::Tensor<double, 4> C_minMaxAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvgRestart;

  Eigen::Tensor<double, 5> F_lastInc;
  Eigen::Tensor<double, 5> F_dot;

  Eigen::Tensor<double, 3> homogenization_F0;

  // Eigen::Matrix<double, 3, 3> F_aimDot = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, 3, 3> F_aim = Eigen::Matrix<double, 3, 3>::Identity();
  // Eigen::Matrix<double, 3, 3> F_aim_lastInc = Eigen::Matrix<double, 3, 3>::Identity();
  // Eigen::Matrix<double, 3, 3> P_av = Eigen::Matrix<double, 3, 3>::Zero();
  // Eigen::Matrix<double, 3, 3> P_aim = Eigen::Matrix<double, 3, 3>::Zero();

  // // 4D tensor of real numbers
  // Eigen::Tensor<double, 4> C_volAvg = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> C_volAvgLastInc = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> C_minMaxAvg = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> C_minMaxAvgLastInc = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> C_minMaxAvgRestart = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> S = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);

  // // Scalar real numbers
  // double err_BC;
  // double err_div;

  SolutionParams params;

  SNES SNES_mechanical;
  int total_iter = 0;
  std::string inc_info;

};
#endif // MECH_BASIC_H