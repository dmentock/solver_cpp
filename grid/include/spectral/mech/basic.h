#ifndef MECH_BASIC_H
#define MECH_BASIC_H

#include "spectral/mech/utilities.h"
#include <petsc.h>

class MechBasic : public MechUtilities {
public:
  MechBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : MechUtilities(config_, grid_, spectral_) {}
  void init();
  static PetscErrorCode formResidual (DMDALocalInfo* residual_subdomain,
                                      void* F,
                                      void* r,
                                      void* dummy);

  Eigen::Tensor<double, 2> P_av;
  Eigen::Tensor<double, 4> C_volAvg;
  Eigen::Tensor<double, 4> C_volAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvg;
  Eigen::Tensor<double, 4> C_minMaxAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvgRestart;

  Eigen::Tensor<double, 5> F_lastInc;
  Eigen::Tensor<double, 5> F_dot;

  Eigen::Tensor<double, 3> homogenization_F0;
};
#endif // MECH_BASIC_H