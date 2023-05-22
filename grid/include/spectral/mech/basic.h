#ifndef MECH_BASIC_H
#define MECH_BASIC_H

#include "spectral/mech/utilities.h"

class MechBasic : public Utilities {
public:
  MechBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : Utilities(config_, grid_, spectral_) {}
  Eigen::Tensor<double, 2> P_av;
  Eigen::Tensor<double, 4> C_volAvg;
  Eigen::Tensor<double, 4> C_volAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvg;
  Eigen::Tensor<double, 4> C_minMaxAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvgRestart;

  Eigen::Tensor<double, 5> F_lastInc;
  Eigen::Tensor<double, 5> F_dot;

  void init();
  static PetscErrorCode formResidual (DMDALocalInfo* residual_subdomain,
                                      void* F,
                                      void* r,
                                      void* dummy);
  static PetscErrorCode converged(SNES snes_local,
                                  PetscInt PETScIter,
                                  PetscReal  devNull1,
                                  PetscReal  devNull2,
                                  PetscReal  devNull3,
                                  SNESConvergedReason* reason, 
                                  void* ctx);
};
#endif // MECH_BASIC_H