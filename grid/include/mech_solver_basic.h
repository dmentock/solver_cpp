#ifndef MECH_BASIC_H
#define MECH_BASIC_H

#include <mech_base.h>
#include <string>

class MechSolverBasic : public MechBase {
public:
  MechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : MechBase(config_, grid_, spectral_) {}
  void init();
  static PetscErrorCode formResidual(DMDALocalInfo* residual_subdomain, void* F_raw, void* r_raw, void *ctx);
};
#endif // MECH_BASIC_H