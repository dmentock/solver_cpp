#ifndef MECH_BASIC_H
#define MECH_BASIC_H
#include <string>

#include "mech_base.h"

using namespace std;
using namespace Eigen;

class MechSolverBasic : public MechBase {
public:
  MechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : MechBase(config_, grid_, spectral_) {}
  void init() override;
  static PetscErrorCode form_residual(DMDALocalInfo* residual_subdomain, double*** F_ptr, double*** r_ptr, void *ctx);
  static PetscErrorCode converged(SNES snes_local, 
                                  PetscInt PETScIter, 
                                  PetscReal devNull1, 
                                  PetscReal devNull2, 
                                  PetscReal fnorm, 
                                  SNESConvergedReason *reason, 
                                  void *ctx);
  Config::SolutionState calculate_solution(std::string& inc_info_) override;
  void forward (bool cutBack, bool guess, double delta_t, double delta_t_old, double t_remaining,
                Config::BoundaryCondition& deformation_BC, 
                Config::BoundaryCondition& stress_BC, 
                Quaterniond& rotation_BC) override;
  void update_coords() override;
};
#endif // MECH_BASIC_H
