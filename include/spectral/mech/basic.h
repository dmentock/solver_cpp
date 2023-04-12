#ifndef BASIC_H
#define BASIC_H

// #include <spectral/spectral.h>
#include <discretization_grid.h>
#include <spectral/mech/mech.h>

class Basic : public Mechanical {
public:
    Basic(DiscretizationGrid& grid_) : Mechanical(grid_) {}
    void init();
    static PetscErrorCode formResidual(DMDALocalInfo* residual_subdomain,
                      void* F,
                      void* r,
                      void* dummy);
    static PetscErrorCode converged(SNES snes_local,
                   PetscInt PETScIter,
                   PetscReal  devNull1,
                   PetscReal  devNull2,
                   PetscReal  devNull3,
                   SNESConvergedReason *reason, 
                   void* ctx);
};
#endif // BASIC_H