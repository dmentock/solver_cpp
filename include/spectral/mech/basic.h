#ifndef BASIC_H
#define BASIC_H

// #include <spectral/spectral.h>
#include <discretization_grid.h>
#include <spectral/mech/mech.h>
#include <unsupported/Eigen/CXX11/Tensor>

class Basic : public Mechanical {
public:
    Basic(DiscretizationGrid& grid_) : Mechanical(grid_) {}
    Eigen::Tensor<double, 2> P_av;
    Eigen::Tensor<double, 4> C_volAvg;
    Eigen::Tensor<double, 4> C_volAvgLastInc;
    Eigen::Tensor<double, 4> C_minMaxAvg;
    Eigen::Tensor<double, 4> C_minMaxAvgLastInc;
    Eigen::Tensor<double, 4> C_minMaxAvgRestart;

    Eigen::Tensor<double, 5> F_lastInc;
    Eigen::Tensor<double, 5> F_dot;

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