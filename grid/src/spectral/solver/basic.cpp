#include <petscsnes.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <mpi.h>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <helper.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "spectral/spectral.h"
#include "spectral/solver/basic.h"

void SolverBasic::init() {
  std::cout << "\n <<<+-  grid_mechanical_spectral_basic init  -+>>>\n";
  std::cout << "P. Eisenlohr et al., International Journal of Plasticity 46:37–53, 2013\n";
  std::cout << "https://doi.org/10.1016/j.ijplas.2012.09.012\n\n";

  std::cout << "P. Shanthraj et al., International Journal of Plasticity 66:31–45, 2015\n";
  std::cout << "https://doi.org/10.1016/j.ijplas.2014.02.006\n";

  PetscOptionsInsertString(NULL, "-mechanical_snes_type ngmres");
  PetscOptionsInsertString(NULL, config.num_grid.petsc_options.c_str());

  SNES SNES_mechanical;
  SNESCreate(PETSC_COMM_WORLD, &SNES_mechanical);
  SNESSetOptionsPrefix(SNES_mechanical, "mechanical_");

  std::vector<int> localK(grid.world_size);
  localK[grid.world_rank] = grid.cells2;

  MPI_Allreduce(MPI_IN_PLACE, localK.data(), grid.world_size, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
  DM da;
  int lx = grid.cells[0];
  int ly = grid.cells[1];
  DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                grid.cells[0], grid.cells[1], grid.cells[2],
                1, 1, grid.world_size,
                9, 0,
                &lx, &ly, localK.data(), &da);

  DMSetFromOptions(da);
  DMSetUp(da);
  Vec solution_vec;
  DMCreateGlobalVector(da, &solution_vec);
  DMDASNESSetFunctionLocal(da, INSERT_VALUES, formResidual, PETSC_NULLPTR);
  SNESSetConvergenceTest(SNES_mechanical, converged, PETSC_NULL, NULL);
  SNESSetDM(SNES_mechanical, da);
  SNESSetFromOptions(SNES_mechanical);

  // if CLI_restartInc == 0 ... 216-242
  F_lastInc.resize(Eigen::array<Eigen::Index, 5>{3, 3, grid.cells[0], grid.cells[1], grid.cells[2]});
  F_dot.resize(Eigen::array<Eigen::Index, 5>{3, 3, grid.cells[0], grid.cells[1], grid.cells[2]});

  // Initialize the tensor with the 3x3 identity matrix repeated across the last three dimensions
  Eigen::Matrix3d math_I3 = Eigen::Matrix3d::Identity();
  for (int i = 0; i < 3; ++i)
  {
    for (int j = 0; j < 3; ++j) {
      F_lastInc.slice(Eigen::array<Eigen::Index, 5>({i, j, 0, 0, 0}),
                      Eigen::array<Eigen::Index, 5>({1, 1, grid.cells[0], grid.cells[1], grid.cells[2]}))
        .setConstant(math_I3(i, j));
    }
  }

  Eigen::Tensor<double, 4> F(9, grid.cells[0], grid.cells[1], grid.cells[2]);
  F = F_lastInc.reshape(Eigen::array<int, 4>({9, grid.cells[0], grid.cells[1], grid.cells[2]}));

  PetscScalar *F_data = F.data();

  DMDAVecGetArray(da, solution_vec, &F_data); //moved down because its recommended to do this on already initialized arrays

  Eigen::Tensor<double, 3> homogenization_F0(3, 3, grid.cells[0] * grid.cells[1] * grid.cells[2]);
  homogenization_F0 = F_lastInc.reshape(Eigen::array<int, 3>({3, 3, grid.cells[0] * grid.cells[1] * grid.cells[2]}));

  Eigen::Tensor<double, 2> x_n(3, 12);
  Eigen::Tensor<double, 2> x_p(3, 2);
  update_coords(F_lastInc, x_n, x_p); // fix after implementing restart

  Eigen::Tensor<double, 5> P(3, 3, grid.cells[0], grid.cells[1], grid.cells[2]);
  P_av.resize(3, 3);
  C_volAvg.resize(3, 3, 3, 3);
  C_minMaxAvg.resize(3, 3, 3, 3);
  // constitutive_response(P,
  //                       P_av,
  //                       C_volAvg,
  //                       C_minMaxAvg, // stress field, stress avg, global average of stiffness and (min+max)/2
  //                       F_lastInc,          // target F
  //                       0);
  DMDAVecRestoreArray(da, solution_vec, &F_data);
  // add restart calls 253-268
  update_gamma(C_minMaxAvg);
  C_minMaxAvgRestart = C_minMaxAvg;
}

PetscErrorCode SolverBasic::converged (SNES snes_local,
                          PetscInt PETScIter,
                          PetscReal devNull1,
                          PetscReal devNull2,
                          PetscReal devNull3,
                          SNESConvergedReason* reason,
                          void* ctx) { 
    return 0;
    // double divTol = std::max(std::max(std::abs(P_av)) * num.eps_div_rtol, num.eps_div_atol);
    // double BCTol = std::max(std::max(std::abs(P_av)) * num.eps_stress_rtol, num.eps_stress_atol);

    // if ((totalIter >= num.itmin && (err_div / divTol < 1.0 && err_BC / BCTol < 1.0)) || terminallyIll) {
    //     reason = 1;
    // } else if (totalIter >= num.itmax) {
    //     reason = -1;
    // } else {
    //     reason = 0;
    // }

    // std::cout << std::endl << " ... reporting ............................................................." << std::endl;
    // std::cout << std::endl
    //           << " error divergence = " << std::fixed << std::setprecision(2) << err_div / divTol
    //           << " (" << std::scientific << std::setprecision(2) << err_div << " / m, tol = " << std::scientific << std::setprecision(2) << divTol << ")"
    //           << std::endl;
    // std::cout << " error stress BC  = " << std::fixed << std::setprecision(2) << err_BC / BCTol
    //           << " (" << std::scientific << std::setprecision(2) << err_BC << " Pa,  tol = " << std::scientific << std::setprecision(2) << BCTol << ")"
    //           << std::endl;
    // std::cout << std::endl << "===========================================================================" << std::endl;
    // std::cout.flush();
    // err_PETSc = 0;
}
PetscErrorCode SolverBasic::formResidual (DMDALocalInfo *residual_subdomain,
                                        void *F_void,
                                        void *r_void,
                                        void *dummy)
{
    Vec *F = static_cast<Vec *>(F_void);
    Vec *r = static_cast<Vec *>(r_void);
    return 0;
    // std::vector<std::vector<double>> deltaF_aim(3, std::vector<double>(3));
    // PetscInt PETScIter, nfuncs;
    // int err_MPI;

    // SNESGetNumberFunctionEvals(SNES_mechanical, &nfuncs, &err_PETSc);
    // CHKERRQ(err_PETSc);
    // SNESGetIterationNumber(SNES_mechanical, &PETScIter, &err_PETSc);
    // CHKERRQ(err_PETSc);

    // if (nfuncs == 0 && PETScIter == 0) {
    //     totalIter = -1; // new increment
    // }

    // if (totalIter <= PETScIter) {
    //     totalIter += 1;
    //     // Print statements and other code from the original function
    //     // ...
    // }

    // // Call your other functions with appropriate parameters
    // utilities_constitutiveResponse(/*...*/);
    // utilities_divergenceRMS(/*...*/);
    // // ...

    // err_MPI = MPI_Allreduce(MPI_IN_PLACE, terminallyIll, 1, MPI_INTEGER_KIND, MPI_LOGICAL, MPI_LOR, MPI_COMM_WORLD);
    // if (err_MPI != MPI_SUCCESS) {
    //     throw std::runtime_error("MPI error");
    // }

    // math_mul3333xx33(/*...*/);

    // // Update F_aim, err_BC, and r
    // // ...
}
