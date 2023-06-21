#include <petsc.h>
#include <petscsys.h>
#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

#include <vector>
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <spectral.h>
#include <mech_solver_basic.h>
#include <fortran_utilities.h>

#include <helper.h>

using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::Matrix;
using Eigen::MatrixXd;
void MechSolverBasic::init() {
  std::cout << "\n <<<+-  grid_mechanical_spectral_basic init  -+>>>\n";
  std::cout << "P. Eisenlohr et al., International Journal of Plasticity 46:37–53, 2013\n";
  std::cout << "https://doi.org/10.1016/j.ijplas.2012.09.012\n\n";

  std::cout << "P. Shanthraj et al., International Journal of Plasticity 66:31–45, 2015\n";
  std::cout << "https://doi.org/10.1016/j.ijplas.2014.02.006\n";

  PetscOptionsInsertString(NULL, "-mechanical_snes_type ngmres");
  PetscOptionsInsertString(NULL, config.num_grid.petsc_options.c_str());

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
  DMDASNESSetFunctionLocal(da, INSERT_VALUES, formResidual, this);
  // SNESSetConvergenceTest(SNES_mechanical, converged, PETSC_NULL, NULL);
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

  homogenization_F0.resize(3, 3, grid.cells[0] * grid.cells[1] * grid.cells[2]);
  homogenization_F0 = F_lastInc.reshape(Eigen::array<int, 3>({3, 3, grid.cells[0] * grid.cells[1] * grid.cells[2]}));

  Eigen::Tensor<double, 2> x_n(3, 12);
  Eigen::Tensor<double, 2> x_p(3, 2);
  update_coords(F_lastInc, x_n, x_p); // fix after implementing restart

  Eigen::Tensor<double, 5> P(3, 3, grid.cells[0], grid.cells[1], grid.cells[2]);
  Eigen::TensorMap<Eigen::Tensor<double, 5>> P_map(P.data(), 3, 3, grid.cells[0], grid.cells[1], grid.cells[2]);
  Eigen::TensorMap<Eigen::Tensor<double, 5>> F_map(F_lastInc.data(), 3, 3, grid.cells[0], grid.cells[1], grid.cells[2]);
  P_av.resize(3, 3);
  C_volAvg.resize(3, 3, 3, 3);
  C_minMaxAvg.resize(3, 3, 3, 3);
  spectral.constitutive_response (P_map,
                                  P_av,
                                  C_volAvg,
                                  C_minMaxAvg,
                                  F_map,
                                  0);
  DMDAVecRestoreArray(da, solution_vec, &F_data);
  // add restart calls 253-268
  update_gamma(C_minMaxAvg);
  C_minMaxAvgRestart = C_minMaxAvg;
}

PetscErrorCode MechSolverBasic::formResidual(DMDALocalInfo* residual_subdomain, void* F_raw, void* r_raw, void *ctx) {

  MechSolverBasic* mech_basic = static_cast<MechSolverBasic*>(ctx);
  double* F_ = static_cast<double*>(F_raw);
  double* r_ = static_cast<double*>(r_raw);
  Eigen::TensorMap<Eigen::Tensor<double, 5>> F(F_, 3, 3, mech_basic->grid.cells[0], mech_basic->grid.cells[1], mech_basic->grid.cells[2]);
  Eigen::TensorMap<Eigen::Tensor<double, 5>> r(r_, 3, 3, mech_basic->grid.cells[0], mech_basic->grid.cells[1], mech_basic->grid.cells[2]);
  
  int n_funcs, petsc_iter;
  SNESGetNumberFunctionEvals(mech_basic->SNES_mechanical, &n_funcs);
  SNESGetIterationNumber(mech_basic->SNES_mechanical, &petsc_iter);

  int total_iter = 0;
  if (n_funcs == 0 && petsc_iter == 0) {
      total_iter = -1; // new increment
  }

  if (total_iter <= petsc_iter) {
      total_iter += 1;
      // Print statements and other code from the original function
      // ...
  }

  if (total_iter <= petsc_iter) {
  total_iter += 1;
  std::cout << mech_basic->inc_info << " @ Iteration " << mech_basic->config.num_grid.itmin << " <= " << total_iter << " <= " << mech_basic->config.num_grid.itmax << "\n";
  if (std::any_of(mech_basic->params.rot_bc_q.coeffs().data(), mech_basic->params.rot_bc_q.coeffs().data() + 4, [](double i) {return i != 1.0;})) {
      Eigen::Tensor<double, 2> F_aim_rotated = FortranUtilities::rotate_tensor2(mech_basic->params.rot_bc_q, mech_basic->F_aim);
      Eigen::Map<const Eigen::Matrix<double, 3, 3>> F_aim_rotated_mat(F_aim_rotated.data());
      std::cout << "deformation gradient aim (lab) = " << F_aim_rotated_mat.transpose() << "\n";
  }
  std::cout << "deformation gradient aim = " << mech_basic->F_aim.transpose() << "\n";
  std::cout.flush();
  }

  print_f_map("P", r);
  mech_basic->spectral.constitutive_response(r, mech_basic->P_av, mech_basic->C_volAvg, mech_basic->C_minMaxAvg, F, mech_basic->params.delta_t, mech_basic->params.rot_bc_q);
  print_f_map("P", r);
  // int err_MPI;
  // MPI_Allreduce(MPI_IN_PLACE, &terminallyIll, 1, MPI_INTEGER, MPI_LOR, MPI_COMM_WORLD, &err_MPI);
  // if (err_MPI != MPI_SUCCESS) {
  //     std::cerr << "MPI error";
  //     exit(-1);  // or handle the error as you prefer
  // }
  // double err_div = utilities_divergenceRMS(P);
  
  return 0;
}

// PetscErrorCode MechSolverBasic::formResidual (DMDALocalInfo* residual_subdomain,
//                                         void *F_void,
//                                         void *r_void,
//                                         void *dummy)
// {
//     Vec *F = static_cast<Vec *>(F_void);
//     Vec *r = static_cast<Vec *>(r_void);
//     return 0;
//     // std::vector<std::vector<double>> deltaF_aim(3, std::vector<double>(3));
//     // PetscInt PETScIter, nfuncs;
//     // int err_MPI;

//     // SNESGetNumberFunctionEvals(SNES_mechanical, &nfuncs, &err_PETSc);
//     // CHKERRQ(err_PETSc);
//     // SNESGetIterationNumber(SNES_mechanical, &PETScIter, &err_PETSc);
//     // CHKERRQ(err_PETSc);

//     // if (nfuncs == 0 && PETScIter == 0) {
//     //     totalIter = -1; // new increment
//     // }

//     // if (totalIter <= PETScIter) {
//     //     totalIter += 1;
//     //     // Print statements and other code from the original function
//     //     // ...
//     // }

//     // // Call your other functions with appropriate parameters
//     // utilities_constitutiveResponse(/*...*/);
//     // utilities_divergenceRMS(/*...*/);
//     // // ...

//     // err_MPI = MPI_Allreduce(MPI_IN_PLACE, terminallyIll, 1, MPI_INTEGER_KIND, MPI_LOGICAL, MPI_LOR, MPI_COMM_WORLD);
//     // if (err_MPI != MPI_SUCCESS) {
//     //     throw std::runtime_error("MPI error");
//     // }

//     // math_mul3333xx33(/*...*/);

//     // // Update F_aim, err_BC, and r
//     // // ...
// }
