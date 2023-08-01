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
#include <tensor_operations.h>

#include <helper.h>

void MechSolverBasic::init() {
  std::cout << "\n <<<+-  grid_mechanical_spectral_basic init  -+>>>\n";
  std::cout << "P. Eisenlohr et al., International Journal of Plasticity 46:37–53, 2013\n";
  std::cout << "https://doi.org/10.1016/j.ijplas.2012.09.012\n\n";

  std::cout << "P. Shanthraj et al., International Journal of Plasticity 66:31–45, 2015\n";
  std::cout << "https://doi.org/10.1016/j.ijplas.2014.02.006\n";

  base_init();

  PetscOptionsInsertString(NULL, "-mechanical_snes_type ngmres");
  PetscOptionsInsertString(NULL, config.numerics.petsc_options.c_str());

  ierr = SNESCreate(PETSC_COMM_WORLD, &SNES_mechanical);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESSetOptionsPrefix(SNES_mechanical, "mechanical_");
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  std::vector<int> localK(grid.world_size);
  localK[grid.world_rank] = grid.cells2;
  MPI_Allreduce(MPI_IN_PLACE, localK.data(), grid.world_size, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);

  ierr = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                      grid.cells[0], grid.cells[1], grid.cells[2],
                      1, 1, grid.world_size,
                      9, 0,
                      &grid.cells[0], &grid.cells[1], localK.data(), &da);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  ierr = DMSetFromOptions(da);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  ierr = DMSetUp(da);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = DMCreateGlobalVector(da, &solution_vec);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = DMDASNESSetFunctionLocal(da, INSERT_VALUES, formResidual, this);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESSetConvergenceTest(SNES_mechanical, converged, PETSC_NULL, NULL);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESSetDM(SNES_mechanical, da);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESSetFromOptions(SNES_mechanical);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  // if CLI_restartInc == 0 ... 216-242


  Matrix3d math_I3 = Matrix3d::Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      F_last_inc.slice(Eigen::array<Index, 5>({i, j, 0, 0, 0}),
                      Eigen::array<Index, 5>({1, 1, grid.cells[0], grid.cells[1], grid.cells2}))
        .setConstant(math_I3(i, j));
    }
  }

  double* F_raw;
  ierr = VecGetArray(solution_vec, &F_raw);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  TensorMap<Tensor<double, 5>> F(reinterpret_cast<double*>(F_raw),  3, 3, grid.cells[0], grid.cells[1], grid.cells2);
  F = F_last_inc;
  *spectral.homogenization_F0 = F_last_inc.reshape(Eigen::DSizes<Eigen::DenseIndex, 3>(3, 3, grid.cells[0]*grid.cells[1]*grid.cells2));
  Tensor<double, 2> x_n(3, 12);
  Tensor<double, 2> x_p(3, 2);

  base_update_coords(F, x_n, x_p); // fix after implementing restart

  Tensor<double, 5> P(3, 3, grid.cells[0], grid.cells[1], grid.cells[2]);
  TensorMap<Tensor<double, 5>> P_map(P.data(), 3, 3, grid.cells[0], grid.cells[1], grid.cells2);
  TensorMap<Tensor<double, 5>> F_map(F_last_inc.data(), 3, 3, grid.cells[0], grid.cells[1], grid.cells[2]);
  P_av.resize(3, 3);
  C_volAvg.resize(3, 3, 3, 3);
  C_minMaxAvg.resize(3, 3, 3, 3);
  spectral.constitutive_response (P_map,
                                  P_av,
                                  C_volAvg,
                                  C_minMaxAvg,
                                  F_map,
                                  0);
  ierr = VecRestoreArray(solution_vec, &F_raw);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  // add restart calls 253-268
  update_gamma(C_minMaxAvg);
  C_minMaxAvgRestart = C_minMaxAvg;
}

Spectral::SolutionState MechSolverBasic::calculate_solution(std::string inc_info_) {

  inc_info = inc_info_;

  Spectral::SolutionState solution;
  SNESConvergedReason reason;

  // update stiffness (and gamma operator)
  S = calculate_masked_compliance(C_volAvg, params.rot_bc_q, params.stress_mask);
  if (config.numerics.update_gamma) update_gamma(C_minMaxAvg);

  ierr = SNESSolve(SNES_mechanical, NULL, solution_vec);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESGetConvergedReason(SNES_mechanical, &reason);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  solution.converged = reason > 0;
  solution.iterationsNeeded = total_iter;
  solution.terminally_ill = *spectral.terminally_ill;
  *spectral.terminally_ill = false;
  P_aim = P_av;

  return solution;
}

void MechSolverBasic::forward(bool cutBack, bool guess, double delta_t, double delta_t_old, double t_remaining,
                              Config::BoundaryCondition& deformation_bc, 
                              Config::BoundaryCondition& stress_bc, 
                              Quaterniond& rot_bc_q) {
  PetscScalar *F_raw;
  ierr = VecGetArray(solution_vec, &F_raw); 
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  TensorMap<Tensor<double, 5>> F(reinterpret_cast<double*>(F_raw), 3, 3, grid.cells[0], grid.cells[1], grid.cells2);

  if (cutBack) {
    C_volAvg = C_volAvgLastInc;
    C_minMaxAvg = C_minMaxAvgLastInc;
  } else {
    C_volAvgLastInc = C_volAvg;
    C_minMaxAvgLastInc = C_minMaxAvg;

    F_aim_dot = (F_aim - F_aim_last_inc) / delta_t_old;
    F_aim_dot = stress_bc.mask.select(F_aim_dot, 0.0);
    if (!guess) F_aim_dot.setZero();
    F_aim_last_inc = F_aim;

    if (deformation_bc.type == "L") {
        F_aim_dot += deformation_bc.mask.select(deformation_bc.values * F_aim_last_inc, 0.0);
    } else if (deformation_bc.type == "dot_F") {
        F_aim_dot += deformation_bc.mask.select(deformation_bc.values, 0.0);
    } else if (deformation_bc.type == "F") {
        F_aim_dot += deformation_bc.mask.select((deformation_bc.values - F_aim_last_inc) / t_remaining, 0.0);
    }

    *spectral.homogenization_F0 = F.reshape(Eigen::DSizes<Eigen::DenseIndex, 3>(3, 3, grid.cells[0]*grid.cells[1]*grid.cells[2]));

    Tensor<double, 2> rotated = FortranUtilities::rotate_tensor2(rot_bc_q, F_aim_dot, true);
    F_dot = calculate_rate (guess, F_last_inc, 
                            spectral.homogenization_F0->reshape(Eigen::array<int, 5>{3, 3, grid.cells[0], grid.cells[1], grid.cells2}), 
                            delta_t_old, rotated);
    F_last_inc = spectral.homogenization_F0->reshape(Eigen::array<int, 5>{3, 3, grid.cells[0], grid.cells[1], grid.cells2});
  }
  F_aim = F_aim_last_inc + F_aim_dot * delta_t;
  Eigen::Tensor<double, 2> zero_tensor(P_aim.dimensions());
  zero_tensor.setZero();
  if (stress_bc.type == "P") {
    P_aim += mat_to_tensor(stress_bc.mask).select((mat_to_tensor(stress_bc.values) - P_aim) / t_remaining, zero_tensor) * delta_t;
  } else {
    P_aim += mat_to_tensor(stress_bc.mask).select(mat_to_tensor(stress_bc.values), zero_tensor) * delta_t;
  }
  Tensor<double, 2> rotated = FortranUtilities::rotate_tensor2(rot_bc_q, F_aim_dot);
  Matrix<double, 3, 3> rotated_mat = tensor_to_mat(rotated);
  F = forward_field(delta_t, F_last_inc, F_dot, &rotated_mat);

  ierr = VecRestoreArray(solution_vec, &F_raw);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  params.stress_mask = stress_bc.mask;
  params.rot_bc_q = rot_bc_q;
  params.delta_t = delta_t;
}

void MechSolverBasic::update_coords() {
  PetscScalar *F_raw;
  ierr = VecGetArray(solution_vec, &F_raw); 
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  TensorMap<Tensor<double, 5>> F(reinterpret_cast<double*>(F_raw), 3, 3, grid.cells[0], grid.cells[1], grid.cells2);

  Tensor<double, 2> x_n;
  Tensor<double, 2> x_p;
  base_update_coords(F, x_n, x_p);
  ierr = VecRestoreArray(solution_vec, &F_raw);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
}

PetscErrorCode MechSolverBasic::converged(SNES snes_local, 
                                          PetscInt PETScIter, 
                                          PetscReal devNull1, 
                                          PetscReal devNull2, 
                                          PetscReal fnorm, 
                                          SNESConvergedReason *reason, 
                                          void *ctx) {

  MechSolverBasic* mech_basic = static_cast<MechSolverBasic*>(ctx);

  double divTol = std::max(mech_basic->err_div * mech_basic->config.numerics.eps_div_rtol, mech_basic->config.numerics.eps_div_atol);
  double BCTol = std::max(mech_basic->err_BC * mech_basic->config.numerics.eps_stress_rtol, mech_basic->config.numerics.eps_stress_atol);

  if ((mech_basic->total_iter >= mech_basic->config.numerics.itmin && std::max(mech_basic->err_div / divTol, mech_basic->err_BC / BCTol) < 1.0) || mech_basic->spectral.terminally_ill) {
    *reason = SNES_CONVERGED_ITERATING;
  } else if (mech_basic->total_iter >= mech_basic->config.numerics.itmax) {
    *reason = SNES_DIVERGED_MAX_IT;
  } else {
    *reason = SNES_CONVERGED_ITERATING;
  }

  std::cout << "\n ... reporting ............................................................." << std::endl;
  std::cout << std::fixed << std::setprecision(2) << "error divergence = " << mech_basic->err_div / divTol << " (" << 
    std::scientific << mech_basic->err_div << " / m, tol = " << divTol << ")" << std::endl;
  std::cout << "error stress BC  = " << mech_basic->err_BC / BCTol << " (" << mech_basic->err_BC << " Pa,  tol = " << BCTol << ")" << std::endl;
  std::cout << "\n===========================================================================" << std::endl;

  return 0;
}

PetscErrorCode MechSolverBasic::formResidual(DMDALocalInfo* residual_subdomain, void* F_raw, void* r_raw, void *ctx) {

  MechSolverBasic* mech_basic = static_cast<MechSolverBasic*>(ctx);
  double* F_ = static_cast<double*>(F_raw);
  double* r_ = static_cast<double*>(r_raw);
  TensorMap<Tensor<double, 5>> F(F_, 3, 3, mech_basic->grid.cells[0], mech_basic->grid.cells[1], mech_basic->grid.cells[2]);
  TensorMap<Tensor<double, 5>> r(r_, 3, 3, mech_basic->grid.cells[0], mech_basic->grid.cells[1], mech_basic->grid.cells[2]);

  int n_funcs, petsc_iter;
  SNESGetNumberFunctionEvals(mech_basic->SNES_mechanical, &n_funcs);
  SNESGetIterationNumber(mech_basic->SNES_mechanical, &petsc_iter);

  int total_iter = 0;
  if (n_funcs == 0 && petsc_iter == 0) total_iter = -1;

  if (total_iter <= petsc_iter) {
    total_iter += 1;
    std::cout << mech_basic->inc_info << " @ Iteration " << mech_basic->config.numerics.itmin << " <= " << total_iter << " <= " << mech_basic->config.numerics.itmax << "\n";
    if (std::any_of(mech_basic->params.rot_bc_q.coeffs().data(), mech_basic->params.rot_bc_q.coeffs().data() + 4, [](double i) {return i != 1.0;})) {
      Tensor<double, 2> F_aim_rotated = FortranUtilities::rotate_tensor2(mech_basic->params.rot_bc_q, mech_basic->F_aim);
      Map<const Matrix<double, 3, 3>> F_aim_rotated_mat(F_aim_rotated.data());
      std::cout << "deformation gradient aim (lab) =\n" << F_aim_rotated_mat.transpose() << "\n";
    }
    std::cout << "deformation gradient aim =\n" << mech_basic->F_aim.transpose() << "\n";
    std::cout.flush();
  }

  mech_basic->spectral.constitutive_response(r, mech_basic->P_av, mech_basic->C_volAvg, mech_basic->C_minMaxAvg, F, mech_basic->params.delta_t, mech_basic->params.rot_bc_q);
  MPI_Allreduce(MPI_IN_PLACE, mech_basic->spectral.terminally_ill, 1, MPI_INTEGER, MPI_LOR, MPI_COMM_WORLD);
  mech_basic->err_div = mech_basic->calculate_divergence_rms(r);
  Matrix<double, 3, 3> delta_F_aim;
  f_math_mul3333xx33(mech_basic->S.data(), mech_basic->P_av.data(), delta_F_aim.data());
  mech_basic->F_aim = mech_basic->F_aim - delta_F_aim;
  Matrix<double, 3, 3> stress_mask_double = (mech_basic->params.stress_mask.unaryExpr([](bool v) { return !v; })).cast<double>();
  Tensor<double, 2> diff = (mech_basic->P_av - mech_basic->P_aim).abs() * 
    mat_to_tensor(stress_mask_double);
  Tensor<double, 0> err_BC_ = diff.maximum();
  mech_basic->err_BC = err_BC_(0);  
  Tensor<double, 2>  rotated = FortranUtilities::rotate_tensor2(mech_basic->params.rot_bc_q, delta_F_aim, true);
  mech_basic->gamma_convolution(r, rotated);
  return 0;
}
