#include <petsc.h>
#include <petscsys.h>
#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

#include <vector>
#include <iostream>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "spectral.h"
#include "mech_solver_basic.h"
#include "utilities_fortran.h"

#include "utilities_tensor.h"

void MechSolverBasic::init() {
  std::cout << "\n <<<+-  mech solver basic init  -+>>>\n";
  std::cout << " P. Eisenlohr et al., International Journal of Plasticity 46:37–53, 2013\n";
  std::cout << " https://doi.org/10.1016/j.ijplas.2012.09.012\n\n";

  std::cout << " P. Shanthraj et al., International Journal of Plasticity 66:31–45, 2015\n";
  std::cout << " https://doi.org/10.1016/j.ijplas.2014.02.006\n";

  base_init();

  PetscOptionsInsertString(NULL, "-mechanical_snes_type ngmres");
  // PetscOptionsInsertString(NULL, PETSC_NULL_OPTIONS); // not needed?
  PetscOptionsInsertString(NULL, config.numerics.fft.petsc_options.c_str());

  ierr = SNESCreate(PETSC_COMM_WORLD, &SNES_mechanical);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESSetOptionsPrefix(SNES_mechanical, "mechanical_");
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  std::vector<int> localK(grid_.world_size);
  localK[grid_.world_rank] = grid_.cells2;
  MPI_Allreduce(MPI_IN_PLACE, localK.data(), grid_.world_size, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);

  ierr = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                      grid_.cells[0], grid_.cells[1], grid_.cells[2],
                      1, 1, grid_.world_size,
                      9, 0,
                      &grid_.cells[0], &grid_.cells[1], localK.data(), &da);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  ierr = DMSetFromOptions(da);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  ierr = DMSetUp(da);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = DMCreateGlobalVector(da, &F_PETSc);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = DMDASNESSetFunctionLocal(da, INSERT_VALUES, 
                                  (PetscErrorCode(*)(DMDALocalInfo *, void *, void *, void *)) form_residual, this);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESSetConvergenceTest(SNES_mechanical, converged, this, PETSC_NULL);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESSetDM(SNES_mechanical, da);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  ierr = SNESSetFromOptions(SNES_mechanical);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  // TODO restart
  // if CLI_restartInc == 0 ... 216-242


  Matrix3d math_I3 = Matrix3d::Identity();
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      F_last_inc.slice(Eigen::array<Index, 5>({i, j, 0, 0, 0}),
                      Eigen::array<Index, 5>({1, 1, grid_.cells[0], grid_.cells[1], grid_.cells[2]}))
        .setConstant(math_I3(i, j));
    }
  }

  double* F_raw;
  ierr = VecGetArray(F_PETSc, &F_raw);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  TensorMap<Tensor<double, 5>> F(F_raw,  3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2);
  print_f("F1", F);
  std::memcpy(F.data(), F_last_inc.data(), sizeof(double) * F.size());
  *spectral.homogenization_F0 = F_last_inc.reshape(Eigen::DSizes<Eigen::DenseIndex, 3>(3, 3, grid_.cells[0]*grid_.cells[1]*grid_.cells2));

  Tensor<double, 4> x_n(3, grid_.cells[0]+1, grid_.cells[1]+1, grid_.cells2+1);
  Tensor<double, 4> x_p(3, grid_.cells[0], grid_.cells[1], grid_.cells2);
  base_update_coords(F, x_n, x_p); // fix after implementing restart

  Tensor<double, 5> P(3, 3, grid_.cells[0], grid_.cells[1], grid_.cells[2]);
  P_av.resize(3, 3);
  C_volAvg.resize(3, 3, 3, 3);
  C_minMaxAvg.resize(3, 3, 3, 3);
  P = spectral.constitutive_response(P_av,
                                     C_volAvg,
                                     C_minMaxAvg,
                                     F,
                                     0, grid_.world_rank);
  cout << "INIT F " << F << endl;

  ierr = VecRestoreArray(F_PETSc, &F_raw);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  // TODO restart
  // add restart calls 253-268
  update_gamma(C_minMaxAvg);
  C_minMaxAvgRestart = C_minMaxAvg;
}

Config::SolutionState MechSolverBasic::calculate_solution(std::string& inc_info_) {
  inc_info = inc_info_;
  Config::SolutionState solution;
  SNESConvergedReason reason;

  // update stiffness (and gamma operator)
  S = calculate_masked_compliance(C_volAvg, params.rot_bc_q, params.stress_mask);
  // print_f("P1", *spectral.homogenization_P);
  // print_f("S", S);
  // print_f("C_volAvg", C_volAvg);
  // print_f("homogenization_F", *spectral.homogenization_F);
  double* F_raw;
  ierr = VecGetArray(F_PETSc, &F_raw);
  TensorMap<Tensor<double, 5>> F(F_raw,  3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2);
  cout << "SOL F1 " << F << endl;
  ierr = VecRestoreArray(F_PETSc, &F_raw);

  if (config.numerics.grid_mechanical.update_gamma) update_gamma(C_minMaxAvg);


  ierr = VecGetArray(F_PETSc, &F_raw);
  TensorMap<Tensor<double, 5>> F_(F_raw,  3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2);
  cout << "SOL F2 " << F_ << endl;
  ierr = VecRestoreArray(F_PETSc, &F_raw);


  ierr = SNESSolve(SNES_mechanical, NULL, F_PETSc);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);


  ierr = VecGetArray(F_PETSc, &F_raw);
  TensorMap<Tensor<double, 5>> F__(F_raw,  3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2);
  cout << "SOL F3 " << F__ << endl;
  ierr = VecRestoreArray(F_PETSc, &F_raw);


  ierr = SNESGetConvergedReason(SNES_mechanical, &reason);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  // print_f("P2", *spectral.homogenization_P);

  solution.converged = reason > 0;
  solution.required_iterations = total_iter;
  solution.terminally_ill = *spectral.terminally_ill;
  *spectral.terminally_ill = false;

  Matrix<double, 3, 3> P_av_ = tensor_to_mat(P_av);
  Matrix<double, 3, 3> P_aim_ = tensor_to_mat(P_aim);
  P_aim_ = merge(P_av_, P_aim_, params.stress_mask);
  P_aim = mat_to_tensor(P_aim_);

  return solution;
}

void MechSolverBasic::forward(bool cutBack, bool guess, double delta_t, double delta_t_old, double t_remaining,
                              Config::BoundaryCondition& deformation_bc, 
                              Config::BoundaryCondition& stress_bc, 
                              Quaterniond& rot_bc_q) {
  PetscErrorCode ierr;
  PetscScalar *F_raw;
  ierr = VecGetArray(F_PETSc, &F_raw);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  TensorMap<Tensor<double, 5>> F(reinterpret_cast<double*>(F_raw), 3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2);
  print_f("F2", F);

  if (cutBack) {
    C_volAvg = C_volAvgLastInc;
    C_minMaxAvg = C_minMaxAvgLastInc;
  } else {
    C_volAvgLastInc = C_volAvg;
    C_minMaxAvgLastInc = C_minMaxAvg;

    Matrix<double, 3, 3> aim_diff = (F_aim-F_aim_last_inc)/delta_t_old;
    Matrix<double, 3, 3> stress_components = merge(0.0, aim_diff, stress_bc.mask);
    F_aim_dot = merge(stress_components, 0.0, guess);

    if (!guess) F_aim_dot.setZero();
    F_aim_last_inc = F_aim;
    Matrix<double, 3, 3> F_aim_step;
    if (deformation_bc.type == "L") {
      F_aim_step = (deformation_bc.values - F_aim_last_inc) / t_remaining;
    } else if (deformation_bc.type == "dot_F") {
      F_aim_step = deformation_bc.values;
    } else if (deformation_bc.type == "F") {
      F_aim_step = (deformation_bc.values - F_aim_last_inc) / t_remaining;
    }
    F_aim_dot += merge(0.0, F_aim_step, deformation_bc.mask);
    *spectral.homogenization_F0 = F.reshape(Eigen::DSizes<Eigen::DenseIndex, 3>(3, 3, grid_.cells[0]*grid_.cells[1]*grid_.cells[2]));
    Tensor<double, 2> rotated = FortranUtilities::rotate_tensor2(rot_bc_q, F_aim_dot, true);
    F_dot = calculate_rate (guess, F_last_inc, 
                            spectral.homogenization_F0->reshape(Eigen::array<int, 5>{3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2}), 
                            delta_t_old, rotated);
    F_last_inc = spectral.homogenization_F0->reshape(Eigen::array<int, 5>{3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2});
  }
  F_aim = F_aim_last_inc + F_aim_dot * delta_t;
  Eigen::Tensor<double, 2> zero_tensor(P_aim.dimensions());
  zero_tensor.setZero();
  if (stress_bc.type == "P") {
    P_aim += mat_to_tensor(stress_bc.mask).select((mat_to_tensor(stress_bc.values) - P_aim) / t_remaining, zero_tensor) * delta_t;
  } else {
    P_aim += mat_to_tensor(stress_bc.mask).select(mat_to_tensor(stress_bc.values), zero_tensor) * delta_t;
  }
  Tensor<double, 2> rotated = FortranUtilities::rotate_tensor2(rot_bc_q, F_aim);
  F = forward_field(delta_t, F_last_inc, F_dot, rotated);
  print_f("FSO", F);
  ierr = VecRestoreArray(F_PETSc, &F_raw);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);

  params.stress_mask = stress_bc.mask;
  params.rot_bc_q = rot_bc_q;
  params.delta_t = delta_t;
} 

void MechSolverBasic::update_coords() {
  PetscScalar *F_raw;
  ierr = VecGetArray(F_PETSc, &F_raw); 

  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  
  TensorMap<Tensor<double, 5>> F(reinterpret_cast<double*>(F_raw), 3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2);
  print_f("F3", F);

  Tensor<double, 4> x_n(3, grid_.cells[0]+1, grid_.cells[1]+1, grid_.cells2+1);
  Tensor<double, 4> x_p(3, grid_.cells[0], grid_.cells[1], grid_.cells2);
  base_update_coords(F, x_n, x_p);
  cout << "UPDATECOORDS F " << F << endl;

  *grid_.ip_coords = x_p.reshape(Eigen::array<Eigen::Index, 2>({3, grid_.n_cells_local}));
  *grid_.node_coords = x_n.reshape(Eigen::array<Eigen::Index, 2>({3, grid_.n_nodes_local}));
  ierr = VecRestoreArray(F_PETSc, &F_raw);
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
  Tensor<double, 0> max_P_av = mech_basic->P_av.maximum();
  double divTol = std::max(max_P_av(0) * mech_basic->config.numerics.grid_mechanical.eps_rel_div, mech_basic->config.numerics.grid_mechanical.eps_abs_div);
  double BCTol = std::max(max_P_av(0) * mech_basic->config.numerics.grid_mechanical.eps_rel_P, mech_basic->config.numerics.grid_mechanical.eps_abs_P);

  // cout << "totalIter" << mech_basic->total_iter << endl;
  // cout << "num%itmin" << mech_basic->config.numerics.grid_mechanical.itmin << endl;
  // cout << "err_div" << mech_basic->err_div << endl;
  // cout << "divTol" << divTol << endl;
  // cout << "err_BC" << mech_basic->err_BC << endl;
  // cout << "BCTol" << BCTol << endl;
  // cout << "terminallyIll" << *mech_basic->spectral.terminally_ill << endl;

  if ((mech_basic->total_iter >= mech_basic->config.numerics.grid_mechanical.itmin && 
       std::max(mech_basic->err_div / divTol, mech_basic->err_BC / BCTol) < 1.0) || 
      *mech_basic->spectral.terminally_ill) {
    *reason = SNES_BREAKOUT_INNER_ITER;
    cout << "reason 1";

  } else if (mech_basic->total_iter >= mech_basic->config.numerics.grid_mechanical.itmax) {
    *reason = SNES_DIVERGED_MAX_IT;
    cout << "reason 2";
  } else {
    *reason = SNES_CONVERGED_ITERATING;
    cout << "reason 3";
  }
  std::cout << "\n ... reporting ............................................................." << std::endl;
  std::cout << " error divergence = " << std::fixed << std::setprecision(2) << std::setw(9) << mech_basic->err_div / divTol 
            << " (" << std::scientific << std::setprecision(2) << std::setw(9) << mech_basic->err_div << " / m, tol = " << std::scientific << std::setprecision(2) << std::setw(9) << divTol << ")" << std::endl;
  std::cout << " error stress BC  = " << std::fixed << std::setprecision(2) << std::setw(9) << mech_basic->err_BC / BCTol 
            << " (" << std::scientific << std::setprecision(2) << std::setw(9) << mech_basic->err_BC << " Pa,  tol = " << std::scientific << std::setprecision(2) << std::setw(9) << BCTol << ")" << std::endl;
  std::cout << "\n ===========================================================================" << std::endl;
  std::cout << std::fixed << std::setprecision(16);
  return 0;
}

PetscErrorCode MechSolverBasic::form_residual(DMDALocalInfo* residual_subdomain, double*** F_ptr, double*** r_ptr, void *ctx) {
  MechSolverBasic* mech_basic = static_cast<MechSolverBasic*>(ctx);
  TensorMap<Tensor<double, 5>> F(&F_ptr[0][0][0], 3, 3, mech_basic->grid_.cells[0], mech_basic->grid_.cells[1], mech_basic->grid_.cells2);
  TensorMap<Tensor<double, 5>> r(&r_ptr[0][0][0], 3, 3, mech_basic->grid_.cells[0], mech_basic->grid_.cells[1], mech_basic->grid_.cells2);
  int n_funcs, petsc_iter;
  print_f("FR F1", F);
  print_f("FR r1", r);
  // exit(8888);
  SNESGetNumberFunctionEvals(mech_basic->SNES_mechanical, &n_funcs);
  SNESGetIterationNumber(mech_basic->SNES_mechanical, &petsc_iter);

  if (n_funcs == 0 && petsc_iter == 0) mech_basic->total_iter = -1;

  if (mech_basic->total_iter <= petsc_iter) {
    mech_basic->total_iter += 1;
    std::cout << mech_basic->inc_info << " @ Iteration " << mech_basic->config.numerics.grid_mechanical.itmin << 
      "≤" << mech_basic->total_iter << "≤" << mech_basic->config.numerics.grid_mechanical.itmax << "\n";
    if (mech_basic->params.rot_bc_q != Quaterniond(1, 0, 0, 0)) {
      Tensor<double, 2> F_aim_rotated = FortranUtilities::rotate_tensor2(mech_basic->params.rot_bc_q, mech_basic->F_aim);
      Map<const Matrix<double, 3, 3>> F_aim_rotated_mat(F_aim_rotated.data());
      std::cout << " deformation gradient aim (lab) =\n" << F_aim_rotated_mat.transpose() << "\n";
    }
    std::cout << " deformation gradient aim       =\n" << mech_basic->F_aim.transpose() << "\n";
    std::cout.flush();
  }
  // print_f("P_av1", mech_basic->P_av);
  r = mech_basic->spectral.constitutive_response(mech_basic->P_av,
                                                 mech_basic->C_volAvg,
                                                 mech_basic->C_minMaxAvg,
                                                 F,
                                                 mech_basic->params.delta_t,
                                                 mech_basic->grid_.world_rank,
                                                 mech_basic->params.rot_bc_q);
  MPI_Allreduce(MPI_IN_PLACE, mech_basic->spectral.terminally_ill, 1, MPI_INTEGER, MPI_LOR, MPI_COMM_WORLD);
  print_f("r", r);
  mech_basic->err_div = mech_basic->calculate_divergence_rms(r);
  Matrix<double, 3, 3> delta_F_aim;
  f_math_mul3333xx33(mech_basic->S.data(), mech_basic->P_av.data(), delta_F_aim.data());
  mech_basic->F_aim = mech_basic->F_aim - delta_F_aim;
  print_f("P_av2", mech_basic->P_av);

  Eigen::Matrix<double, 3, 3> full_err = 
    Eigen::Map<Eigen::Matrix<double, 3, 3>>(mech_basic->P_av.data(), 3, 3) - 
    Eigen::Map<Eigen::Matrix<double, 3, 3>>(mech_basic->P_aim.data(), 3, 3);
  // Eigen::Matrix<double, 3, 3> a = merge(0.0, full_err, mech_basic->params.stress_mask).cwiseAbs();
  // print_f("Haha", a);
  mech_basic->err_BC = merge(0.0, full_err, mech_basic->params.stress_mask).cwiseAbs().maxCoeff();
  Tensor<double, 2>  rotated = FortranUtilities::rotate_tensor2(mech_basic->params.rot_bc_q, delta_F_aim, true);
  mech_basic->gamma_convolution(r, rotated);
  print_f("FR F2", F);
  print_f("FR r2", r);
  // exit(123);
  return 0;
}
