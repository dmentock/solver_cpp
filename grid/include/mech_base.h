#ifndef MECH_BASE_H
#define MECH_BASE_H

#include <config.h>
#include <discretization_grid.h>
#include <spectral.h>
#include <petsc.h>
#include <petscsnes.h>

#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

//!  Base class for the mechanical spectral grid solver. 
/*!
  Implements all functionalities required by the basic and polarization solving scheme, extended by solver modules.
*/
class MechBase {
public:
  MechBase(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : config(config_), grid(grid_), spectral(spectral_) {
    S.resize(3, 3, 3, 3);
    S.setZero();
    P_av.resize(3, 3);
    P_av.setZero();
    P_aim.resize(3, 3);
    P_aim.setZero();
    F_last_inc.resize(3, 3, grid.cells[0], grid.cells[1], grid.cells2);
    F_last_inc.setZero();
  }

  // Abstract solver interface
  virtual void init() = 0;
  virtual Config::SolutionState calculate_solution(std::string& inc_info_) = 0;
  virtual void forward (bool cutBack, bool guess, double Delta_t, double Delta_t_old, double t_remaining,
                Config::BoundaryCondition& deformation_BC, 
                Config::BoundaryCondition& stress_BC, 
                Eigen::Quaterniond& rotation_BC) = 0;
  virtual void update_coords() = 0;

  // Base interface
  virtual void base_init();

  /**
   * Calculate coordinates in current configuration for given defgrad field using integration in Fourier space.
   * @param F Deformation Gradient.
   * @param reshaped_x_n resulting node coordinates.
   * @param reshaped_x_p resulting point/cell center coordinates.
   */
  virtual void base_update_coords(TensorMap<Tensor<double, 5>>& F, 
                                  Tensor<double, 2>& x_n_, 
                                  Tensor<double, 2>& x_p_);
  /**
   * Calculate coordinates in current configuration for given defgrad field using integration in Fourier space.
   * @param C input stiffness to store as reference stiffness.
   */
  virtual void update_gamma(Tensor<double, 4> &C); 
  virtual Tensor<double, 5> forward_field(double delta_t, 
                                          Tensor<double, 5> &field_last_inc, 
                                          Tensor<double, 5> &rate, 
                                          Matrix<double, 3, 3>* aim = nullptr);
  virtual Tensor<double, 4> calculate_masked_compliance(Tensor<double, 4> &C,
                                                        Eigen::Quaterniond &rot_bc_q,
                                                        Matrix<bool, 3, 3> &mask_stress);
  virtual double calculate_divergence_rms(const Tensor<double, 5>& tensor_field);
  virtual void gamma_convolution (TensorMap<Tensor<double, 5>> &field, 
                                  Tensor<double, 2> &field_aim);
  virtual Eigen::Tensor<double, 5> calculate_rate(bool heterogeneous, 
                                                  const Eigen::Tensor<double, 5>& field0, 
                                                  const Eigen::Tensor<double, 5>& field, 
                                                  double dt, 
                                                  const Eigen::Tensor<double, 2>& avRate);

  Tensor<double, 4> C_ref;
  Tensor<complex<double>, 7> gamma_hat;

  Eigen::Tensor<double, 2> P_av;
  Eigen::Tensor<double, 2> P_aim;
  Eigen::Tensor<double, 4> C_volAvg;
  Eigen::Tensor<double, 4> C_volAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvg;
  Eigen::Tensor<double, 4> C_minMaxAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvgRestart;

  Eigen::Tensor<double, 5> F_last_inc;
  Eigen::Tensor<double, 5> F_dot;

  Eigen::Matrix<double, 3, 3> F_aim_dot = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, 3, 3> F_aim = Eigen::Matrix<double, 3, 3>::Identity();
  Eigen::Matrix<double, 3, 3> F_aim_last_inc = Eigen::Matrix<double, 3, 3>::Identity();
  // Eigen::Matrix<double, 3, 3> P_av = Eigen::Matrix<double, 3, 3>::Zero();
  // Eigen::Matrix<double, 3, 3> P_aim = Eigen::Matrix<double, 3, 3>::Zero();

  // // 4D tensor of real numbers
  // Eigen::Tensor<double, 4> C_volAvg = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> C_volAvgLastInc = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> C_minMaxAvg = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> C_minMaxAvgLastInc = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> C_minMaxAvgRestart = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);
  // Eigen::Tensor<double, 4> S = Eigen::Tensor<double, 4>::Zero(3, 3, 3, 3);

  // // Scalar real numbers
  // double err_BC;
  // double err_div;

  Config::SolutionParams params;

  DM da;
  SNES SNES_mechanical;
  Vec solution_vec;

  int total_iter = 0;
  std::string inc_info;

  double err_BC = 0;
  double err_div = 0;

  Tensor<double, 4> S; 
  Spectral& spectral;
  Config& config;
  DiscretizationGrid& grid;

protected:
  PetscErrorCode ierr;
};
#endif // MECH_BASE_H