#ifndef MECH_BASE_H
#define MECH_BASE_H

#include <petsc.h>
#include <petscsnes.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "config.h"
#include "discretization_grid.h"
#include "spectral.h"

using namespace std;
using namespace Eigen;

//!  Base class for the mechanical spectral grid solver. 
/*!
  Implements all functionalities required by the basic and polarization solving scheme, extended by solver modules.
*/
class MechBase {
public:
  MechBase(Config& config_, DiscretizationGrid& grid__, Spectral& spectral_)
      : config(config_), grid_(grid__), spectral(spectral_) {
    S.resize(3, 3, 3, 3);
    S.setZero();
    P_av.resize(3, 3);
    P_av.setZero();
    P_aim.resize(3, 3);
    P_aim.setZero();
    F_last_inc.resize(3, 3, grid_.cells[0], grid_.cells[1], grid_.cells2);
    F_last_inc.setZero();
  }

  // Abstract solver interface
  virtual void init() = 0;
  virtual Config::SolutionState calculate_solution(std::string& inc_info_) = 0;
  virtual void forward (bool cutBack, bool guess, double delta_t, double delta_t_old, double t_remaining,
                Config::BoundaryCondition& deformation_BC, 
                Config::BoundaryCondition& stress_BC, 
                Quaterniond& rotation_BC) = 0;
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
                                  Tensor<double, 4>& x_n,
                                  Tensor<double, 4>& x_p);
  /**
   * Calculate coordinates in current configuration for given defgrad field using integration in Fourier space.
   * @param C input stiffness to store as reference stiffness.
   */
  virtual void update_gamma(Tensor<double, 4> &C); 
  virtual Tensor<double, 5> forward_field(double delta_t,
                                          Tensor<double, 5>& field_last_inc,
                                          Tensor<double, 5>& rate,
                                          const std::optional<Tensor<double, 2>>& aim = std::nullopt);
  virtual Tensor<double, 4> calculate_masked_compliance(Tensor<double, 4>& C,
                                                        Quaterniond &rot_bc_q,
                                                        Matrix<bool, 3, 3>& mask_stress);
  virtual double calculate_divergence_rms(const Tensor<double, 5>& tensor_field);
  virtual void gamma_convolution (TensorMap<Tensor<double, 5>>& field, 
                                  Tensor<double, 2>& field_aim);
  virtual Tensor<double, 5> calculate_rate(bool heterogeneous, 
                                                  const Tensor<double, 5>& field0, 
                                                  const Tensor<double, 5>& field, 
                                                  double dt, 
                                                  const Tensor<double, 2>& avRate);

  Tensor<double, 4> C_ref;
  Tensor<complex<double>, 7> gamma_hat;

  Tensor<double, 2> P_av;
  Tensor<double, 2> P_aim;
  Tensor<double, 4> C_volAvg;
  Tensor<double, 4> C_volAvgLastInc;
  Tensor<double, 4> C_minMaxAvg;
  Tensor<double, 4> C_minMaxAvgLastInc;
  Tensor<double, 4> C_minMaxAvgRestart;

  Tensor<double, 5> F_last_inc;
  Tensor<double, 5> F_dot;

  Matrix<double, 3, 3> F_aim_dot = Matrix<double, 3, 3>::Zero();
  Matrix<double, 3, 3> F_aim = Matrix<double, 3, 3>::Identity();
  Matrix<double, 3, 3> F_aim_last_inc = Matrix<double, 3, 3>::Identity();

  Config::SolutionParams params;

  DM da;
  SNES SNES_mechanical;
  Vec F_PETSc;

  int total_iter = 0;
  std::string inc_info;

  double err_BC = 0;
  double err_div = 0;

  Tensor<double, 4> S; 
  Spectral& spectral;
  Config& config;
  DiscretizationGrid& grid_;

protected:
  PetscErrorCode ierr;
};
#endif // MECH_BASE_H
