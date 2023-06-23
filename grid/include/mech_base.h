#ifndef MECH_BASE_H
#define MECH_BASE_H

#include <config.h>
#include <discretization_grid.h>
#include <spectral.h>
#include <petsc.h>
#include <petscsnes.h>

#include <unsupported/Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::Matrix;
using Eigen::MatrixXd;
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
  }

  void init_utilities();

  /**
   * Calculate coordinates in current configuration for given defgrad field using integration in Fourier space.
   * @param F Deformation Gradient.
   * @param reshaped_x_n resulting node coordinates.
   * @param reshaped_x_p resulting point/cell center coordinates.
   */
  virtual void update_coords (Tensor<double, 5> &F, 
                              Tensor<double, 2>& reshaped_x_n, 
                              Tensor<double, 2>& reshaped_x_p);

  /**
   * Calculate coordinates in current configuration for given defgrad field using integration in Fourier space.
   * @param C input stiffness to store as reference stiffness.
   */
  virtual void update_gamma(Tensor<double, 4> &C); 
  virtual void forward_field (double delta_t, 
                              Tensor<double, 5> &field_last_inc, 
                              Tensor<double, 5> &rate, 
                              Tensor<double, 5> &forwarded_field,
                              Matrix<double, 3, 3>* aim = nullptr);
  virtual void calculate_masked_compliance (Tensor<double, 4> &C,
                                            Eigen::Quaterniond &rot_bc_q,
                                            const Matrix<bool, 3, 3> &mask_stress,
                                            Tensor<double, 4> &masked_compliance);
  virtual double calculate_divergence_rms(const Tensor<double, 5>& tensor_field);
  virtual Tensor<double, 5> gamma_convolution(TensorMap<Tensor<double, 5>> &field, 
                                              Tensor<double, 2> &field_aim);

  Tensor<double, 4> C_ref;
  Tensor<complex<double>, 7> gamma_hat;

  Eigen::Tensor<double, 2> P_av;
  Eigen::Tensor<double, 2> P_aim;
  Eigen::Tensor<double, 4> C_volAvg;
  Eigen::Tensor<double, 4> C_volAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvg;
  Eigen::Tensor<double, 4> C_minMaxAvgLastInc;
  Eigen::Tensor<double, 4> C_minMaxAvgRestart;

  Eigen::Tensor<double, 5> F_lastInc;
  Eigen::Tensor<double, 5> F_dot;

  Eigen::Tensor<double, 3> homogenization_F0;

  // Eigen::Matrix<double, 3, 3> F_aimDot = Eigen::Matrix<double, 3, 3>::Zero();
  Eigen::Matrix<double, 3, 3> F_aim = Eigen::Matrix<double, 3, 3>::Identity();
  // Eigen::Matrix<double, 3, 3> F_aim_lastInc = Eigen::Matrix<double, 3, 3>::Identity();
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

  SolutionParams params;

  SNES SNES_mechanical;
  int total_iter = 0;
  std::string inc_info;

  double err_BC;
  double err_div;

  Tensor<double, 4> S; 

protected:
  Spectral& spectral;
  DiscretizationGrid& grid;
  Config& config;
};
#endif // MECH_BASE_H