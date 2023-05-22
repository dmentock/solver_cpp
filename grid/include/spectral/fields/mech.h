#ifndef MECH_H
#define MECH_H

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <spectral/spectral.h>
#include <config.h>
#include <discretization_grid.h>

extern "C" {
  void f_math_invert(double *InvA, int *err, double *A, int *n);
  void f_rotate_tensor4(double *qu, double *T, double *rotated);
  void f_math_3333to99(double* m3333, double* m99);
  void f_math_99to3333(double* m99, double* m3333);
}

class Mech {
public:
  Mech(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : config(config_), grid(grid_), spectral(spectral_) {}


  void init_mech();
  virtual void update_coords (Eigen::Tensor<double, 5> &F, 
                              Eigen::Tensor<double, 2>& reshaped_x_n, 
                              Eigen::Tensor<double, 2>& reshaped_x_p);
  virtual void update_gamma(Eigen::Tensor<double, 4> &C);
  // virtual void constitutive_response (Eigen::Tensor<double, 5> &P, 
  //                                     Eigen::Tensor<double, 2> &P_av, 
  //                                     Eigen::Tensor<double, 4> &C_volAvg, 
  //                                     Eigen::Tensor<double, 4> &C_minMaxAvg,
  //                                     Eigen::Tensor<double, 5> &F,
  //                                     double Delta_t);
  virtual void forward_field (double delta_t, 
                              Eigen::Tensor<double, 5> &field_last_inc, 
                              Eigen::Tensor<double, 5> &rate, 
                              Eigen::Tensor<double, 5> &forwarded_field,
                              Eigen::Matrix<double, 3, 3>* aim = nullptr);
  virtual void calculate_masked_compliance (Eigen::Tensor<double, 4> &C,
                                            std::array<double, 4> &rot_bc_q,
                                            const Eigen::Matrix<bool, 3, 3> &mask_stress,
                                            Eigen::Tensor<double, 4> &masked_compliance);
  virtual double calculate_divergence_rms(const Eigen::Tensor<double, 5>& tensor_field);
  virtual Eigen::Tensor<double, 5> gamma_convolution (Eigen::Tensor<double, 5> &field, 
                                                      Eigen::Tensor<double, 2> &field_aim);

  Eigen::Tensor<double, 4> C_ref;
  Eigen::Tensor<std::complex<double>, 7> gamma_hat;

protected:
  Spectral& spectral;
  DiscretizationGrid& grid;
  Config& config;
};
#endif // MECH_H