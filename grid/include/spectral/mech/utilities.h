#ifndef UTILITIES_H
#define UTILITIES_H

#include <vector>
#include <array>
#include <string>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
#include <spectral/spectral.h>
#include <config.h>
#include <discretization_grid.h>


using Eigen::Tensor;
using Eigen::TensorMap;
using Eigen::Matrix;
using Eigen::MatrixXd;
using namespace std;
using namespace Eigen;

class MechUtilities {
public:
  MechUtilities(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : config(config_), grid(grid_), spectral(spectral_) {}


  void init_utilities();
  virtual void update_coords (Tensor<double, 5> &F, 
                              Tensor<double, 2>& reshaped_x_n, 
                              Tensor<double, 2>& reshaped_x_p);
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
  virtual Tensor<double, 5> gamma_convolution(Tensor<double, 5> &field, 
                                              Tensor<double, 2> &field_aim);

  Tensor<double, 4> C_ref;
  Tensor<complex<double>, 7> gamma_hat;


  

protected:
  Spectral& spectral;
  DiscretizationGrid& grid;
  Config& config;
};
#endif // UTILITIES_H