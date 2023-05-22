#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <discretization_grid.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fftw3-mpi.h>
#include <complex>
#include <memory>
#include <cmath>
#include <iostream>
#include <config.h>

class Spectral {
public:
  Spectral(Config& config_, DiscretizationGrid& grid_)
      : config(config_), grid(grid_) {}

  void init();
  template <int Rank>
  void set_up_fftw(ptrdiff_t& cells1_fftw, 
                    ptrdiff_t& cells1_offset, 
                    ptrdiff_t& cells2_fftw,
                    int size,
                    std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, Rank>>>& field_real,
                    std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, Rank>>>& field_fourier,
                    fftw_complex*& field_fourier_fftw,
                    int fftw_planner_flag,
                    fftw_plan &plan_forth, 
                    fftw_plan &plan_back,
                    const std::string& label);
  virtual std::array<std::complex<double>, 3> get_freq_derivative(std::array<int, 3>& k_s);

  double wgt;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 5>>> tensorField_real;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>>> tensorField_fourier;
  fftw_complex* tensorField_fourier_fftw;
  fftw_plan plan_tensor_forth;
  fftw_plan plan_tensor_back;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 4>>> vectorField_real;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 4>>> vectorField_fourier;
  fftw_complex* vectorField_fourier_fftw;
  fftw_plan plan_vector_forth;
  fftw_plan plan_vector_back;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 3>>> scalarField_real;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>>> scalarField_fourier;
  fftw_complex* scalarField_fourier_fftw;
  fftw_plan plan_scalar_forth;
  fftw_plan plan_scalar_back;
  Eigen::Tensor<std::complex<double>, 4> xi1st;
  Eigen::Tensor<std::complex<double>, 4> xi2nd;

protected:
  DiscretizationGrid& grid;
  Config& config;
  int tensor_size = 9;
  int vector_size = 3;
  int scalar_size = 1;

private:
  const double TAU = 2 * M_PI;
};
#endif // SPECTRAL_H