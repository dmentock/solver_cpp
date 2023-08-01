#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <fft.h>
#include <discretization_grid.h>
#include <config.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Geometry> 

#include <fftw3-mpi.h>
#include <complex>
#include <optional>

using namespace std;
using namespace Eigen;

extern "C" {
  void f_homogenization_fetch_tensor_pointers(int* n_cells_global,
                                              double** homogenization_F0, double** homogenization_F,
                                              double** homogenization_P, double** homogenization_dPdF,
                                              void** terminally_ill);
  void f_homogenization_mechanical_response(double* Delta_t, int* cell_start, int* cell_end);
  void f_homogenization_thermal_response(double* Delta_t, int* cell_start, int* cell_end);
  void f_homogenization_mechanical_response2(double* Delta_t, int* FEsolving_execIP, int* FEsolving_execElem);
}

class Spectral {
public:
  void init(int spectral_derivative_id, DiscretizationGrid& grid);
  virtual std::array<std::complex<double>, 3> get_freq_derivative(int spectral_derivative_id, 
                                                                  std::array<int, 3> cells, 
                                                                  std::array<double, 3> geom_size, 
                                                                  std::array<int, 3>& k_s);
  virtual void constitutive_response (TensorMap<Tensor<double, 5>> &P,
                                      Tensor<double, 2> &P_av, 
                                      Tensor<double, 4> &C_volAvg, 
                                      Tensor<double, 4> &C_minMaxAvg,
                                      TensorMap<Tensor<double, 5>> &F,
                                      double Delta_t,
                                      std::optional<Eigen::Quaterniond> rot_bc_q = std::nullopt);   

  virtual void homogenization_fetch_tensor_pointers(int n_cells_global) {
    double* homogenization_F0_raw_ptr;
    double* homogenization_F_raw_ptr;
    double* homogenization_P_raw_ptr;
    double* homogenization_dPdF_raw_ptr;
    void* raw_terminally_ill_raw_ptr;
    f_homogenization_fetch_tensor_pointers (&n_cells_global, 
                                            &homogenization_F0_raw_ptr, &homogenization_F_raw_ptr, 
                                            &homogenization_P_raw_ptr, &homogenization_dPdF_raw_ptr,
                                            &raw_terminally_ill_raw_ptr);
    homogenization_F0 = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(homogenization_F0_raw_ptr, 3, 3, n_cells_global);
    homogenization_F = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(homogenization_F_raw_ptr, 3, 3, n_cells_global);
    homogenization_P = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(homogenization_P_raw_ptr, 3, 3, n_cells_global);
    homogenization_dPdF = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 5>>>(homogenization_dPdF_raw_ptr, 3, 3, 3, 3, n_cells_global);
    terminally_ill = static_cast<bool*>(raw_terminally_ill_raw_ptr);
  }

  virtual void mechanical_response(double Delta_t, int cell_start, int cell_end);
  virtual void thermal_response(double Delta_t, int cell_start, int cell_end);
  virtual void mechanical_response2(double Delta_t, 
                                    std::array<int, 2>& FEsolving_execIP, 
                                    std::array<int, 2>& FEsolving_execElem);

  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 3>>> homogenization_F0;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 3>>> homogenization_F;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 3>>> homogenization_P;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, 5>>> homogenization_dPdF;
  bool* terminally_ill;

  double wgt = 0.5;
  
  Eigen::Tensor<std::complex<double>, 4> xi1st;
  Eigen::Tensor<std::complex<double>, 4> xi2nd;

  unique_ptr<FFT<3>> scalarfield;
  unique_ptr<FFT<4>> vectorfield;
  unique_ptr<FFT<5>> tensorfield;

  struct SolutionState {
    bool converged;
    int iterationsNeeded;
    bool terminally_ill;
  };

private:
  const double TAU = 2 * M_PI;
};

struct SolutionParams {
  Eigen::Matrix<double, 3, 3> stress_bc;
  Eigen::Matrix<bool, 3, 3> stress_mask;
  Eigen::Quaterniond rot_bc_q;
  double delta_t;
};


#endif // SPECTRAL_H