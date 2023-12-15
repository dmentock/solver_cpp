#ifndef SPECTRAL_H
#define SPECTRAL_H
#include <complex>
#include <optional>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Geometry> 
#include <fftw3-mpi.h>

#include "fft.h"
#include "discretization_grid.h"
#include "config.h"

using namespace std;
using namespace Eigen;

extern "C" {
  void f_homogenization_fetch_tensor_pointers(double** homogenization_F0, double** homogenization_F,
                                              double** homogenization_P, double** homogenization_dPdF,
                                              void** terminally_ill);
  void f_homogenization_mechanical_response(double* delta_t, int* cell_start, int* cell_end);
  void f_homogenization_thermal_response(double* delta_t, int* cell_start, int* cell_end);
  void f_homogenization_mechanical_response2(double* delta_t, int* FEsolving_execIP, int* FEsolving_execElem);
}

class Spectral {
public:
  void init(const int& spectral_derivative_id,
            DiscretizationGrid& grid_,
            const int& fftw_planner_flag,
            const double& fftw_timelimit);
  virtual std::array<std::complex<double>, 3> get_freq_derivative(int spectral_derivative_id, 
                                                                  std::array<int, 3> cells, 
                                                                  std::array<double, 3> geom_size, 
                                                                  std::array<int, 3>& k_s);
  virtual Tensor<double, 5> constitutive_response(Tensor<double, 2> &P_av, 
                                                  Tensor<double, 4> &C_volAvg, 
                                                  Tensor<double, 4> &C_minMaxAvg,
                                                  TensorMap<Tensor<double, 5>> &F,
                                                  double delta_t,
                                                  int rank,
                                                  std::optional<Quaterniond> rot_bc_q = std::nullopt);   

  virtual void homogenization_fetch_tensor_pointers(int n_cells_local) {
    double* homogenization_F0_raw_ptr;
    double* homogenization_F_raw_ptr;
    double* homogenization_P_raw_ptr;
    double* homogenization_dPdF_raw_ptr;
    void* raw_terminally_ill_raw_ptr;
    f_homogenization_fetch_tensor_pointers (&homogenization_F0_raw_ptr, &homogenization_F_raw_ptr, 
                                            &homogenization_P_raw_ptr, &homogenization_dPdF_raw_ptr,
                                            &raw_terminally_ill_raw_ptr);
    homogenization_F0 = std::make_unique<TensorMap<Tensor<double, 3>>>(homogenization_F0_raw_ptr, 3, 3, n_cells_local);
    homogenization_F = std::make_unique<TensorMap<Tensor<double, 3>>>(homogenization_F_raw_ptr, 3, 3, n_cells_local);
    homogenization_P = std::make_unique<TensorMap<Tensor<double, 3>>>(homogenization_P_raw_ptr, 3, 3, n_cells_local);
    homogenization_dPdF = std::make_unique<TensorMap<Tensor<double, 5>>>(homogenization_dPdF_raw_ptr, 3, 3, 3, 3, n_cells_local);
    terminally_ill = static_cast<bool*>(raw_terminally_ill_raw_ptr);
  }

  virtual void mechanical_response(double delta_t, int cell_start, int cell_end){
    f_homogenization_mechanical_response(&delta_t, &cell_start, &cell_end);
  }
  virtual void thermal_response(double delta_t, int cell_start, int cell_end){
    f_homogenization_thermal_response(&delta_t, &cell_start, &cell_end);
  }
  virtual void mechanical_response2(double delta_t, std::array<int, 2>& FEsolving_execIP, std::array<int, 2>& FEsolving_execElem){
    f_homogenization_mechanical_response2(&delta_t, FEsolving_execIP.data(), FEsolving_execElem.data());
  }

  std::unique_ptr<TensorMap<Tensor<double, 3>>> homogenization_F0;
  std::unique_ptr<TensorMap<Tensor<double, 3>>> homogenization_F;
  std::unique_ptr<TensorMap<Tensor<double, 3>>> homogenization_P;
  std::unique_ptr<TensorMap<Tensor<double, 5>>> homogenization_dPdF;
  bool* terminally_ill;

  double wgt = 0.5;
  
  Tensor<std::complex<double>, 4> xi1st;
  Tensor<std::complex<double>, 4> xi2nd;

  unique_ptr<FFT<3>> scalarfield;
  unique_ptr<FFT<4>> vectorfield;
  unique_ptr<FFT<5>> tensorfield;

private:
  const double TAU = 2 * M_PI;
};

#endif // SPECTRAL_H
