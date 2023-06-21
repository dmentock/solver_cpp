#ifndef SPECTRAL_H
#define SPECTRAL_H

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
  void f_homogenization_fetch_tensor_pointers(double** homogenization_F0, double** homogenization_F, 
                                              double** homogenization_P, double** homogenization_dPdF,
                                              void** terminally_ill);
  void f_homogenization_mechanical_response(double* Delta_t, int* cell_start, int* cell_end);
  void f_homogenization_thermal_response(double* Delta_t, int* cell_start, int* cell_end);
  void f_homogenization_mechanical_response2(double* Delta_t, int* FEsolving_execIP, int* FEsolving_execElem);
}

template <int Rank>
class FFT {
public:
  FFT(std::array<int, 3>& cells,
      int cells2,
      std::vector<int>& extra_dims,
      int fftw_planner_flag,
      ptrdiff_t* cells1_fftw = nullptr,
      ptrdiff_t* cells1_offset = nullptr,
      ptrdiff_t* cells2_fftw = nullptr) {
    init_fft(cells, cells2, extra_dims, fftw_planner_flag, cells1_fftw, cells1_offset, cells2_fftw);
  }
    void set_field_real(Eigen::Tensor<double, Rank> &field_real_);
    void set_field_fourier(Eigen::Tensor<complex<double>, Rank> &field_fourier_);
    Eigen::Tensor<double, Rank> get_field_real();
    Eigen::Tensor<complex<double>, Rank> get_field_fourier();


  template <typename TensorType>
  Eigen::Tensor<std::complex<double>, Rank> forward(TensorType* field_real_) {
    if (field_real_->data() != field_real->data()) {
      field_real->slice(indices_nullify_start, indices_nullify_extents).setZero();
      field_real->slice(indices_values_start, indices_values_extents_real) = *field_real_;
    }
    fftw_mpi_execute_dft_r2c(plan_forth, field_real->data(), field_fourier_fftw);
    Eigen::TensorMap<Eigen::Tensor<complex<double>, Rank>> field_fourier_map = *field_fourier;
    Eigen::Tensor<complex<double>, Rank> field_fourier_ = field_fourier_map;
    return field_fourier_.slice(indices_values_start, indices_values_extents_fourier);
  }

  template <typename TensorType>
  Eigen::Tensor<double, Rank> backward(TensorType* field_fourier_, double &wgt) {
    if (field_fourier_->data() != field_fourier->data()) {
      field_fourier->slice(indices_values_start, indices_values_extents_fourier) = *field_fourier_;
    }
    fftw_mpi_execute_dft_c2r(plan_back, field_fourier_fftw, field_real->data());
    Eigen::TensorMap<Eigen::Tensor<double, Rank>> field_real_map = *field_real;
    Eigen::Tensor<double, Rank> field_real_ = field_real_map * wgt;
    return field_real_.slice(indices_values_start, indices_values_extents_real);
  }


  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, Rank>>> field_real;
  std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<complex<double>, Rank>>> field_fourier;

protected:
    void init_fft(std::array<int, 3>& cells,
                int cells2,
                std::vector<int>& extra_dims,
                int fftw_planner_flag,
                ptrdiff_t* cells1_fftw = nullptr,
                ptrdiff_t* cells1_offset = nullptr,
                ptrdiff_t* cells2_fftw = nullptr);

  fftw_complex* field_fourier_fftw;
  fftw_plan plan_forth;
  fftw_plan plan_back;

  Eigen::array<ptrdiff_t, Rank> dims_real;
  Eigen::array<ptrdiff_t, Rank> dims_fourier;

  Eigen::array<Eigen::Index, Rank> indices_nullify_start;
  Eigen::array<Eigen::Index, Rank> indices_nullify_extents;
  Eigen::array<Eigen::Index, Rank> indices_values_start;
  Eigen::array<Eigen::Index, Rank> indices_values_extents_real;
  Eigen::array<Eigen::Index, Rank> indices_values_extents_fourier;
};

class Spectral {
public:
  Spectral(Config& config_, DiscretizationGrid& grid_)
      : config(config_), grid(grid_) {}

  void init();
  virtual std::array<std::complex<double>, 3> get_freq_derivative(std::array<int, 3>& k_s);
  virtual void constitutive_response (TensorMap<Tensor<double, 5>> &P,
                                      Tensor<double, 2> &P_av, 
                                      Tensor<double, 4> &C_volAvg, 
                                      Tensor<double, 4> &C_minMaxAvg,
                                      TensorMap<Tensor<double, 5>> &F,
                                      double Delta_t,
                                      std::optional<Eigen::Quaterniond> rot_bc_q = std::nullopt);   

  virtual void homogenization_fetch_tensor_pointers() {
    double* homogenization_F0_ptr;
    double* homogenization_F_ptr;
    double* homogenization_P_ptr;
    double* homogenization_dPdF_ptr;
    void* raw_terminally_ill_ptr;
    f_homogenization_fetch_tensor_pointers (&homogenization_F0_ptr, &homogenization_F_ptr, 
                                            &homogenization_P_ptr, &homogenization_dPdF_ptr,
                                            &raw_terminally_ill_ptr);
    homogenization_F0 = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(homogenization_F0_ptr, 3, 3, grid.n_cells_global);
    homogenization_F = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(homogenization_F_ptr, 3, 3, grid.n_cells_global);
    homogenization_P = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(homogenization_P_ptr, 3, 3, grid.n_cells_global);
    homogenization_dPdF = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 5>>>(homogenization_dPdF_ptr, 3, 3, 3, 3, grid.n_cells_global);
    terminally_ill = static_cast<bool*>(raw_terminally_ill_ptr);
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

  std::unique_ptr<FFT<3>> scalarfield;
  std::unique_ptr<FFT<4>> vectorfield;
  std::unique_ptr<FFT<5>> tensorfield;
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