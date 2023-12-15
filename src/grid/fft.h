#ifndef FFT_H
#define FFT_H

#include <unsupported/Eigen/CXX11/Tensor>
#include <fftw3-mpi.h>
#include <complex>
#include <iostream>
#include <type_traits>

using namespace std;
using namespace Eigen;

template <int Rank>
class FFT {
public:
  FFT(std::array<int, 3>& cells,
      int cells2,
      std::vector<int>& field_dims,
      int fftw_planner_flag,
      ptrdiff_t& cells1_tensor,
      ptrdiff_t& cells1_tensor_offset) {
      init(cells, cells2, field_dims, fftw_planner_flag, cells1_tensor, cells1_tensor_offset);
  }

  void init(std::array<int, 3>& cells,
            int cells2,
            std::vector<int>& field_dims,
            int fftw_planner_flag,
            ptrdiff_t& cells1_tensor,
            ptrdiff_t& cells1_tensor_offset);

  void set_field_real(Tensor<double, Rank> &field_real_);
  void set_field_fourier(Tensor<std::complex<double>, Rank> &field_fourier_);
  Tensor<double, Rank> get_field_real();
  Tensor<std::complex<double>, Rank> get_field_fourier();

  template <typename TensorType>
  Tensor<std::complex<double>, Rank> forward(TensorType& field_real_) {
    if (field_real_.data() != field_real->data()) {
      field_real->slice(indices_nullify_start, indices_nullify_extents).setZero();
      field_real->slice(indices_values_start, indices_values_extents_real) = field_real_;
    }
    fftw_mpi_execute_dft_r2c(plan_forth, field_real->data(), field_fourier_fftw);
    TensorMap<Tensor<std::complex<double>, Rank>> field_fourier_map = *field_fourier;
    Tensor<std::complex<double>, Rank> field_fourier_ = field_fourier_map;
    return field_fourier_.slice(indices_values_start, indices_values_extents_fourier);
  }

  template <typename TensorType>
  Tensor<double, Rank> backward(TensorType* field_fourier_, double &wgt) {
    // PERFORMANCE TODO: really necessary? Compiler may skip those actions automatically
    if (field_fourier_->data() != field_fourier->data()) {
      field_fourier->slice(indices_values_start, indices_values_extents_fourier) = *field_fourier_;
    }
    fftw_mpi_execute_dft_c2r(plan_back, field_fourier_fftw, field_real->data());

    TensorMap<Tensor<double, Rank>> field_real_map = *field_real;
    Tensor<double, Rank> field_real_ = field_real_map * wgt;
    return field_real_.slice(indices_values_start, indices_values_extents_real);
  }

  std::unique_ptr<TensorMap<Tensor<double, Rank>>> field_real;
  std::unique_ptr<TensorMap<Tensor<std::complex<double>, Rank>>> field_fourier;

  std::array<int, 3> dims_real;
  std::array<int, 3> dims_fourier;
  ptrdiff_t dims_fourier_offset_y;

protected:
  fftw_complex* field_fourier_fftw;
  fftw_plan plan_forth;
  fftw_plan plan_back;

  Eigen::array<Index, Rank> indices_nullify_start;
  Eigen::array<Index, Rank> indices_nullify_extents;
  Eigen::array<Index, Rank> indices_values_start;
  Eigen::array<Index, Rank> indices_values_extents_real;
  Eigen::array<Index, Rank> indices_values_extents_fourier;
};
#endif // FFT_H
