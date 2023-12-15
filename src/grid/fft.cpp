#include <petsc.h>
#include <fftw3-mpi.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <complex>

#include "fft.h"

template <int Rank>
void FFT<Rank>::init (std::array<int, 3>& cells,
                      int cells2,
                      std::vector<int>& field_dims,
                      int fftw_planner_flag,
                      ptrdiff_t& cells1_tensor,
                      ptrdiff_t& cells1_tensor_offset) {

  int cells0_reduced = cells[0]/2+1;

  ptrdiff_t dim_fourier_y, dim_fourier_z, N, dims_fourier_offset_z;
  std::array<ptrdiff_t, 3> fftw_dims = {(ptrdiff_t)cells[2], (ptrdiff_t)cells[1], cells0_reduced};
  int size = std::accumulate(field_dims.begin(), field_dims.end(), 1, std::multiplies<int>());

  N = fftw_mpi_local_size_many_transposed(3, fftw_dims.data(), size, 
                                          FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                          PETSC_COMM_WORLD, 
                                          &dim_fourier_z, &dims_fourier_offset_z, 
                                          &dim_fourier_y, &dims_fourier_offset_y);

  cells1_tensor = dim_fourier_y;
  cells1_tensor_offset = dims_fourier_offset_y;

  dims_real = {cells0_reduced * 2, cells[1], cells2};
  dims_fourier = {cells0_reduced, cells[2], (int)dim_fourier_y};

  field_fourier_fftw = fftw_alloc_complex(N);
  Eigen::array<ptrdiff_t, Rank> field_dims_real;
  std::copy(field_dims.begin(), field_dims.end(), field_dims_real.begin());
  std::copy(dims_real.begin(), dims_real.end(), field_dims_real.begin() + field_dims.size());
  field_real.reset(new TensorMap<Tensor<double, Rank>>(
    reinterpret_cast<double*>(field_fourier_fftw), field_dims_real));
  Eigen::array<ptrdiff_t, Rank> field_dims_fourier;
  std::copy(field_dims.begin(), field_dims.end(), field_dims_fourier.begin());
  std::copy(dims_fourier.begin(), dims_fourier.end(), field_dims_fourier.begin() + field_dims.size());
  field_fourier.reset(new TensorMap<Tensor<std::complex<double>, Rank>>(
    reinterpret_cast<std::complex<double>*>(field_fourier_fftw), field_dims_fourier));
  std::array<ptrdiff_t, 3> cells_reversed = {cells[2], cells[1], cells[0]};

  plan_forth = fftw_mpi_plan_many_dft_r2c(3, cells_reversed.data(), size,
                                          FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                          field_real->data(),
                                          reinterpret_cast<fftw_complex*>(field_fourier->data()),
                                          PETSC_COMM_WORLD, fftw_planner_flag | FFTW_MPI_TRANSPOSED_OUT);

  plan_back = fftw_mpi_plan_many_dft_c2r (3, cells_reversed.data(), size,
                                          FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                          reinterpret_cast<fftw_complex*>(field_fourier->data()),
                                          field_real->data(),
                                          PETSC_COMM_WORLD, fftw_planner_flag | FFTW_MPI_TRANSPOSED_IN);

  // set indices for field value assignments
  indices_nullify_start.fill(0);
  indices_nullify_start[Rank - 3] = cells[0];
  indices_nullify_extents.fill(3);
  indices_nullify_extents[Rank - 3] = cells0_reduced * 2 - cells[0];
  indices_nullify_extents[Rank - 2] = cells[1];
  indices_nullify_extents[Rank - 1] = cells2;

  indices_values_start.fill(0);
  indices_values_extents_real.fill(3);
  indices_values_extents_real[Rank - 3] = cells[0];
  indices_values_extents_real[Rank - 2] = cells[1];
  indices_values_extents_real[Rank - 1] = cells2;

  indices_values_extents_fourier.fill(3);
  indices_values_extents_fourier[Rank - 3] = cells0_reduced;
  indices_values_extents_fourier[Rank - 2] = cells[2];
  indices_values_extents_fourier[Rank - 1] = dim_fourier_y;
}


template <int Rank>
void FFT<Rank>::set_field_real(Tensor<double, Rank> &field_real_) {
  field_real->slice(indices_nullify_start, indices_nullify_extents).setZero(); 
  field_real->slice(indices_values_start, indices_values_extents_real) = field_real_;
}

template <int Rank>
void FFT<Rank>::set_field_fourier(Tensor<std::complex<double>, Rank> &field_fourier_) {
  field_fourier->slice(indices_values_start, indices_values_extents_fourier) = field_fourier_;
}

template <int Rank>
Tensor<double, Rank> FFT<Rank>::get_field_real() {
  Tensor<double, Rank> field_real_ = *field_real;
  return field_real_;
}

template <int Rank>
Tensor<std::complex<double>, Rank> FFT<Rank>::get_field_fourier() {
  Tensor<std::complex<double>, Rank> field_fourier_ = *field_fourier;

  return field_fourier_;
}

template class FFT<3>;
template class FFT<4>;
template class FFT<5>;
