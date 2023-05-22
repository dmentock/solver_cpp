#include "spectral/spectral.h"

#include <mpi.h>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <helper.h>
#include <tensor_operations.h>

#include <complex>
#include <vector>
#include <algorithm>
#include <fftw3-mpi.h>
#include <petsc.h>
#include <stdexcept>
#include <cmath>

void Spectral::init(){
  std::cout << "\n <<<+-  spectral init  -+>>>" << std::endl;

  std::cout << "\n M. Diehl, Diploma Thesis TU München, 2010" << std::endl;
  std::cout << "https://doi.org/10.13140/2.1.3234.3840" << std::endl;

  std::cout << "\n P. Eisenlohr et al., International Journal of Plasticity 46:37–53, 2013" << std::endl;
  std::cout << "https://doi.org/10.1016/j.ijplas.2012.09.012" << std::endl;

  std::cout << "\n P. Shanthraj et al., International Journal of Plasticity 66:31–45, 2015" << std::endl;
  std::cout << "https://doi.org/10.1016/j.ijplas.2014.02.006" << std::endl;

  std::cout << "\n P. Shanthraj et al., Handbook of Mechanics of Materials, 2019" << std::endl;
  std::cout << "https://doi.org/10.1007/978-981-10-6855-3_80" << std::endl;

  grid.cells0_reduced = grid.cells[0] / 2 + 1;
  wgt = std::pow(grid.cells[0] * grid.cells[1] * grid.cells[2], -1);




    //get num vairables 177-210
  double scaled_geom_size[3] = {grid.geom_size[0], grid.geom_size[1], grid.geom_size[2]};
  int fftw_planner_flag = FFTW_MEASURE;
  //  call fftw_set_timelimit(num_grid%get_asFloat('fftw_timelimit',defaultVal=300.0_pReal)) 229
  fftw_set_timelimit(300.0);
  std::cout << "\n FFTW initialized" << std::endl;

  std::array<ptrdiff_t, 3> cells_fftw = {grid.cells[0], grid.cells[1], grid.cells[2]};

  ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;

  // call tensor func
  set_up_fftw(cells1_fftw, cells1_offset, 
              cells2_fftw,
              tensor_size, 
              tensorField_real, tensorField_fourier, tensorField_fourier_fftw,
              fftw_planner_flag, plan_tensor_forth, plan_tensor_back,
              "tensor");
  grid.cells1_tensor = cells1_fftw;
  grid.cells1_offset_tensor = cells1_offset;

  set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, vector_size, 
              vectorField_real, vectorField_fourier, vectorField_fourier_fftw,
              fftw_planner_flag, plan_vector_forth, plan_vector_back,
              "vector");

  set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, scalar_size, 
              scalarField_real, scalarField_fourier, scalarField_fourier_fftw,
              fftw_planner_flag, plan_scalar_forth, plan_scalar_back,
              "scalar");

  xi1st.resize(3, grid.cells0_reduced, grid.cells[2], grid.cells1_tensor);
    xi1st.setConstant(std::complex<double>(0,0));
  xi2nd.resize(3, grid.cells0_reduced, grid.cells[2], grid.cells1_tensor);
    xi2nd.setConstant(std::complex<double>(0,0));

  std::array<int, 3> k_s;
  std::array<std::complex<double>, 3> freq_derivative;
  std::array<int, 3>  loop_indices;
  for (int j = grid.cells1_offset_tensor; j < grid.cells1_offset_tensor + grid.cells1_tensor; ++j) {
    k_s[1] = j;
    if (j > grid.cells[1] / 2) k_s[1] = k_s[1] - grid.cells[1];
    for (int k = 0; k < grid.cells[2]; ++k) {
      k_s[2] = k;
      if (k > grid.cells[2] / 2) k_s[2] = k_s[2] - grid.cells[2];
      for (int i = 0; i < grid.cells0_reduced; ++i) {
        k_s[0] = i;
        freq_derivative = get_freq_derivative(k_s);
        for (int p = 0; p < 3; ++p) xi2nd(p, i, k, j-grid.cells1_offset_tensor) = freq_derivative[p];
        loop_indices = {i,j,k};
        for (int p = 0; p < 3; ++p) {
          if (grid.cells[p] == loop_indices[p]+1){
            xi1st(p, i, k, j - grid.cells1_offset_tensor) = std::complex<double>(0.0, 0.0);
          } else {
            for (int p = 0; p < 3; ++p) {
              xi1st(p, i, k, j-grid.cells1_offset_tensor) = xi2nd(p, i, k, j - grid.cells1_offset_tensor);
            }
          }
        }
      }
    }
  }
}

template <int Rank>
void Spectral::set_up_fftw (ptrdiff_t& cells1_fftw, 
                            ptrdiff_t& cells1_offset, 
                            ptrdiff_t& cells2_fftw,
                            int size,
                            std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<double, Rank>>>& field_real,
                            std::unique_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, Rank>>>& field_fourier,
                            fftw_complex*& field_fourier_fftw,
                            int fftw_planner_flag,
                            fftw_plan& plan_forth, 
                            fftw_plan& plan_back,
                            const std::string& label){
    ptrdiff_t N, cells2_offset;
    ptrdiff_t fftw_dims[3] = {grid.cells[2], grid.cells[1], grid.cells0_reduced};
    N = fftw_mpi_local_size_many_transposed(3, fftw_dims, size, 
                                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                            PETSC_COMM_WORLD, 
                                            &cells2_fftw, &cells2_offset, 
                                            &cells1_fftw, &cells1_offset);
    field_fourier_fftw = fftw_alloc_complex(N);
    
    auto create_dimensions = [](const std::initializer_list<ptrdiff_t>& init_list) {
      Eigen::array<ptrdiff_t, Rank> arr;
      std::copy(init_list.begin(), init_list.end(), arr.begin());
      return arr;
    };
    Eigen::array<ptrdiff_t, Rank> dims_real;
    Eigen::array<ptrdiff_t, Rank> dims_fourier;
    if (label == "tensor") {
      dims_real = create_dimensions({3, 3, grid.cells0_reduced * 2, grid.cells[1], cells2_fftw});
      dims_fourier = create_dimensions({3, 3, grid.cells0_reduced, grid.cells[2], cells1_fftw});
    } else {
      if (cells1_fftw != grid.cells1_tensor) throw std::runtime_error("domain decomposition mismatch ("+ label +", Fourier space)");
      if (label == "vector") {
        dims_real = create_dimensions({3, grid.cells0_reduced * 2, grid.cells[1], cells2_fftw});
        dims_fourier = create_dimensions({3, grid.cells0_reduced, grid.cells[2], cells1_fftw});
      } else if (label == "scalar") {
        dims_real = create_dimensions({grid.cells0_reduced * 2, grid.cells[1], cells2_fftw});
        dims_fourier = create_dimensions({grid.cells0_reduced, grid.cells[2], cells1_fftw});
      } else {
        throw std::runtime_error("Invalid label");
      }
    } 
    field_real.reset(new Eigen::TensorMap<Eigen::Tensor<double, Rank>>(reinterpret_cast<double*>(field_fourier_fftw), dims_real));
    field_fourier.reset(new Eigen::TensorMap<Eigen::Tensor<std::complex<double>, Rank>>(reinterpret_cast<std::complex<double>*>(field_fourier_fftw), dims_fourier));
    ptrdiff_t cells_reversed[3] = {grid.cells[2], grid.cells[1], grid.cells[0]};              
    plan_forth = fftw_mpi_plan_many_dft_r2c(3, cells_reversed, size,
                                               FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                               field_real->data(),
                                               reinterpret_cast<fftw_complex*>(field_fourier->data()),
                                               PETSC_COMM_WORLD, fftw_planner_flag | FFTW_MPI_TRANSPOSED_OUT);

    plan_back = fftw_mpi_plan_many_dft_c2r(3, cells_reversed, size,
                                              FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                              reinterpret_cast<fftw_complex*>(field_fourier->data()),
                                              field_real->data(),
                                              PETSC_COMM_WORLD, fftw_planner_flag | FFTW_MPI_TRANSPOSED_IN);
    if (!plan_tensor_forth) throw std::runtime_error("FFTW error r2c " + label);
    if (!plan_tensor_back) throw std::runtime_error("FFTW error c2r " + label);
}

std::array<std::complex<double>, 3> Spectral::get_freq_derivative(std::array<int, 3>& k_s) {
    std::array<std::complex<double>, 3> freq_derivative;
    switch (config.num_grid.spectral_derivative_id) {
        case Config::DERIVATIVE_CONTINUOUS_ID:
            for (int i = 0; i < 3; ++i) {
                freq_derivative[i] = std::complex<double>(0.0, TAU * k_s[i] / grid.geom_size[i]);
            }
            break;
        case Config::DERIVATIVE_CENTRAL_DIFF_ID:
            for (int i = 0; i < 3; ++i) {
                freq_derivative[i] = std::complex<double>(0.0, sin(TAU * k_s[i] / grid.cells[i])) /
                                     std::complex<double>(2.0 * grid.geom_size[i] / grid.cells[i], 0.0);
            }
            break;
        case Config::DERIVATIVE_FWBW_DIFF_ID:
            for (int i = 0; i < 3; ++i) {
                freq_derivative[i] = (std::complex<double>(cos(TAU * k_s[i] / grid.cells[i]) - (i == 0 ? 1.0 : -1.0),
                                                      sin(TAU * k_s[i] / grid.cells[i])) *
                                      std::complex<double>(cos(TAU * k_s[(i + 1) % 3] / grid.cells[(i + 1) % 3]) + 1.0,
                                                      sin(TAU * k_s[(i + 1) % 3] / grid.cells[(i + 1) % 3])) *
                                      std::complex<double>(cos(TAU * k_s[(i + 2) % 3] / grid.cells[(i + 2) % 3]) + 1.0,
                                                      sin(TAU * k_s[(i + 2) % 3] / grid.cells[(i + 2) % 3])) /
                                      std::complex<double>(4.0 * grid.geom_size[i] / grid.cells[i]), 0.0);
            }
            break;
        default:
            throw std::runtime_error("Invalid spectral_derivative_ID value.");
    }
    return freq_derivative;
}

