#include "spectral/spectral.h"
#include <mpi.h>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include <complex>
#include <vector>
#include <algorithm>
#include <fftw3-mpi.h>
#include <petsc.h>
#include <stdexcept>
#include <eigen_debug.h>
#include <cmath>

void Spectral::init(){
    std::cout << "\n <<<+-  spectral_utilities init  -+>>>" << std::endl;

    std::cout << "\n M. Diehl, Diploma Thesis TU München, 2010" << std::endl;
    std::cout << "https://doi.org/10.13140/2.1.3234.3840" << std::endl;

    std::cout << "\n P. Eisenlohr et al., International Journal of Plasticity 46:37–53, 2013" << std::endl;
    std::cout << "https://doi.org/10.1016/j.ijplas.2012.09.012" << std::endl;

    std::cout << "\n P. Shanthraj et al., International Journal of Plasticity 66:31–45, 2015" << std::endl;
    std::cout << "https://doi.org/10.1016/j.ijplas.2014.02.006" << std::endl;

    std::cout << "\n P. Shanthraj et al., Handbook of Mechanics of Materials, 2019" << std::endl;
    std::cout << "https://doi.org/10.1007/978-981-10-6855-3_80" << std::endl;

    //get num_grid 167-224
    cells0_reduced = grid.cells[0] / 2 + 1;
    Spectral::wgt = std::pow(grid.cells[0] * grid.cells[1] * grid.cells[2], -1);
    //get num vairables 177-210
    spectral_derivative_ID = DERIVATIVE_CONTINUOUS_ID;
    double scaled_geom_size[3] = {grid.geom_size[0], grid.geom_size[1], grid.geom_size[2]};
    int fftw_planner_flag = FFTW_MEASURE;
    //  call fftw_set_timelimit(num_grid%get_asFloat('fftw_timelimit',defaultVal=300.0_pReal)) 229
    fftw_set_timelimit(300.0);
    std::cout << "\n FFTW initialized" << std::endl;
    int cells_fftw[3] = {grid.cells[0], grid.cells[1], grid.cells[2]};
    ptrdiff_t fftw_dims[3] = {cells_fftw[2], cells_fftw[1], cells0_reduced};
    ptrdiff_t cells2_fftw, cells2_offset, cells1_fftw, cells1_offset, N;

    N = fftw_mpi_local_size_many_transposed(3, fftw_dims, Spectral::tensor_size, 
                                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                            PETSC_COMM_WORLD, 
                                            &cells2_fftw, &cells2_offset, 
                                            &cells1_fftw, &cells1_offset);
    cells1_tensor = cells1_fftw;
    cells1_offset_tensor = cells1_offset;
    if (cells2_fftw != grid.cells2)
        throw std::runtime_error("domain decomposition mismatch (tensor, real space)");
    tensorField_fourier_fftw = fftw_alloc_complex(N);
    tensorField_real.reset(new Eigen::TensorMap<Eigen::Tensor<double, 5>>(reinterpret_cast<double*>(tensorField_fourier_fftw), 
                                                                      {3,3,cells0_reduced*2, cells_fftw[1], cells2_fftw}));
    tensorField_fourier.reset(new Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>>(reinterpret_cast<std::complex<double>*>(tensorField_fourier_fftw), 
                                                                      {3,3,cells0_reduced, cells_fftw[2], cells1_fftw}));

    N = fftw_mpi_local_size_many_transposed(3, fftw_dims, Spectral::vector_size,
                                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                            PETSC_COMM_WORLD, 
                                            &cells2_fftw, &cells2_offset, 
                                            &cells1_fftw, &cells1_offset);
    if (cells2_fftw != grid.cells2)
        throw std::runtime_error("domain decomposition mismatch (vector, real space)");
    if (cells1_fftw != cells1_tensor)
        throw std::runtime_error("domain decomposition mismatch (vector, Fourier space)");
    vectorField_fourier_fftw = fftw_alloc_complex(N);
    vectorField_real.reset(new Eigen::TensorMap<Eigen::Tensor<double, 4>>(reinterpret_cast<double*>(vectorField_fourier_fftw), 
                                                                      {3,cells0_reduced*2, cells_fftw[1], cells2_fftw}));
    vectorField_fourier.reset(new Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 4>>(reinterpret_cast<std::complex<double>*>(vectorField_fourier_fftw),
                                                                      {3,cells0_reduced, cells_fftw[2], cells1_fftw}));

    N = fftw_mpi_local_size_many_transposed(3, fftw_dims, 1, //fftw_mpi_local_size_3d_transposed generates free(): invalid pointer or arithmetic error -> using many_transposed instead
                                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                            PETSC_COMM_WORLD, 
                                            &cells2_fftw, &cells2_offset, 
                                            &cells1_fftw, &cells1_offset);
    if (cells2_fftw != grid.cells2)
        throw std::runtime_error("domain decomposition mismatch (vector, real space)");
    if (cells1_fftw != cells1_tensor)
        throw std::runtime_error("domain decomposition mismatch (vector, Fourier space)");
    scalarField_fourier_fftw = fftw_alloc_complex(N);
    scalarField_real.reset(new Eigen::TensorMap<Eigen::Tensor<double, 3>>(reinterpret_cast<double*>(scalarField_fourier_fftw), 
                                                                      {cells0_reduced*2, cells_fftw[1], cells2_fftw}));
    scalarField_fourier.reset(new Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 3>>(reinterpret_cast<std::complex<double>*>(scalarField_fourier_fftw),
                                                                      {cells0_reduced, cells_fftw[2], cells1_fftw}));

    xi1st.resize(3, cells0_reduced, grid.cells[2], cells1_tensor);
    xi1st.setConstant(std::complex<double>(0,0));
    xi2nd.resize(3, cells0_reduced, grid.cells[2], cells1_tensor);
    xi2nd.setConstant(std::complex<double>(0,0));

    ptrdiff_t cells_fftw_reversed[3] = {cells_fftw[2], cells_fftw[1], cells_fftw[0]};

    Spectral::generate_plans(tensorField_real->data(), tensorField_fourier->data(), tensor_size, cells_fftw_reversed,
                             fftw_planner_flag, plan_tensor_forth, plan_tensor_back);
    if (!plan_tensor_forth) throw std::runtime_error("FFTW error r2c tensor");
    if (!plan_tensor_back) throw std::runtime_error("FFTW error c2r tensor");

    Spectral::generate_plans(vectorField_real->data(), vectorField_fourier->data(), vector_size, cells_fftw_reversed,
                             fftw_planner_flag, plan_vector_forth, plan_vector_back);
    if (!plan_vector_forth) throw std::runtime_error("FFTW error r2c vector");
    if (!plan_vector_back) throw std::runtime_error("FFTW error c2r vector");

    Spectral::generate_plans(scalarField_real->data(), scalarField_fourier->data(), 1, cells_fftw_reversed,
                             fftw_planner_flag, plan_scalar_forth, plan_scalar_back);
    if (!plan_scalar_forth) throw std::runtime_error("FFTW error r2c scalar");
    if (!plan_scalar_back) throw std::runtime_error("FFTW error c2r scalar");

    int k_s[3];
    std::array<std::complex<double>, 3>  freq_derivative;
    for (int j = cells1_offset_tensor; j < cells1_offset_tensor + cells1_tensor; ++j) {
        k_s[1] = j;
        if (j > grid.cells[1] / 2) k_s[1] = k_s[1] - grid.cells[1];

        for (int k = 0; k < grid.cells[2]; ++k) {
            k_s[2] = k;
            if (k > grid.cells[2] / 2) k_s[2] = k_s[2] - grid.cells[2];
            for (int i = 0; i < cells0_reduced; ++i) {
                k_s[0] = i;
                freq_derivative = get_freq_derivative(k_s);
                for (int p = 0; p < 3; ++p) {
                    xi2nd(p, i, k, j-cells1_offset_tensor) = freq_derivative[p];
                }
                if (grid.cells[0] % 2 == 0 && grid.cells[1] % 2 == 0 && grid.cells[2] % 2 == 0 && 
                    spectral_derivative_ID == DERIVATIVE_CONTINUOUS_ID) {
                    for (int p = 0; p < 3; ++p) {
                        xi1st(p, i, k, j-cells1_offset_tensor) = std::complex<double>(0.0, 0.0);
                    }
                } else {
                    for (int p = 0; p < 3; ++p) {
                        xi1st(p, i, k, j-cells1_offset_tensor) = xi2nd(p, i, k, j - cells1_offset_tensor);
                    }
                }
            }
        }
    }
    // if (num.memory_efficient) {
    //     gamma_hat.resize(3, 3, 3, 3, 1, 1, 1);
    // } else {
    //     gamma_hat.resize(3, 3, 3, 3, cells1Red, cells[2], cells2);
    // }
    gamma_hat.resize(3,3,3,cells0_reduced,grid.cells[2],cells1_tensor);
    gamma_hat.setZero();
}

