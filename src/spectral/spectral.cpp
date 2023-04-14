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

    int cells0_reduced = grid.cells[0] / 2 + 1;
    double wgt = std::pow(grid.cells[0] * grid.cells[1] * grid.cells[2], -1);

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
    Eigen::DSizes<Eigen::DenseIndex, 5> fieldshape_real(3,3,grid.cells[2],grid.cells[1],grid.cells[0]);
    Eigen::DSizes<Eigen::DenseIndex, 5> fieldshape_fourier(3,3,grid.cells[2],grid.cells[1],grid.cells[0]/2);

    N = fftw_mpi_local_size_many_transposed(3, fftw_dims, Spectral::tensor_size, 
                                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                            PETSC_COMM_WORLD, 
                                            &cells2_fftw, &cells2_offset, 
                                            &cells1_fftw, &cells1_offset);
    cells1_tensor = cells1_fftw;
    cells1_offset_tensor = cells1_offset;
    // if (cells2_fftw != grid.cells2)
    //     throw std::runtime_error("domain decomposition mismatch (tensor, real space)");
    fftw_complex* tensorField = fftw_alloc_complex(N);
    tensorField_real = new Eigen::TensorMap<Eigen::Tensor<double, 5>>(reinterpret_cast<double*>(tensorField), fieldshape_real);
    tensorField_fourier = new Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>>(reinterpret_cast<std::complex<double>*>(tensorField), fieldshape_fourier);

    N = fftw_mpi_local_size_many_transposed(3, fftw_dims, Spectral::vector_size,
                                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                            PETSC_COMM_WORLD, 
                                            &cells2_fftw, &cells2_offset, 
                                            &cells1_fftw, &cells1_offset);
    if (cells2_fftw != grid.cells2)
        throw std::runtime_error("domain decomposition mismatch (vector, real space)");
    if (cells1_fftw != cells1_tensor)
        throw std::runtime_error("domain decomposition mismatch (vector, Fourier space)");
    fftw_complex* vectorField = fftw_alloc_complex(N);
    vectorField_real = new Eigen::TensorMap<Eigen::Tensor<double, 5>>(reinterpret_cast<double*>(vectorField), fieldshape_real);
    vectorField_fourier = new Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>>(reinterpret_cast<std::complex<double>*>(vectorField), fieldshape_fourier);

    //fftw_mpi_local_size_3d_transposed generates free(): invalid pointer or arithmetic error 
    N = fftw_mpi_local_size_many_transposed(3, fftw_dims, 1,
                                            FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                            PETSC_COMM_WORLD, 
                                            &cells2_fftw, &cells2_offset, 
                                            &cells1_fftw, &cells1_offset);
    if (cells2_fftw != grid.cells2)
        throw std::runtime_error("domain decomposition mismatch (vector, real space)");
    if (cells1_fftw != cells1_tensor)
        throw std::runtime_error("domain decomposition mismatch (vector, Fourier space)");
    fftw_complex* scalarField = fftw_alloc_complex(N);
    scalarField_real = new Eigen::TensorMap<Eigen::Tensor<double, 5>>(reinterpret_cast<double*>(scalarField), fieldshape_real);
    scalarField_fourier = new Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>>(reinterpret_cast<std::complex<double>*>(scalarField), fieldshape_fourier);


    xi1st.resize(3, cells0_reduced, grid.cells[2], grid.cells[2]);
    xi1st.setConstant(std::complex<double>(0,0));
    xi2nd.resize(3, cells0_reduced, grid.cells[2], grid.cells[2]);
    xi2nd.setConstant(std::complex<double>(0,0));

    ptrdiff_t cells_fftw_reversed[3] = {cells_fftw[2], cells_fftw[1], cells_fftw[0]};

    Spectral::generate_plans(tensorField_real, tensorField_fourier, tensor_size, cells_fftw_reversed,
                             fftw_planner_flag, plan_tensor_forth, plan_tensor_back);
    if (!plan_tensor_forth) throw std::runtime_error("FFTW error r2c tensor");
    if (!plan_tensor_back) throw std::runtime_error("FFTW error c2r tensor");

    Spectral::generate_plans(vectorField_real, vectorField_fourier, vector_size, cells_fftw_reversed,
                             fftw_planner_flag, plan_vector_forth, plan_vector_back);
    if (!plan_vector_forth) throw std::runtime_error("FFTW error r2c vector");
    if (!plan_vector_back) throw std::runtime_error("FFTW error c2r vector");

    Spectral::generate_plans(scalarField_real, scalarField_fourier, 1, cells_fftw_reversed,
                             fftw_planner_flag, plan_scalar_forth, plan_scalar_back);
    if (!plan_scalar_forth) throw std::runtime_error("FFTW error r2c scalar");
    if (!plan_scalar_back) throw std::runtime_error("FFTW error c2r scalar");


    int k_s[3];
    for (int j = cells1_offset_tensor + 1; j <= cells1_offset_tensor + cells1_tensor; ++j) {
        k_s[1] = j - 1;
        if (j > grid.cells[1] / 2 + 1) k_s[1] = k_s[1] - grid.cells[1];

        for (int k = 1; k <= grid.cells[2]; ++k) {
            k_s[2] = k - 1;
            if (k > grid.cells[2] / 2 + 1) k_s[2] = k_s[2] - grid.cells[2];

            for (int i = 1; i <= cells0_reduced; ++i) {
                k_s[0] = i - 1;
                for (int l = 0; l <= 2; ++l) {
                    xi2nd(l, i, k, j - cells1_offset_tensor) = Spectral::get_freq_derivative(k_s);
                }
                if (grid.cells[0] % 2 == 0 && grid.cells[1] % 2 == 0 && grid.cells[2] % 2 == 0 && 
                    spectral_derivative_ID == DERIVATIVE_CONTINUOUS_ID) {
                    // for even grids, set the Nyquist Freq component to 0.0
                    for (int l = 0; l <= 2; ++l) {
                        xi1st(l, i, k, j - cells1_offset_tensor) = std::complex<double>(0.0, 0.0);
                    }
                } else {
                    for (int l = 0; l <= 2; ++l) {
                        xi1st(l, i, k, j - cells1_offset_tensor) = xi2nd(l, i, k, j - cells1_offset_tensor);
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
    gamma_hat.setZero();
}

std::complex<double> Spectral::get_freq_derivative(int k_s[3]){
    return std::complex<double>(0,0);
}

void Spectral::generate_plans(Eigen::TensorMap<Eigen::Tensor<double, 5>>* field_real,
                              Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>>* field_fourier, 
                              int size, ptrdiff_t cells_fftw_reversed[3], int fftw_planner_flag,
                              fftw_plan &plan_forth, 
                              fftw_plan &plan_back){
    plan_forth = fftw_mpi_plan_many_dft_r2c(3, cells_fftw_reversed, size,
                                                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                                field_real->data(),
                                                reinterpret_cast<fftw_complex*>(field_fourier->data()),
                                                PETSC_COMM_WORLD, fftw_planner_flag | FFTW_MPI_TRANSPOSED_OUT);

    plan_back = fftw_mpi_plan_many_dft_c2r(3, cells_fftw_reversed, size,
                                                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                                reinterpret_cast<fftw_complex*>(field_fourier->data()),
                                                field_real->data(),
                                                PETSC_COMM_WORLD, fftw_planner_flag | FFTW_MPI_TRANSPOSED_IN);
}

void Spectral::update_coords(Eigen::Tensor<double, 5> &F) {

    // Additional data structures needed
    // Eigen::Tensor<double, 4> x_p;
    // Eigen::Tensor<double, 4> u_tilde_p_padded;
    // Eigen::Tensor<double, 4> x_n;
    // Eigen::Tensor<double, 5> tensorField_real;
    // Eigen::Tensor<std::complex<double>, 5> tensorField_fourier;
    // Eigen::Tensor<std::complex<double>, 4> vectorField_fourier;
    // Eigen::Tensor<double, 4> vectorField_real;
    // Eigen::Tensor<std::complex<double>, 4> xi2nd;

    
    Eigen::Tensor<double, 4> x_p(3, grid.cells[0], grid.cells[1], grid.cells2);
    Eigen::Tensor<double, 4> u_tilde_p_padded(3, grid.cells[0], grid.cells[1], grid.cells2 + 2);
    Eigen::Tensor<double, 4> x_n(3, grid.cells[0] + 1, grid.cells[1] + 1, grid.cells2 + 1);


    // Replace the MPI-related code with the appropriate C++/MPI code for your MPI library.
    // For example, you might use the MPI C++ bindings or the Boost.MPI library.
    // The code below assumes the use of the MPI C++ bindings.

    std::vector<MPI::Request> request(4);
    std::vector<MPI::Status> status(4);

    // Eigen::Vector3d step = geomSize.cast<double>() / cells.cast<double>();

    // Replace tensorField_real and tensorField_fourier with the appropriate Eigen tensors.
    // For example, if they are defined as global variables, simply replace them with the
    // corresponding tensor names.
    // Spectral::tensorField_real.slice(Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
    //     Eigen::array<Eigen::Index, 5>({3, 3, grid.cells[0], grid.cells[1], grid.cells2})) = F;


    
    // .setConstant(math_I3(i, j));
    // .block(0, 0, 0, 3, 3, cells(1), cells(2), cells3) = F;
    // Spectral::tensorField_real.block(0, 0, 0, 3, 3, cells(1) + 1, cells(2), cells3) = 0.0;

    // fftw_mpi_execute_dft_r2c(planTensorForth, tensorField_real, tensorField_fourier);

    // // average F
    // Eigen::Matrix<double, 3, 3> Favg;
    // if (cells3Offset == 0) Favg = tensorField_fourier.real().block(0, 0, 0, 1, 1, 1) * wgt;

    // MPI::COMM_WORLD.Bcast(Favg.data(), 9, MPI::DOUBLE, 0);



    // Eigen::Vector3d step = grid.geom_size.array() / cells.array().cast<double>();

}
void Spectral::constitutive_response(Eigen::Tensor<double, 5> &P, 
                                     Eigen::Tensor<double, 2> &P_av, 
                                     Eigen::Tensor<double, 4> &C_volAvg, 
                                     Eigen::Tensor<double, 4> &C_minMaxAvg,
                                     Eigen::Tensor<double, 5> &F,
                                     double Delta_t) {

  // Reshape F
  std::cout << F.dimension(2);
  Eigen::Tensor<double, 3> homogenization_F;
  homogenization_F = F.reshape(Eigen::array<int, 3>({3, 3, grid.cells[0] * grid.cells[1] * grid.cells2}));

  // Call the homogenization functions
//   homogenization_mechanical_response(Delta_t, 1, product_cells[0] * product_cells[1]);
//   if (!terminallyIll)
//     homogenization_thermal_response(Delta_t, 1, product_cells[0] * product_cells[1]);
//   if (!terminallyIll)
//     homogenization_mechanical_response2(Delta_t, {1, 1}, {1, product_cells[0] * product_cells[1]});

//   // Compute P and P_av
//   Eigen::Tensor<double, 5> homogenization_P = /* ... */;
//   P = homogenization_P.reshape(F.dimensions());
//   P_av = (P.sum(Eigen::array<Eigen::Index, 1>({2})).sum(Eigen::array<Eigen::Index, 1>({3})).sum(Eigen::array<Eigen::Index, 1>({4})) * wgt).eval();

//   // MPI Communication
//   MPI_Allreduce(MPI_IN_PLACE, P_av.data(), 9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &err_MPI);
//   if (err_MPI != MPI_SUCCESS)
//     throw std::runtime_error("MPI error");

//   // Rotation of load frame
//   if (rotation_BC.is_present()) {
//     if (!rotation_BC.asQuaternion().isApprox(Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0))) {
//       std::cout << "Piola--Kirchhoff stress (lab) / MPa = " << P_av.transpose() * 1.e-6 << std::endl;
//     }
//     P_av = rotation_BC.rotate(P_av);
//   }

//   std::cout << "Piola--Kirchhoff stress       / MPa = " << P_av.transpose() * 1.e-6 << std::endl;

//   // Find dPdF_min and dPdF_max
//   // ...

//   // Compute C_minMaxAvg
//   C_minMaxAvg = 0.5 * (dPdF_max + dPdF_min);

//   // Compute C_volAvg
//   C_volAvg = homogenization_dPdF.sum(Eigen::array<Eigen::Index, 1>({4})) * wgt;

//   // MPI Communication
//   MPI_Allreduce(MPI_IN_PLACE, C_volAvg.data(), 81, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD)
}
void Spectral::update_gamma(Eigen::Tensor<double, 4> &C_minMaxAvg) {
}