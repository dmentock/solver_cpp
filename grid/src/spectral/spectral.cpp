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

std::array<std::complex<double>, 3> Spectral::get_freq_derivative(int k_s[3]) {
    std::array<std::complex<double>, 3> freq_derivative;
    switch (spectral_derivative_ID) {
        case DERIVATIVE_CONTINUOUS_ID:
            for (int i = 0; i < 3; ++i) {
                freq_derivative[i] = std::complex<double>(0.0, M_2_PI * k_s[i] / grid.geom_size[i]);
            }
            break;
        case DERIVATIVE_CENTRAL_DIFF_ID:
            for (int i = 0; i < 3; ++i) {
                freq_derivative[i] = std::complex<double>(0.0, sin(M_2_PI * k_s[i] / grid.cells[i])) /
                                     std::complex<double>(2.0 * grid.geom_size[i] / grid.cells[i], 0.0);
            }
            break;
        case DERIVATIVE_FWBW_DIFF_ID:
            for (int i = 0; i < 3; ++i) {
                freq_derivative[i] = (std::complex<double>(cos(M_2_PI * k_s[i] / grid.cells[i]) - (i == 0 ? 1.0 : -1.0),
                                                      sin(M_2_PI * k_s[i] / grid.cells[i])) *
                                      std::complex<double>(cos(M_2_PI * k_s[(i + 1) % 3] / grid.cells[(i + 1) % 3]) + 1.0,
                                                      sin(M_2_PI * k_s[(i + 1) % 3] / grid.cells[(i + 1) % 3])) *
                                      std::complex<double>(cos(M_2_PI * k_s[(i + 2) % 3] / grid.cells[(i + 2) % 3]) + 1.0,
                                                      sin(M_2_PI * k_s[(i + 2) % 3] / grid.cells[(i + 2) % 3])) /
                                      std::complex<double>(4.0 * grid.geom_size[i] / grid.cells[i]), 0.0);
            }
            break;
        default:
            throw std::runtime_error("Invalid spectral_derivative_ID value.");
    }
    return freq_derivative;
}

void Spectral::generate_plans(double* field_real_data,
                              std::complex<double>* field_fourier_data, 
                              int size, ptrdiff_t cells_fftw_reversed[3], int fftw_planner_flag,
                              fftw_plan &plan_forth, 
                              fftw_plan &plan_back){
    plan_forth = fftw_mpi_plan_many_dft_r2c(3, cells_fftw_reversed, size,
                                                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                                field_real_data,
                                                reinterpret_cast<fftw_complex*>(field_fourier_data),
                                                PETSC_COMM_WORLD, fftw_planner_flag | FFTW_MPI_TRANSPOSED_OUT);

    plan_back = fftw_mpi_plan_many_dft_c2r(3, cells_fftw_reversed, size,
                                                FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
                                                reinterpret_cast<fftw_complex*>(field_fourier_data),
                                                field_real_data,
                                                PETSC_COMM_WORLD, fftw_planner_flag | FFTW_MPI_TRANSPOSED_IN);
}

void Spectral::update_coords(Eigen::Tensor<double, 5> &F) {

    Spectral::tensorField_real->slice(Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
        Eigen::array<Eigen::Index, 5>({3, 3, grid.cells[0], grid.cells[1], grid.cells2})).device(Eigen::DefaultDevice{}) = F;
    Spectral::tensorField_real->slice(Eigen::array<Eigen::Index, 5>({0, 0, grid.cells[0], 0, 0}),
        Eigen::array<Eigen::Index, 5>({3, 3, cells0_reduced*2-grid.cells[0], grid.cells[1], grid.cells2})).setConstant(0);
    fftw_mpi_execute_dft_r2c(plan_tensor_forth, tensorField_real->data(), tensorField_fourier_fftw);

    // Average F
    Eigen::Tensor<double, 2> Favg(3, 3);
    if (cells1_offset_tensor == 0) {
        auto sliced_tensor = tensorField_fourier->slice(Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
                                                        Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1}));
        Favg = sliced_tensor.real().reshape(Eigen::array<Eigen::Index, 2>({3, 3})) * Spectral::wgt;
    }

    // Integration in Fourier space to get fluctuations of cell center displacements
    for (int j = 0; j < cells1_tensor ; ++j) {
        for (int k = 0; k < grid.cells[2]; ++k) {
            for (int i = 0; i < cells0_reduced ; ++i) {
                std::array<int, 3> indices = {i, j + cells1_offset_tensor, k};
                if (std::any_of(indices.begin(), indices.end(), [](int x) { return x != 0; })) {
                    Eigen::Tensor<std::complex<double>, 2> tensor_slice = Spectral::tensorField_fourier->slice(Eigen::array<Eigen::Index, 5>({0, 0, i, k, j}),
                                            Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1})).reshape(Eigen::array<Eigen::Index, 2>({3, 3}));
                    Eigen::Tensor<std::complex<double>, 1>  xi2_slice = xi2nd.slice(Eigen::array<Eigen::Index, 4>({0, i, k, j}),
                        Eigen::array<Eigen::Index, 4>({3, 1, 1, 1})).reshape(Eigen::array<Eigen::Index, 1>({3}));
                    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
                    Eigen::Tensor<std::complex<double>, 1> result = tensor_slice.contract(xi2_slice, product_dims);
                    Eigen::Array<std::complex<double>, 3, 1> xi2_array;
                    for (Eigen::Index l = 0; l < 3; ++l)xi2_array(l) = -xi2_slice(l);
                    std::complex<double> denominator = (xi2_array.conjugate() * xi2_array).sum();
                    for (Eigen::Index l = 0; l < 3; ++l){
                        (*Spectral::vectorField_fourier)(l,i,k,j) = result(l)/denominator;
                    }
                } else {
                    Eigen::Tensor<std::complex<double>, 4> zero_tensor(3, 1, 1, 1);
                    zero_tensor.setConstant(std::complex<double>(0.0, 0.0));
                    Spectral::vectorField_fourier->slice(Eigen::array<Eigen::Index, 4>({0, i, k, j}),
                                                         Eigen::array<Eigen::Index, 4>({3, 1, 1, 1})).device(Eigen::DefaultDevice{}) = zero_tensor;
                }
            }
        }
    }
    fftw_mpi_execute_dft_c2r(Spectral::plan_vector_back, vectorField_fourier_fftw, vectorField_real->data());

    Eigen::Tensor<double, 4>u_tilde_p_padded(3,grid.cells[0],grid.cells[1],grid.cells2+2);
    u_tilde_p_padded.slice(Eigen::array<Eigen::Index, 4>({0, 0, 0, 1}),
                           Eigen::array<Eigen::Index, 4>({3, grid.cells[0], grid.cells[1], grid.cells2})) = 
    vectorField_real->slice(Eigen::array<Eigen::Index, 4>({0, 0, 0, 0}),
                            Eigen::array<Eigen::Index, 4>({3, grid.cells[0], grid.cells[1], grid.cells2}))*Spectral::wgt;

    // Pad cell center fluctuations along z-direction (needed when running MPI simulation)
    int c = 3 * grid.cells[0] * grid.cells[1]; // amount of data to transfer
    int rank_t = (grid.world_rank + 1) % grid.world_size;
    int rank_b = (grid.world_rank - 1 + grid.world_size) % grid.world_size;
    MPI_Request request[4];
    MPI_Status status[4];
    Eigen::array<Eigen::Index, 3> sub_dims = {3, grid.cells[0], grid.cells[1]};

    // Send bottom layer to process below
    Eigen::TensorMap<Eigen::Tensor<double, 3>> bottom_layer_send(u_tilde_p_padded.data() + 3 * grid.cells[0] * grid.cells[1], sub_dims);
    MPI_Isend(bottom_layer_send.data(), c, MPI_DOUBLE, rank_b, 0, MPI_COMM_WORLD, &request[0]);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> top_layer_recv(u_tilde_p_padded.data() + 3 * grid.cells[0] * grid.cells[1] * (grid.cells2 + 1), sub_dims);
    MPI_Irecv(top_layer_recv.data(), c, MPI_DOUBLE, rank_t, 0, MPI_COMM_WORLD, &request[1]);

    // Send top layer to process above
    Eigen::TensorMap<Eigen::Tensor<double, 3>> top_layer_send(u_tilde_p_padded.data() + 3 * grid.cells[0] * grid.cells[1] * grid.cells2, sub_dims);
    MPI_Isend(top_layer_send.data(), c, MPI_DOUBLE, rank_t, 1, MPI_COMM_WORLD, &request[2]);
    Eigen::TensorMap<Eigen::Tensor<double, 3>> bot_layer_recv(u_tilde_p_padded.data(), sub_dims);
    MPI_Irecv(bot_layer_recv.data(), c, MPI_DOUBLE, rank_b, 1, MPI_COMM_WORLD, &request[3]);

    MPI_Waitall(4, request, status);
    if (std::any_of(status, status + 4, [](MPI_Status s) { return s.MPI_ERROR != 0; })) throw std::runtime_error("MPI error");

    // Calculate cell center/point positions
    int neighbor[8][3] = {
        {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
        {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
    };
    Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
    Eigen::Array<double, 3, 1> step(grid.geom_size[0] / grid.cells[0], grid.geom_size[1] / grid.cells[1], grid.geom_size[2] / grid.cells[2]);
    std::unique_ptr<Eigen::Tensor<double, 4>> x_n(new Eigen::Tensor<double, 4>(3, grid.cells[0] + 1, grid.cells[1] + 1, grid.cells2 + 1));
    x_n->setZero();   
    for (int j = 0; j <= grid.cells[1]; j++) {
        for (int k = 0; k <= grid.cells2; k++) {
            for (int i = 0; i <= grid.cells[0]; i++) {
                std::array<double, 3> pos_data = {i * step[0], 
                                                  j * step[1], 
                                                  (k + grid.cells2_offset) * step[2]};
                Eigen::TensorMap<Eigen::Tensor<double, 1>> pos_tensor(pos_data.data(), 3);
                Eigen::Tensor<double, 1> result = Favg.contract(pos_tensor, product_dims);
                for (int p = 0; p < 3; ++p) (*x_n)(p,i,j,k) = result(p);
                for (int n = 0; n < 8; ++n) {
                    int me[3] = {i+neighbor[n][0],j+neighbor[n][1],k+neighbor[n][2]};
                    Eigen::Tensor<double, 1> ut_padded_slice = u_tilde_p_padded.chip(me[2], 3)
                                                                               .chip(grid.modulo((me[1]-1), grid.cells[1]), 2)
                                                                               .chip(grid.modulo((me[0]-1), grid.cells[0]), 1);
                    for (int p = 0; p < 3; ++p) (*x_n)(p,i,j,k) += ut_padded_slice(p)* 0.125;
                }
            }
        }
    }

    // Calculate cell center/point positions
    std::unique_ptr<Eigen::Tensor<double, 4>> x_p(new Eigen::Tensor<double, 4>(3, grid.cells[0], grid.cells[1], grid.cells2));
    for (int k = 0; k < grid.cells2; ++k) {
        for (int j = 0; j < grid.cells[1]; ++j) {
            for (int i = 0; i < grid.cells[0]; ++i) {
                std::array<double, 3> pos_data = {(i + 0.5) * step[0], 
                                                  (j + 0.5) * step[1], 
                                                  (k + 0.5 + grid.cells2_offset)*step[2]};
                Eigen::TensorMap<Eigen::Tensor<double, 1>> pos_tensor(pos_data.data(), 3);
                Eigen::Tensor<double, 1> result = Favg.contract(pos_tensor, product_dims);
                for (int p = 0; p < 3; ++p) (*x_p)(p, i, j, k) += result(p);
            }
        }
    }
    Eigen::Tensor<double, 2> reshaped_x_n = x_n->reshape(Eigen::array<Eigen::Index, 2>({3, (grid.cells[0] + 1) * (grid.cells[1] + 1) * (grid.cells2 + 1)}));
    Spectral::grid.discretization_.set_node_coords(&reshaped_x_n);
    Eigen::Tensor<double, 2> reshaped_x_p = x_p->reshape(Eigen::array<Eigen::Index, 2>({3, grid.cells[0] * grid.cells[1] * grid.cells2}));
    Spectral::grid.discretization_.set_ip_coords(&reshaped_x_p);
}

