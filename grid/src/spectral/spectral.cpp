#include "spectral/spectral.h"
#include <mpi.h>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>

#include <complex>
#include <vector>
#include <algorithm>
#include <fftw3-mpi.h>
#include <petsc.h>
#include <stdexcept>
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
  grid.cells0_reduced = grid.cells[0] / 2 + 1;
  wgt = std::pow(grid.cells[0] * grid.cells[1] * grid.cells[2], -1);
    //get num vairables 177-210
    spectral_derivative_ID = DERIVATIVE_CONTINUOUS_ID;
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

  // std::cout << "hhe 3 " << grid.cells0_reduced << " " << grid.cells[2] << " " <<  grid.cells1_tensor << std::endl;
  // exit(0);
  xi1st.resize(3, grid.cells0_reduced, grid.cells[2], grid.cells1_tensor);
    xi1st.setConstant(std::complex<double>(0,0));
  xi2nd.resize(3, grid.cells0_reduced, grid.cells[2], grid.cells1_tensor);
    xi2nd.setConstant(std::complex<double>(0,0));

  std::array<int, 3> k_s;
  std::array<std::complex<double>, 3> freq_derivative;
  // use lambda function for get_freq_derivative
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
    // if (num.memory_efficient) {
    //     gamma_hat.resize(3, 3, 3, 3, 1, 1, 1);
    // } else {
    //     gamma_hat.resize(3, 3, 3, 3, cells1Red, cells[2], cells2);
    // }
  gamma_hat.resize(3,3,3,3,grid.cells0_reduced,grid.cells[2],grid.cells1_tensor);
    gamma_hat.setZero();
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
    switch (spectral_derivative_ID) {
        case DERIVATIVE_CONTINUOUS_ID:
            for (int i = 0; i < 3; ++i) {
                freq_derivative[i] = std::complex<double>(0.0, TAU * k_s[i] / grid.geom_size[i]);
            }
            break;
        case DERIVATIVE_CENTRAL_DIFF_ID:
            for (int i = 0; i < 3; ++i) {
                freq_derivative[i] = std::complex<double>(0.0, sin(TAU * k_s[i] / grid.cells[i])) /
                                     std::complex<double>(2.0 * grid.geom_size[i] / grid.cells[i], 0.0);
            }
            break;
        case DERIVATIVE_FWBW_DIFF_ID:
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

void Spectral::update_coords(Eigen::Tensor<double, 5> &F, Eigen::Tensor<double, 2>& x_n_, Eigen::Tensor<double, 2>& x_p_) {
  Spectral::tensorField_real->slice(Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
      Eigen::array<Eigen::Index, 5>({3, 3, grid.cells[0], grid.cells[1], grid.cells2})).device(Eigen::DefaultDevice{}) = F;
  Spectral::tensorField_real->slice(Eigen::array<Eigen::Index, 5>({0, 0, grid.cells[0], 0, 0}),
      Eigen::array<Eigen::Index, 5>({3, 3, grid.cells0_reduced*2-grid.cells[0], grid.cells[1], grid.cells2})).setConstant(0);
  fftw_mpi_execute_dft_r2c(plan_tensor_forth, tensorField_real->data(), tensorField_fourier_fftw);

  // Average F
  Eigen::Tensor<double, 2> Favg(3, 3);
  if (grid.cells1_offset_tensor == 0) {
    auto sliced_tensor = tensorField_fourier->slice(Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
                                                    Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1}));
    Favg = sliced_tensor.real().reshape(Eigen::array<Eigen::Index, 2>({3, 3})) * wgt;
  }

  // Integration in Fourier space to get fluctuations of cell center displacements
  for (int j = 0; j < grid.cells1_tensor ; ++j) {
    for (int k = 0; k < grid.cells[2]; ++k) {
      for (int i = 0; i < grid.cells0_reduced ; ++i) {
        std::array<int, 3> indices = {i, j + grid.cells1_offset_tensor, k};
        if (std::any_of(indices.begin(), indices.end(), [](int x) { return x != 0; })) {
          Eigen::Tensor<std::complex<double>, 2> tensor_slice = Spectral::tensorField_fourier->slice(Eigen::array<Eigen::Index, 5>({0, 0, i, k, j}),
                                  Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1})).reshape(Eigen::array<Eigen::Index, 2>({3, 3}));
          Eigen::Tensor<std::complex<double>, 1>  xi2_slice = xi2nd.slice(Eigen::array<Eigen::Index, 4>({0, i, k, j}),
              Eigen::array<Eigen::Index, 4>({3, 1, 1, 1})).reshape(Eigen::array<Eigen::Index, 1>({3}));
          Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
          Eigen::Tensor<std::complex<double>, 1> result = tensor_slice.contract(xi2_slice, product_dims);
          Eigen::Array<std::complex<double>, 3, 1> xi2_array;
          for (Eigen::Index l = 0; l < 3; ++l) xi2_array(l) = -xi2_slice(l);
          std::complex<double> denominator = (xi2_array.conjugate() * xi2_array).sum();
          for (Eigen::Index l = 0; l < 3; ++l) (*Spectral::vectorField_fourier)(l,i,k,j) = result(l)/denominator;
        } else {
          Eigen::Tensor<std::complex<double>, 4> zero_tensor(3, 1, 1, 1);
          zero_tensor.setConstant(std::complex<double>(0.0, 0.0));
          Spectral::vectorField_fourier->slice( Eigen::array<Eigen::Index, 4>({0, i, k, j}),
                                                Eigen::array<Eigen::Index, 4>({3, 1, 1, 1})).device(Eigen::DefaultDevice{}) = zero_tensor;
        }
      }
    }
  }
  fftw_mpi_execute_dft_c2r(Spectral::plan_vector_back, vectorField_fourier_fftw, vectorField_real->data());

  Eigen::Tensor<double, 4> u_tilde_p_padded(3,grid.cells[0],grid.cells[1],grid.cells2+2);
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

  x_n_.resize(3, (grid.cells[0] + 1) * (grid.cells[1] + 1) * (grid.cells2 + 1));
  x_n_.setConstant(0);
  Eigen::TensorMap<Eigen::Tensor<double, 4>> x_n(x_n_.data(), 
      Eigen::array<Eigen::Index, 4>({3, grid.cells[0] + 1, grid.cells[1] + 1, grid.cells2 + 1}));
  for (int j = 0; j <= grid.cells[1]; j++) {
      for (int k = 0; k <= grid.cells2; k++) {
          for (int i = 0; i <= grid.cells[0]; i++) {
              std::array<double, 3> pos_data = {i * step[0], 
                                                j * step[1], 
                                                (k + grid.cells2_offset) * step[2]};
              Eigen::TensorMap<Eigen::Tensor<double, 1>> pos_tensor(pos_data.data(), 3);
              Eigen::Tensor<double, 1> result = Favg.contract(pos_tensor, product_dims);
              for (int p = 0; p < 3; ++p) x_n(p,i,j,k) = result(p);
              for (int n = 0; n < 8; ++n) {
                  int me[3] = {i+neighbor[n][0],j+neighbor[n][1],k+neighbor[n][2]};
                  Eigen::Tensor<double, 1> ut_padded_slice = u_tilde_p_padded.chip(me[2], 3)
                                                                              .chip(grid.modulo((me[1]-1), grid.cells[1]), 2)
                                                                              .chip(grid.modulo((me[0]-1), grid.cells[0]), 1);
                  for (int p = 0; p < 3; ++p) x_n(p,i,j,k) += ut_padded_slice(p)* 0.125;
              }
          }
      }
  }

  // Calculate cell center/point positions
  x_p_.resize(3, grid.cells[0] * grid.cells[1] * grid.cells2);
  Eigen::TensorMap<Eigen::Tensor<double, 4>> x_p(x_p_.data(), 
      Eigen::array<Eigen::Index, 4>({3, grid.cells[0], grid.cells[1], grid.cells2}));
  for (int k = 0; k < grid.cells2; ++k) {
      for (int j = 0; j < grid.cells[1]; ++j) {
          for (int i = 0; i < grid.cells[0]; ++i) {
              std::array<double, 3> pos_data = {(i + 0.5) * step[0], 
                                                (j + 0.5) * step[1], 
                                                (k + 0.5 + grid.cells2_offset)*step[2]};
              Eigen::TensorMap<Eigen::Tensor<double, 1>> pos_tensor(pos_data.data(), 3);
              Eigen::Tensor<double, 1> result = Favg.contract(pos_tensor, product_dims);
              for (int p = 0; p < 3; ++p) x_p(p, i, j, k) += result(p);
          }
      }
  }
}

void Spectral::constitutive_response(Eigen::Tensor<double, 5> &P, 
                                     Eigen::Tensor<double, 2> &P_av, 
                                     Eigen::Tensor<double, 4> &C_volAvg, 
                                     Eigen::Tensor<double, 4> &C_minMaxAvg,
                                     Eigen::Tensor<double, 5> &F,
                                     double Delta_t) {

  // Reshape F
//   std::cout << F.dimension(2);
//   Eigen::Tensor<double, 3> homogenization_F;
//   homogenization_F = F.reshape(Eigen::array<int, 3>({3, 3, grid.cells[0] * grid.cells[1] * grid.cells2}));

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

void Spectral::update_gamma(Eigen::Tensor<double, 4> &C) {
  C_ref = C / wgt;
  gamma_hat.setConstant(std::complex<double>(0.0, 0.0));
  for (int j = grid.cells1_offset_tensor; j < grid.cells1_offset_tensor + grid.cells1_tensor; ++j) {
    for (int k = 0; k < grid.cells[2]; ++k) {
      for (int i = 0; i < grid.cells0_reduced; ++i) {
        if (i != 0 || j != 0 || k != 0) {
          Eigen::Matrix<std::complex<double>, 3, 3> xiDyad_cmplx;
          Eigen::TensorMap<Eigen::Tensor<const std::complex<double>, 2>> xiDyad_cmplx_map(xiDyad_cmplx.data(), 3, 3);
          Eigen::Matrix<std::complex<double>, 3, 3> temp33_cmplx;
          for (int l = 0; l < 3; ++l) {
              for (int m = 0; m < 3; ++m) {
                xiDyad_cmplx(l, m) = std::conj(-xi1st(l, i, k, j - grid.cells1_offset_tensor)) * xi1st(m, i, k, j - grid.cells1_offset_tensor);
              }
          }
          for (int l = 0; l < 3; ++l) {
              for (int m = 0; m < 3; ++m) {
                  temp33_cmplx(l, m) = 0; // use loops instead of contraction because of missing Tensor-Matrix interoperability in Eigen
                  for (int n = 0; n < 3; ++n) {
                      for (int o = 0; o < 3; ++o) {
                          temp33_cmplx(l, m) += std::complex<double>(C_ref(l, n, m, o), 0) * xiDyad_cmplx(n, o);
                      }
                  }
              }   
          }
          Eigen::Matrix<double, 6, 6> A;
          A.block<3, 3>(0, 0) = temp33_cmplx.real(); 
          A.block<3, 3>(3, 3) = temp33_cmplx.real();
          A.block<3, 3>(0, 3) = temp33_cmplx.imag(); 
          A.block<3, 3>(3, 0) = -temp33_cmplx.imag();
          if (std::abs(A.block<3, 3>(0, 0).determinant()) > 1e-16) {
            Eigen::Matrix<double, 6, 6> A_inv;
            A_inv = A.inverse();
            for (int i = 0; i < 3; ++i) {
              for (int j = 0; j < 3; ++j) {
                temp33_cmplx(i, j) = std::complex<double>(A_inv(i, j), A_inv(i + 3, j));
              }
            }
            for (int m = 0; m < 3; ++m) {
              for (int n = 0; n < 3; ++n) {
                  for (int o = 0; o < 3; ++o) {
              for (int l = 0; l < 3; ++l) 
                gamma_hat(l, m, n, o, i, k, j - grid.cells1_offset_tensor) = temp33_cmplx(l, n) * xiDyad_cmplx(o, m);
                }
              }
            }
          }
        }
      }
    }
  }
}

void Spectral::forward_field(double delta_t, 
                            Eigen::Tensor<double, 5> &field_last_inc, 
                            Eigen::Tensor<double, 5> &rate, 
                            Eigen::Tensor<double, 5> &forwarded_field,
                            Eigen::Matrix<double, 3, 3>* aim) {

    forwarded_field = field_last_inc + rate*delta_t;
    if (aim != nullptr){
        Eigen::array<int, 3> reduce_dims = {2, 3, 4};
        Eigen::Matrix<double, 3, 3> field_diff;
        field_diff.setZero();
        double sum;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                sum = 0;

                for (int k = 0; k < grid.cells[0]; ++k) {
                    for (int l = 0; l < grid.cells[1]; ++l) {
                        for (int m = 0; m < grid.cells2; ++m) {
                             sum += forwarded_field(i, j, k, l, m);
                        }
                    }
                }
                field_diff(i, j) = sum * wgt;
            }
        }
        int count = field_diff.size();
        MPI_Allreduce(MPI_IN_PLACE, field_diff.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        field_diff = field_diff - *aim;  
        for (int i = 0; i < grid.cells[0]; ++i) {
            for (int j = 0; j < grid.cells[1]; ++j) {
                for (int k = 0; k < grid.cells2; ++k) {
                    for (int m = 0; m < 3; ++m) {
                        for (int n = 0; n < 3; ++n) {
                            forwarded_field(m, n, i, j, k) -= field_diff(m, n);
                        }
                    }
                }
            }
        }
    }
}

}