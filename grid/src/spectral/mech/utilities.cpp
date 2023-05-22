#include "spectral/mech/utilities.h"
#include "helper.h"

#include <mpi.h>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <fftw3-mpi.h>
#include <petsc.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <tensor_operations.h>

#include <complex>
#include <algorithm>
#include <cmath>

void Utilities::init_utilities(){
  std::cout << "\n <<<+-  spectral mech init  -+>>>" << std::endl;

  if (config.num_grid.divergence_correction == 1) {
    double* min_geom_size = std::min_element(grid.geom_size.begin(), grid.geom_size.end());
    double* max_geom_size = std::max_element(grid.geom_size.begin(), grid.geom_size.end());
    for (int i = 0; i < 3; ++i) {
      if (i != std::distance(grid.geom_size.begin(), min_geom_size) && 
          i != std::distance(grid.geom_size.begin(), max_geom_size)) {
        for (int j = 0; j < 3; ++j)
          grid.scaled_geom_size[j] = grid.geom_size[j] / grid.geom_size[i];
      }
    }
  } else if (config.num_grid.divergence_correction == 2) {
    std::array<double, 3> normalized_geom_size = {grid.geom_size[0]/grid.cells[0], grid.geom_size[1]/grid.cells[1], grid.geom_size[2]/grid.cells[2]};
    double* min_normalized_geom_size = std::min_element(normalized_geom_size.begin(), normalized_geom_size.end());
    double* max_normalized_geom_size = std::max_element(normalized_geom_size.begin(), normalized_geom_size.end());
    for (int i = 0; i < 3; ++i) {
      if (i != std::distance(grid.geom_size.begin(), min_normalized_geom_size) && 
          i != std::distance(grid.geom_size.begin(), max_normalized_geom_size)) {
        for (int j = 0; j < 3; ++j)
          grid.scaled_geom_size[j] = grid.geom_size[j] / grid.geom_size[i] * grid.cells[i];
      }
    }
  } else {
      grid.scaled_geom_size = grid.geom_size;
  }

  if (config.num_grid.memory_efficient) {
    gamma_hat.resize(3, 3, 3, 3, 1, 1, 1);
  } else {
    gamma_hat.resize(3,3,3,3,grid.cells0_reduced,grid.cells[2],grid.cells1_tensor);
      gamma_hat.setZero();
  }
}

void Utilities::update_coords(Eigen::Tensor<double, 5> &F, Eigen::Tensor<double, 2>& x_n_, Eigen::Tensor<double, 2>& x_p_) {
  spectral.tensorField_real->slice(Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
      Eigen::array<Eigen::Index, 5>({3, 3, grid.cells[0], grid.cells[1], grid.cells2})).device(Eigen::DefaultDevice{}) = F;
  spectral.tensorField_real->slice(Eigen::array<Eigen::Index, 5>({0, 0, grid.cells[0], 0, 0}),
      Eigen::array<Eigen::Index, 5>({3, 3, grid.cells0_reduced*2-grid.cells[0], grid.cells[1], grid.cells2})).setConstant(0);
  fftw_mpi_execute_dft_r2c(spectral.plan_tensor_forth, spectral.tensorField_real->data(), spectral.tensorField_fourier_fftw);

  // Average F
  Eigen::Tensor<double, 2> Favg(3, 3);
  if (grid.cells1_offset_tensor == 0) {
    auto sliced_tensor = spectral.tensorField_fourier->slice(Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
                                                             Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1}));
    Favg = sliced_tensor.real().reshape(Eigen::array<Eigen::Index, 2>({3, 3})) * spectral.wgt;
  }

  // Integration in Fourier space to get fluctuations of cell center displacements
  for (int j = 0; j < grid.cells1_tensor ; ++j) {
    for (int k = 0; k < grid.cells[2]; ++k) {
      for (int i = 0; i < grid.cells0_reduced ; ++i) {
        std::array<int, 3> indices = {i, j + grid.cells1_offset_tensor, k};
        if (std::any_of(indices.begin(), indices.end(), [](int x) { return x != 0; })) {
          Eigen::Tensor<std::complex<double>, 2> tensor_slice = spectral.tensorField_fourier->slice(Eigen::array<Eigen::Index, 5>({0, 0, i, k, j}),
                                  Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1})).reshape(Eigen::array<Eigen::Index, 2>({3, 3}));
          Eigen::Tensor<std::complex<double>, 1>  xi2_slice = spectral.xi2nd.slice(Eigen::array<Eigen::Index, 4>({0, i, k, j}),
              Eigen::array<Eigen::Index, 4>({3, 1, 1, 1})).reshape(Eigen::array<Eigen::Index, 1>({3}));
          Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
          Eigen::Tensor<std::complex<double>, 1> result = tensor_slice.contract(xi2_slice, product_dims);
          Eigen::Array<std::complex<double>, 3, 1> xi2_array;
          for (Eigen::Index l = 0; l < 3; ++l) xi2_array(l) = -xi2_slice(l);
          std::complex<double> denominator = (xi2_array.conjugate() * xi2_array).sum();
          for (Eigen::Index l = 0; l < 3; ++l) (*spectral.vectorField_fourier)(l,i,k,j) = result(l)/denominator;
        } else {
          Eigen::Tensor<std::complex<double>, 4> zero_tensor(3, 1, 1, 1);
          spectral.vectorField_fourier->slice(Eigen::array<Eigen::Index, 4>({0, i, k, j}),
                                              Eigen::array<Eigen::Index, 4>({3, 1, 1, 1})).setZero();
        }
      }
    }
  }
  fftw_mpi_execute_dft_c2r (spectral.plan_vector_back, 
                            spectral.vectorField_fourier_fftw, 
                            spectral.vectorField_real->data());

  Eigen::Tensor<double, 4> u_tilde_p_padded(3,grid.cells[0],grid.cells[1],grid.cells2+2);
  u_tilde_p_padded.slice(Eigen::array<Eigen::Index, 4>({0, 0, 0, 1}),
                          Eigen::array<Eigen::Index, 4>({3, grid.cells[0], grid.cells[1], grid.cells2})) = 
  spectral.vectorField_real->slice (Eigen::array<Eigen::Index, 4>({0, 0, 0, 0}),
                                    Eigen::array<Eigen::Index, 4>({3, grid.cells[0], grid.cells[1], grid.cells2}))*spectral.wgt;

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

// void Utilities::constitutive_response(Eigen::Tensor<double, 5> &P, 
//                                      Eigen::Tensor<double, 2> &P_av, 
//                                      Eigen::Tensor<double, 4> &C_volAvg, 
//                                      Eigen::Tensor<double, 4> &C_minMaxAvg,
//                                      Eigen::Tensor<double, 5> &F,
//                                      double Delta_t) {

//   // Reshape F
// //   std::cout << F.dimension(2);
// //   Eigen::Tensor<double, 3> homogenization_F;
// //   homogenization_F = F.reshape(Eigen::array<int, 3>({3, 3, grid.cells[0] * grid.cells[1] * grid.cells2}));

//   // Call the homogenization functions
// //   homogenization_mechanical_response(Delta_t, 1, product_cells[0] * product_cells[1]);
// //   if (!terminallyIll)
// //     homogenization_thermal_response(Delta_t, 1, product_cells[0] * product_cells[1]);
// //   if (!terminallyIll)
// //     homogenization_mechanical_response2(Delta_t, {1, 1}, {1, product_cells[0] * product_cells[1]});

// //   // Compute P and P_av
// //   Eigen::Tensor<double, 5> homogenization_P = /* ... */;
// //   P = homogenization_P.reshape(F.dimensions());
// //   P_av = (P.sum(Eigen::array<Eigen::Index, 1>({2})).sum(Eigen::array<Eigen::Index, 1>({3})).sum(Eigen::array<Eigen::Index, 1>({4})) * wgt).eval();

// //   // MPI Communication
// //   MPI_Allreduce(MPI_IN_PLACE, P_av.data(), 9, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &err_MPI);
// //   if (err_MPI != MPI_SUCCESS)
// //     throw std::runtime_error("MPI error");

// //   // Rotation of load frame
// //   if (rotation_BC.is_present()) {
// //     if (!rotation_BC.asQuaternion().isApprox(Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0))) {
// //       std::cout << "Piola--Kirchhoff stress (lab) / MPa = " << P_av.transpose() * 1.e-6 << std::endl;
// //     }
// //     P_av = rotation_BC.rotate(P_av);
// //   }

// //   std::cout << "Piola--Kirchhoff stress       / MPa = " << P_av.transpose() * 1.e-6 << std::endl;

// //   // Find dPdF_min and dPdF_max
// //   // ...

// //   // Compute C_minMaxAvg
// //   C_minMaxAvg = 0.5 * (dPdF_max + dPdF_min);

// //   // Compute C_volAvg
// //   C_volAvg = homogenization_dPdF.sum(Eigen::array<Eigen::Index, 1>({4})) * wgt;

// //   // MPI Communication
// //   MPI_Allreduce(MPI_IN_PLACE, C_volAvg.data(), 81, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD)
// }

void Utilities::update_gamma(Eigen::Tensor<double, 4> &C) {
  C_ref = C / spectral.wgt;
  if (!config.num_grid.memory_efficient){
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
                  xiDyad_cmplx(l, m) =  std::conj(-spectral.xi1st(l, i, k, j - grid.cells1_offset_tensor)) * 
                                        spectral.xi1st(m, i, k, j - grid.cells1_offset_tensor);
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
}

void Utilities::forward_field(double delta_t, 
                            Eigen::Tensor<double, 5> &field_last_inc, 
                            Eigen::Tensor<double, 5> &rate, 
                            Eigen::Tensor<double, 5> &forwarded_field,
                            Eigen::Matrix<double, 3, 3>* aim) {


  forwarded_field = field_last_inc + rate*delta_t;
  if (aim != nullptr){
    Eigen::array<int, 3> reduce_dims = {2, 3, 4};
    Eigen::Matrix<double, 3, 3> field_diff;
    field_diff.setZero();
    Eigen::Tensor<double, 3> field_slice;
    for (int l = 0; l < 3; ++l) {
      for (int m = 0; m < 3; ++m) {
        field_slice = forwarded_field.chip(m, 1).chip(l, 0).slice(
                        Eigen::array<Eigen::Index, 3>({0, 0, 0}),
                        Eigen::array<Eigen::Index, 3>({grid.cells[0], grid.cells[1], grid.cells2}));
        field_diff(l, m) = tensor_sum(field_slice) * spectral.wgt;
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

void Utilities::calculate_masked_compliance( Eigen::Tensor<double, 4> &C,
                                            std::array<double, 4> &rot_bc_q,
                                            const Eigen::Matrix<bool, 3, 3> &mask_stress,
                                            Eigen::Tensor<double, 4> &masked_compliance) {

  std::array<bool, 9> mask_stress_vector;
  for (int col = 0; col < mask_stress.cols(); ++col) {
    for (int row = 0; row < mask_stress.rows(); ++row) {
      mask_stress_vector[col * mask_stress.rows() + row] = !mask_stress(row, col);
    }
  }
  int size_reduced = std::count(mask_stress_vector.begin(), mask_stress_vector.end(), true);

  Eigen::Matrix<double, 9, 9> temp99_real;
  if (size_reduced > 0) {
    Eigen::Tensor<double, 4> rotated;
    rotated.resize(3, 3, 3, 3);
    f_rotate_tensor4(rot_bc_q.data(), C.data(), rotated.data());
    f_math_3333to99(rotated.data(), temp99_real.data());
    Eigen::Matrix<bool, 9, 9> mask;
    for (int i = 0; i < 9; ++i) {
      for (int j = 0; j < 9; ++j) {
        mask(i, j) = mask_stress_vector[i] && mask_stress_vector[j];
      }
    }
    Eigen::MatrixXd c_reduced(size_reduced, size_reduced);
    Eigen::MatrixXd s_reduced(size_reduced, size_reduced);
    int idx = 0;
    for (int i = 0; i < 9; ++i) {
      for (int j = 0; j < 9; ++j) {
        if (mask(i, j)) {
          c_reduced(idx / size_reduced, idx % size_reduced) = temp99_real(i, j);
          ++idx;    
        }
      }
    }
    int errmatinv = 0;
    f_math_invert(s_reduced.data(), &errmatinv, c_reduced.data(), &size_reduced);
    if (errmatinv) {
      std::cerr << "Matrix inversion error: " << std::endl;
      std::cerr << c_reduced << std::endl;
      exit(1);
    }
    temp99_real.setZero();
    idx = 0;
    for (int i = 0; i < 9; ++i) {
      for (int j = 0; j < 9; ++j) {
        if (mask(i, j)) {
          temp99_real(i, j) = s_reduced(idx / size_reduced, idx % size_reduced);
          ++idx;
        }
      }
    }
  } else {
    temp99_real.setZero();
  }
  f_math_99to3333(temp99_real.data(), masked_compliance.data());
}

double Utilities::calculate_divergence_rms(const Eigen::Tensor<double, 5>& tensor_field) {

  spectral.tensorField_real->slice(
    Eigen::array<Eigen::Index, 5>({0, 0, grid.cells[0], 0, 0}), 
    Eigen::array<Eigen::Index, 5>({3, 3, grid.cells0_reduced * 2 - grid.cells[0], grid.cells[1], grid.cells2}))
    .setZero();
  spectral.tensorField_real->slice(
    Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
    Eigen::array<Eigen::Index, 5>({3, 3, grid.cells[0], grid.cells[1], grid.cells2})) = tensor_field;
  fftw_mpi_execute_dft_r2c(spectral.plan_tensor_forth, spectral.tensorField_real->data(), spectral.tensorField_fourier_fftw);

  Eigen::Vector3cd rescaled_geom;
  for (int i = 0; i < 3; ++i) rescaled_geom[i] = std::complex<double>(grid.geom_size[i] / grid.scaled_geom_size[i], 0);
  double rms = 0;
  Eigen::Tensor<std::complex<double>, 2> tensorField_fourier_slice_; //chipped and sliced view on data
  Eigen::Tensor<std::complex<double>, 2> tensorField_fourier_slice; //tensor with actual memory layout of chipped/sliced data
  Eigen::Tensor<std::complex<double>, 1> xi1st_slice_;
  Eigen::Tensor<std::complex<double>, 1> xi1st_slice;
  Eigen::Vector3cd product;
  for (int j = 0; j < grid.cells2; ++j) {
    for (int k = 0; k < grid.cells[2]; ++k) {
      for (int i = 1; i < grid.cells0_reduced - 1; ++i) {
        tensorField_fourier_slice_ =  (*spectral.tensorField_fourier).chip(j, 4).chip(k,3).chip(i,2)
              .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
        tensorField_fourier_slice = tensorField_fourier_slice; //create copy to generate continuous storage of data
        Eigen::Map<const Eigen::Matrix<std::complex<double>, 3, 3>> tensorField_fourier_mat(tensorField_fourier_slice_.data());
        xi1st_slice = -spectral.xi1st.chip(j, 3).chip(k, 2).chip(i, 1);
        xi1st_slice_ = xi1st_slice; //create copy to generate continuous storage of data
        Eigen::Map<Eigen::Vector3cd> xi1st_vec(xi1st_slice_.data());
        product = tensorField_fourier_mat * ((xi1st_vec.conjugate().array() * rescaled_geom.array()).matrix());
        // std::cout << "scaled " << rescaled_geom[0] << " " << rescaled_geom[1] << " " << rescaled_geom[2] << " " << std::endl;
        // std::cout << "xi1st_slice "<< std::setprecision(17) << -spectral.xi1st.chip(j, 3).chip(k,2).chip(i,1) << std::endl;
        // std::cout << "conjg xi1st_slice "<< std::setprecision(17) << xi1st_vec.conjugate() << std::endl;
        // std::cout << "conjg xi1st_slice scaled"<< std::setprecision(17) << xi1st_vec.conjugate().array() * rescaled_geom.array() << std::endl;
        // // std::cout << "conjg xi1st_slice scaled"<< std::setprecision(17) << result << std::endl;
        // std::cout << "fourier_slice "<< std::setprecision(17) << tensorField_fourier_mat << std::endl;
        // std::cout << "matmul "<< std::setprecision(17) << product  << std::endl;
        // std::cout << "power " << product.real().array().square() << std::endl;
        // std::cout << "power i" << product.imag().array().square() << std::endl;
        // std::cout << "sum " << product.real().array().square().sum() << std::endl;
        // std::cout << "sum i " << product.imag().array().square().sum() << std::endl;
        // sum with c++ eigen produces a slightly different result than the inbuilt fortran library 
        // (c++: 8776197.1903837956, fortran: 8776197.1903837975)
        rms += 2*(product.real().array().square().sum() + product.imag().array().square().sum());
        }
        std::vector<int> indeces = {0, grid.cells0_reduced-1};
        for (int i : indeces) {
          tensorField_fourier_slice_ =  (*spectral.tensorField_fourier).chip(j, 4).chip(k,3).chip(i,2)
                .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
          tensorField_fourier_slice = tensorField_fourier_slice; //create copy to generate continuous storage of data
          Eigen::Map<const Eigen::Matrix<std::complex<double>, 3, 3>> tensorField_fourier_mat(tensorField_fourier_slice_.data());
          xi1st_slice = -spectral.xi1st.chip(j, 3).chip(k, 2).chip(i, 1);
          xi1st_slice_ = xi1st_slice; //create copy to generate continuous storage of data
          Eigen::Map<Eigen::Vector3cd> xi1st_vec(xi1st_slice_.data());
          product = tensorField_fourier_mat * ((xi1st_vec.conjugate().array() * rescaled_geom.array()).matrix());
          rms += (product.real().array().square().sum() + product.imag().array().square().sum());
        }
      }
  }
  MPI_Allreduce(MPI_IN_PLACE, &rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  rms = sqrt(rms) * spectral.wgt;
  if (grid.cells[0] == 1) rms = rms/2;
  return rms;
}

Eigen::Tensor<double, 5> Utilities::gamma_convolution(Eigen::Tensor<double, 5> &field, Eigen::Tensor<double, 2> &field_aim){

  spectral.tensorField_real->slice(
    Eigen::array<Eigen::Index, 5>({0, 0, grid.cells[0], 0, 0}), 
    Eigen::array<Eigen::Index, 5>({3, 3, grid.cells0_reduced, grid.cells[1], grid.cells2})).setConstant(0);
  spectral.tensorField_real->slice(
    Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}), 
    Eigen::array<Eigen::Index, 5>({3, 3, grid.cells[0], grid.cells[1], grid.cells2})) = field;

  fftw_mpi_execute_dft_r2c(spectral.plan_tensor_forth, spectral.tensorField_real->data(), spectral.tensorField_fourier_fftw);

  Eigen::Matrix<std::complex<double>, 3, 3> temp33_cmplx;
  if (config.num_grid.memory_efficient) {
    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 6> A_inv;
    Eigen::Matrix<std::complex<double>, 3, 3> xiDyad_cmplx;
    #pragma omp parallel for private(l, m, n, o, temp33_cmplx, xiDyad_cmplx, A, A_inv, err, gamma_hat)
    for (int j = 0; j < grid.cells1_tensor; ++j) {
      for (int k = 0; k < grid.cells[2]; ++k) {
        for (int i = 0; i < grid.cells0_reduced; ++i) {
          if (!(i == 0 && j + grid.cells1_offset_tensor == 0 && k == 0)) { // singular point at xi=(0.0,0.0,0.0) i.e. i=j=k=1
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    xiDyad_cmplx(l, m) = std::conj(-spectral.xi1st(l, i, k, j)) * spectral.xi1st(m, i, k, j);
                }
            }
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                    for (int n = 0; n < 3; ++n) {
                        for (int o = 0; o < 3; ++o) {
                            temp33_cmplx(l, m) += std::complex<double>(C_ref(l, n, m, o)) * xiDyad_cmplx(n, o);
                        }
                    }
                }
            }
            A.block(0, 0, 3, 3) = temp33_cmplx.real();
            A.block(3, 3, 3, 3) = temp33_cmplx.real();
            A.block(0, 3, 3, 3) = temp33_cmplx.imag();
            A.block(3, 0, 3, 3) = -temp33_cmplx.imag();
            // TODO: if det(A(1:3,1:3)> ...)
            spectral.tensorField_fourier->slice(Eigen::array<Eigen::Index, 5>({0, 0, i, k, j}),
                                                Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1})).setConstant(std::complex<double>(0,0));
          }
        }
      }
    }
  } else {
    Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> res;
    for (int j = 0; j < grid.cells1_tensor; ++j) {
      for (int k = 0; k < grid.cells[2]; ++k) {
        for (int i = 0; i < grid.cells0_reduced; ++i) {
          for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {
              Eigen::Tensor<std::complex<double>, 2> gamma_hat_slice =  gamma_hat.chip(j, 6).chip(k,5).chip(i,4).chip(m,1).chip(l,0)
                              .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
              Eigen::Tensor<std::complex<double>, 2> gamma_hat_slice_ = gamma_hat_slice; //create copy to generate continuous storage of data
              Eigen::Map<const Eigen::Matrix<std::complex<double>, 3, 3>> gamma_hat_slice_mat(gamma_hat_slice_.data());
              Eigen::Tensor<std::complex<double>, 2> tensorField_fourier_slice = (*spectral.tensorField_fourier).chip(j, 4).chip(k,3).chip(i,2)
                              .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
              Eigen::Tensor<std::complex<double>, 2> tensorField_fourier_slice_ = tensorField_fourier_slice; //create copy to generate continuous storage of data
              Eigen::Map<const Eigen::Matrix<std::complex<double>, 3, 3>> tensorField_fourier_slice_mat(tensorField_fourier_slice_.data());
              res = gamma_hat_slice_mat * tensorField_fourier_slice_mat;
              temp33_cmplx(l, m) = tensor_sum(res);
            }
          }
          for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {
              (*spectral.tensorField_fourier)(l, m, i, k, j) = temp33_cmplx(l, m);
            }
          }
        }
      }
    }
  }
  if (grid.cells2_offset==0) {
    for (int l = 0; l < 3; ++l) {
      for (int m = 0; m < 3; ++m) {
        (*spectral.tensorField_fourier)(l, m, 0, 0, 0) = std::complex<double>(field_aim(l, m), 0);
      }
    }
  }
  fftw_mpi_execute_dft_c2r(spectral.plan_tensor_back, spectral.tensorField_fourier_fftw, spectral.tensorField_real->data());
  return spectral.tensorField_real->slice(Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}), 
                                          Eigen::array<Eigen::Index, 5>({3, 3, grid.cells[0], grid.cells[1], grid.cells2}));
}



// Eigen::Tensor<double, 4> Utilities::calculate_scalar_gradient(const Eigen::Tensor<double, 3>& field) {
//     std::cout << "calling calculate_scalar_gradient" << std::endl;

//     Eigen::Tensor<double, 4> grad(3, grid.cells[0], grid.cells[1], grid.cells2);
//     // print_tensor("field", &field);
//     // print_tensor_map("scalarField_real0", *scalarField_real);
//     // // Zero out the extended part of scalarField_real
//     // scalarField_real->slice(Eigen::array<int, 3>({grid.cells[0], 0, 0}),
//     //                        Eigen::array<int, 3>({grid.cells0_reduced*2-grid.cells[0], grid.cells[1], grid.cells2}))
//     //     .setConstant(0);
//     // // Copy field to the first part of scalarField_real
//     // scalarField_real->slice(Eigen::array<int, 3>({0, 0, 0}),
//     //                        Eigen::array<int, 3>({grid.cells[0], grid.cells[1], grid.cells2})) = field;
//     // print_tensor_map("scalarField_real1", *scalarField_real);

//     // // Perform forward FFT
//     // fftw_mpi_execute_dft_r2c(plan_scalar_forth, scalarField_real->data(), scalarField_fourier_fftw);
//     // print_tensor_map("scalarField_real2", *scalarField_real);
//     // print_tensor_map("scalarField_fourier", *scalarField_fourier);

//     // // Multiply scalarField_fourier by xi1st
//     // for (int j = 0; j < grid.cells[1]; ++j) {
//     //     for (int k = 0; k < grid.cells[2]; ++k) {
//     //         for (int i = 0; i < grid.cells2; ++i) {
//     //             for (int l = 0; l < 3; ++l) {
//     //                 (*vectorField_fourier)(l, i, k, j) = (*scalarField_fourier)(i, k, j) * xi1st(l, i, k, j);
//     //             }
//     //         }
//     //     }
//     // }

//     // // // Perform backward FFT
//     // // fftw_mpi_execute_dft_c2r(planVectorBack, vectorField_fourier.data(), vectorField_real.data());

//     // // // Multiply vectorField_real by wgt and extract the relevant portion
//     // // Eigen::Tensor<double, 4> grad(3, grid.cells[0], grid.cells[1], grid.cells[2]);
//     // // grad = vectorField_real.slice(Eigen::array<int, 4>({0, 0, 0, 0}),
//     // //                               Eigen::array<int, 4>({3, grid.cells[0], grid.cells[1], grid.cells[2]})) *
//     // //        wgt;

//     return grad;
// }