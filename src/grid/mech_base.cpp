#include <iostream>
#include <array>
#include <string>
#include <limits>

#include "utilities_fortran.h"
#include "utilities_tensor.h"

#include "mech_base.h"

void MechBase::base_init(){
  std::cout << "\n <<<+-  mech base init  -+>>>" << std::endl;
  std::array<int, 3> indices = {0, 1, 2};
  if (config.numerics.fft.divergence_correction == Config::DIVERGENCE_CORRECTION_SIZE_ID) {  
    std::sort(indices.begin(), indices.end(), 
        [&](int a, int b) { return grid_.geom_size[a] < grid_.geom_size[b]; });
    for (int j = 0; j < 3; ++j) {
        grid_.scaled_geom_size[j] = grid_.geom_size[j] / grid_.geom_size[indices[1]];
    }
  } else if (config.numerics.fft.divergence_correction == Config::DIVERGENCE_CORRECTION_SIZE_GRID_ID) {
    std::array<double, 3> normalized_geom_size = {
        grid_.geom_size[0] / static_cast<double>(grid_.cells[0]), 
        grid_.geom_size[1] / static_cast<double>(grid_.cells[1]), 
        grid_.geom_size[2] / static_cast<double>(grid_.cells[2])
    };
    std::sort(indices.begin(), indices.end(), 
        [&](int a, int b) { return normalized_geom_size[a] < normalized_geom_size[b]; });
    for (int j = 0; j < 3; ++j) {
        grid_.scaled_geom_size[j] = grid_.geom_size[j] / grid_.geom_size[indices[1]] * grid_.cells[indices[1]];
    }
  } else {
    grid_.scaled_geom_size = grid_.geom_size;
  }

  if (config.numerics.fft.memory_efficient) {
    gamma_hat.resize(3, 3, 3, 3, 1, 1, 1);
  } else {
    gamma_hat.resize(3, 3, 3, 3, grid_.cells0_reduced,
                                 grid_.cells[2],
                                 grid_.cells1_tensor);
    gamma_hat.setZero();
  }
}

void MechBase::base_update_coords(TensorMap<Tensor<double, 5>>& F, Tensor<double, 4>& x_n, Tensor<double, 4>& x_p) {
  Eigen::Tensor<std::complex<double>, 5> tensorfield_fourier = spectral.tensorfield->forward(F);
  Tensor<double, 2> Favg(3, 3);
  if (grid_.cells1_offset_tensor == 0) {
    Tensor<std::complex<double>, 5> tensor_slice = tensorfield_fourier.slice(
      Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
      Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1}));
    Favg = tensor_slice.real().reshape(Eigen::array<Eigen::Index, 2>({3, 3})) * spectral.wgt;
  }

  // Integration in Fourier space to get fluctuations of cell center displacements
  for (int j = 0; j < grid_.cells2; ++j) {
    for (int k = 0; k < grid_.cells[3]; ++k) {
      for (int i = 0; i < grid_.cells0_reduced; ++i) {
        std::array<int, 3> indices = {i, j + grid_.cells1_offset_tensor, k};
        if (std::any_of(indices.begin(), indices.end(), [](int x) { return x != 0; })) {
          Tensor<complex<double>, 2> tensor_slice = tensorfield_fourier.slice(
            Eigen::array<Index, 5>({0, 0, i, k, j}),
            Eigen::array<Index, 5>({3, 3, 1, 1, 1})).reshape(Eigen::array<Eigen::Index, 2>({3, 3}));
          Tensor<complex<double>, 1> xi2_slice = spectral.xi2nd.slice(
            Eigen::array<Index, 4>({0, i, k, j}),
            Eigen::array<Index, 4>({3, 1, 1, 1})).reshape(Eigen::array<Eigen::Index, 1>({3}));
          Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
          Tensor<complex<double>, 1> result = tensor_slice.contract(xi2_slice, product_dims);
          Eigen::Array<complex<double>, 3, 1> xi2_array;
          for (Eigen::Index l = 0; l < 3; ++l) xi2_array(l) = -xi2_slice(l);
          complex<double> denominator = (xi2_array.conjugate() * xi2_array).sum();
          for (Eigen::Index l = 0; l < 3; ++l) (*spectral.vectorfield->field_fourier)(l,i,k,j) = result(l)/denominator;
        } else {
          spectral.vectorfield->field_fourier->slice(Eigen::array<Eigen::Index, 4>({0, i, k, j}),
                                                     Eigen::array<Eigen::Index, 4>({3, 1, 1, 1})).setConstant(std::complex<double>(0, 0));
        }
      }
    }
  }
  Tensor<double, 4> vectorfield_real = spectral.vectorfield->backward((spectral.vectorfield->field_fourier).get(), spectral.wgt);
  Tensor<double, 4> u_tilde_p_padded(3,grid_.cells[0],grid_.cells[1],grid_.cells2+2);
  u_tilde_p_padded.slice (Eigen::array<Eigen::Index, 4>({0, 0, 0, 1}),
                          Eigen::array<Eigen::Index, 4>({3, grid_.cells[0], grid_.cells[1], grid_.cells2})) = 
    vectorfield_real.slice (Eigen::array<Eigen::Index, 4>({0, 0, 0, 0}),
                            Eigen::array<Eigen::Index, 4>({3, grid_.cells[0], grid_.cells[1], grid_.cells2})) * spectral.wgt;

  // Pad cell center fluctuations along z-direction (needed when running MPI simulation)
  int c = 3 * grid_.cells[0] * grid_.cells[1]; // amount of data to transfer
  int rank_t = (grid_.world_rank + 1) % grid_.world_size;
  int rank_b = (grid_.world_rank - 1 + grid_.world_size) % grid_.world_size;
  MPI_Request request[4];
  MPI_Status status[4];
  Eigen::array<Eigen::Index, 3> sub_dims = {3, grid_.cells[0], grid_.cells[1]};

  // Send bottom layer to process below
  TensorMap<Tensor<double, 3>> bottom_layer_send(u_tilde_p_padded.data() + 3 * grid_.cells[0] * grid_.cells[1], sub_dims);
  MPI_Isend(bottom_layer_send.data(), c, MPI_DOUBLE, rank_b, 0, MPI_COMM_WORLD, &request[0]);
  TensorMap<Tensor<double, 3>> top_layer_recv(u_tilde_p_padded.data() + 3 * grid_.cells[0] * grid_.cells[1] * (grid_.cells2 + 1), sub_dims);
  MPI_Irecv(top_layer_recv.data(), c, MPI_DOUBLE, rank_t, 0, MPI_COMM_WORLD, &request[1]);

  // Send top layer to process above
  TensorMap<Tensor<double, 3>> top_layer_send(u_tilde_p_padded.data() + 3 * grid_.cells[0] * grid_.cells[1] * grid_.cells2, sub_dims);
  MPI_Isend(top_layer_send.data(), c, MPI_DOUBLE, rank_t, 1, MPI_COMM_WORLD, &request[2]);
  TensorMap<Tensor<double, 3>> bot_layer_recv(u_tilde_p_padded.data(), sub_dims);
  MPI_Irecv(bot_layer_recv.data(), c, MPI_DOUBLE, rank_b, 1, MPI_COMM_WORLD, &request[3]);

  MPI_Waitall(4, request, status);
  if (std::any_of(status, status + 4, [](MPI_Status s) { return s.MPI_ERROR != 0; })) throw std::runtime_error("MPI error");

  // Calculate cell center/point positions
  int neighbor[8][3] = {
      {0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
      {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}
  };
  Eigen::array<Eigen::IndexPair<int>, 1> product_dims = {Eigen::IndexPair<int>(1, 0)};
  Eigen::Array<double, 3, 1> step(grid_.geom_size[0] / grid_.cells[0], grid_.geom_size[1] / grid_.cells[1], grid_.geom_size[2] / grid_.cells[2]);

  *spectral.homogenization_F0 = F_last_inc.reshape(Eigen::DSizes<Eigen::DenseIndex, 3>(3, 3, grid_.cells[0]*grid_.cells[1]*grid_.cells2));

  x_n.setZero();
  for (int j = 0; j <= grid_.cells[1]; j++) {
    for (int k = 0; k <= grid_.cells2; k++) {
      for (int i = 0; i <= grid_.cells[0]; i++) {
        std::array<double, 3> pos_data = {i * step[0], 
                                          j * step[1], 
                                          (k + grid_.cells_local_offset_z) * step[2]};
        TensorMap<Tensor<double, 1>> pos_tensor(pos_data.data(), 3);
        Tensor<double, 1> result = Favg.contract(pos_tensor, product_dims);
        for (int p = 0; p < 3; ++p) x_n(p,i,j,k) = result(p);
        for (int n = 0; n < 8; ++n) {
          int me[3] = {i+neighbor[n][0],j+neighbor[n][1],k+neighbor[n][2]};
          Tensor<double, 1> ut_padded_slice = u_tilde_p_padded.chip(me[2], 3)
                                                              .chip(grid_.modulo((me[1]-1), grid_.cells[1]), 2)
                                                              .chip(grid_.modulo((me[0]-1), grid_.cells[0]), 1);
          for (int p = 0; p < 3; ++p) x_n(p,i,j,k) += ut_padded_slice(p)* 0.125;
        }
      }
    }
  }

  // Calculate cell center/point positions
  for (int k = 0; k < grid_.cells2; ++k) {
    for (int j = 0; j < grid_.cells[1]; ++j) {
      for (int i = 0; i < grid_.cells[0]; ++i) {
        std::array<double, 3> pos_data = {(i + 0.5) * step[0], 
                                          (j + 0.5) * step[1], 
                                          (k + 0.5 + grid_.cells_local_offset_z)*step[2]};
        TensorMap<Tensor<double, 1>> pos_tensor(pos_data.data(), 3);
        Tensor<double, 1> result = Favg.contract(pos_tensor, product_dims);
        for (int p = 0; p < 3; ++p) x_p(p, i, j, k) += result(p);
      }
    }
  }
}

void MechBase::update_gamma(Tensor<double, 4> &C) {
  C_ref = C / spectral.wgt;
  if (!config.numerics.fft.memory_efficient){
    gamma_hat.setConstant(complex<double>(0.0, 0.0));
    for (int j = grid_.cells1_offset_tensor; j < grid_.cells1_offset_tensor + grid_.cells1_tensor; ++j) {
      for (int k = 0; k < grid_.cells[3]; ++k) {
        for (int i = 0; i < grid_.cells0_reduced; ++i) {
          if (i != 0 || j != 0 || k != 0) {
            Eigen::Matrix<complex<double>, 3, 3> xiDyad_cmplx;
            Eigen::Matrix<complex<double>, 3, 3> temp33_cmplx;
            for (int l = 0; l < 3; ++l) {
                for (int m = 0; m < 3; ++m) {
                  xiDyad_cmplx(l, m) =  std::conj(-spectral.xi1st(l, i, k, j - grid_.cells1_offset_tensor)) * 
                                        spectral.xi1st(m, i, k, j - grid_.cells1_offset_tensor);
                }
            }
            for (int l = 0; l < 3; ++l) {
              for (int m = 0; m < 3; ++m) {
                temp33_cmplx(l, m) = 0;
                for (int n = 0; n < 3; ++n) {
                  for (int o = 0; o < 3; ++o) {
                    temp33_cmplx(l, m) += complex<double>(C_ref(l, n, m, o), 0) * xiDyad_cmplx(n, o);
                  }
                }
              }   
            }
            Eigen::Matrix<double, 6, 6> A;
            A.block<3, 3>(0, 0) = temp33_cmplx.real(); 
            A.block<3, 3>(3, 3) = temp33_cmplx.real();
            A.block<3, 3>(0, 3) = temp33_cmplx.imag(); 
            A.block<3, 3>(3, 0) = -temp33_cmplx.imag();

            // TODO
            // if (std::abs(A.block<3, 3>(0, 0).determinant()) > 1e-16) {
            //   Eigen::Matrix<double, 6, 6> A_inv;
            //   A_inv = A.inverse();
            //   for (int i = 0; i < 3; ++i) {
            //     for (int j = 0; j < 3; ++j) {
            //       temp33_cmplx(i, j) = std::complex<double>(A_inv(i, j), A_inv(i + 3, j));
            //     }
            //   }
            //   for (int m = 0; m < 3; ++m) {
            //     for (int n = 0; n < 3; ++n) {
            //         for (int o = 0; o < 3; ++o) {
            //     for (int l = 0; l < 3; ++l) 
            //       gamma_hat(l, m, n, o, i, k, j - grid_.cells1_offset_tensor) = temp33_cmplx(l, n) * xiDyad_cmplx(o, m);
            //       }
            //     }
            //   }
            // }
          }
        }
      }
    }
  }
}

Tensor<double, 5> MechBase::forward_field(double delta_t, 
                                          Tensor<double, 5>& field_last_inc, 
                                          Tensor<double, 5>& rate,
                                          const std::optional<Tensor<double, 2>>& aim) {
  Tensor<double, 5> forwarded_field = field_last_inc + rate*delta_t;
  print_f("field_last_inc",field_last_inc);
  print_f("rate",rate);
  print_f("aim",*aim);
  // print_f("utilities_forwardTensorField",forwarded_field);
  if (aim.has_value()) {
    Eigen::array<int, 3> reduce_dims = {2, 3, 4};
    Tensor<double, 2> field_diff = forwarded_field.sum(reduce_dims) * spectral.wgt;
    int count = field_diff.size();
    MPI_Allreduce(MPI_IN_PLACE, field_diff.data(), count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    print_f("field_diff1",field_diff);

    field_diff = field_diff - aim.value();  
    // print_f("fieldDiff2", field_diff);
    print_f("field_diff2",field_diff);
    print_f("forwarded_field",forwarded_field);

    for (int i = 0; i < grid_.cells[0]; ++i) {
      for (int j = 0; j < grid_.cells[1]; ++j) {
        for (int k = 0; k < grid_.cells2; ++k) {
          for (int m = 0; m < 3; ++m) {
            for (int n = 0; n < 3; ++n) {
              forwarded_field(m, n, i, j, k) -= field_diff(m, n);
            }
          }
        }
      }
    }
  }
  return forwarded_field;
}

Tensor<double, 4> MechBase::calculate_masked_compliance( Tensor<double, 4> &C,
                                            Eigen::Quaterniond &rot_bc_q,
                                            Eigen::Matrix<bool, 3, 3> &mask_stress) {
  Tensor<double, 4> masked_compliance(3, 3, 3, 3);
  std::array<bool, 9> mask_stress_1d;
  for (int col = 0; col < mask_stress.cols(); ++col) {
    for (int row = 0; row < mask_stress.rows(); ++row) {
      mask_stress_1d[col * mask_stress.rows() + row] = !mask_stress(row, col);
    }
  }
  int size_reduced = std::count(mask_stress_1d.begin(), mask_stress_1d.end(), true);

  Eigen::Matrix<double, 9, 9> temp99_real;
  if (size_reduced > 0) {
    Tensor<double, 4> rotated = FortranUtilities::rotate_tensor4(rot_bc_q, C);

    f_math_3333to99(rotated.data(), temp99_real.data());
    Eigen::Matrix<bool, 9, 9> mask;
    for (int i = 0; i < 9; ++i) {
      for (int j = 0; j < 9; ++j) {
        mask(i, j) = mask_stress_1d[i] && mask_stress_1d[j];
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
    FortranUtilities::invert_matrix(s_reduced, c_reduced);
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
  return masked_compliance;
}

double MechBase::calculate_divergence_rms(const Tensor<double, 5>& tensor_field) {
  // print_f("tensor_field",tensor_field);
  Eigen::Tensor<std::complex<double>, 5> tensorfield_fourier = spectral.tensorfield->forward(tensor_field);
  // cout << "tensorfield_fourier" << tensorfield_fourier << endl;

  Eigen::Vector3cd rescaled_geom;
  // cout << "grid_.geom_size" << endl;
  for (double i: grid_.geom_size)
    std::cout << i << ' ';
  cout << endl;
  // cout << "grid_.scaled_geom_size" << endl;
  for (double i: grid_.scaled_geom_size)
    std::cout << i << ' ';
  cout << endl;
  for (int i = 0; i < 3; ++i) rescaled_geom[i] = complex<double>(grid_.geom_size[i] / grid_.scaled_geom_size[i], 0);
  // cout << "rescaledGeom" << rescaled_geom << endl;

  double rms = 0;
  Tensor<complex<double>, 2> tensorField_fourier_slice;
  Tensor<complex<double>, 1> xi1st_slice;
  Eigen::Vector3cd product;
  for (int j = 0; j < grid_.cells1_tensor; ++j) {
    for (int k = 0; k < grid_.cells[2]; ++k) {
      for (int i = 1; i < grid_.cells0_reduced - 1; ++i) {
        tensorField_fourier_slice = tensorfield_fourier.chip(j, 4).chip(k,3).chip(i,2)
          .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
        Eigen::Map<const Eigen::Matrix<complex<double>, 3, 3>> tensorField_fourier_mat(tensorField_fourier_slice.data());
        xi1st_slice = -spectral.xi1st.chip(j, 3).chip(k, 2).chip(i, 1);
        Eigen::Map<Eigen::Vector3cd> xi1st_vec(xi1st_slice.data());

        cout << j << " " << k << " " << i << " " << tensorField_fourier_slice << endl;
        // cout << "-xi1st_vec.conjugate().array() " << (-xi1st_vec).conjugate().array() << endl;
        // cout << "rescaled_geom " << rescaled_geom << endl;
        product = tensorField_fourier_mat * (((-xi1st_vec).conjugate().array() * rescaled_geom.array()).matrix());
        rms += 2*(product.real().array().square().sum() + product.imag().array().square().sum());
      }
      // cout << j << " " << k << " " << rms << endl;
      
      tensorField_fourier_slice = tensorfield_fourier.chip(j, 4).chip(k,3).chip(0,2)
          .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
      Eigen::Map<const Eigen::Matrix<complex<double>, 3, 3>> tensorField_fourier_mat(tensorField_fourier_slice.data());
      xi1st_slice = -spectral.xi1st.chip(j, 3).chip(k, 2).chip(0, 1);
      Eigen::Map<Eigen::Vector3cd> xi1st_vec(xi1st_slice.data());
      product = tensorField_fourier_mat * ((-xi1st_vec.conjugate().array() * rescaled_geom.array()).matrix());

      // cout << "-xi1st(1:3,1,k,j) " << xi1st_vec << endl;
      rms += product.real().array().square().sum() +
             product.imag().array().square().sum();
      // auto c = product.real().array().square().sum();
      // cout << "c " << c << endl;
      // auto d = product.imag().array().square().sum();
      // cout << "d " << d << endl;

      tensorField_fourier_slice = tensorfield_fourier.chip(j, 4).chip(k,3).chip(grid_.cells0_reduced-1,2)
          .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
      Eigen::Map<const Eigen::Matrix<complex<double>, 3, 3>> tensorField_fourier_matt(tensorField_fourier_slice.data());
      xi1st_slice = -spectral.xi1st.chip(j, 3).chip(k, 2).chip(grid_.cells0_reduced-1, 1);
      Eigen::Map<Eigen::Vector3cd> xi1st_vecc(xi1st_slice.data());
      product = tensorField_fourier_matt * ((-xi1st_vecc.conjugate().array() * rescaled_geom.array()).matrix());

      rms += product.real().array().square().sum() +
             product.imag().array().square().sum();

      // cout << "-xi1st(1:3,cells1Red,k,j) " << xi1st_vecc << endl;

      // auto e = product.real().array().square().sum();
      // cout << "e " << e << endl;
      // auto f = product.imag().array().square().sum();
      // cout << "f " << f << endl;

      // std::vector<int> indices = {0, grid_.cells0_reduced-1};
      // for (int i : indices) {


      //   tensorField_fourier_slice =  tensorfield_fourier.chip(j, 4).chip(k,3).chip(i,2)
      //     .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
      //   // Eigen::Map<const Eigen::Matrix<complex<double>, 3, 3>> tensorField_fourier_mat(tensorField_fourier_slice.data());
      //   // xi1st_slice = -spectral.xi1st.chip(j, 3).chip(k, 2).chip(i, 1);
      //   // Eigen::Map<Eigen::Vector3cd> xi1st_vec(xi1st_slice.data());
      //   // product = tensorField_fourier_mat * ((xi1st_vec.conjugate().array() * rescaled_geom.array()).matrix());
      //   // rms += (product.real().array().square().sum() + product.imag().array().square().sum());
      // }
      // cout << j << " " << k << " " << rms << endl;
      // cout << " ==========================================================================================" << endl;
    }
  }
  // cout << "xi1st2" << spectral.xi1st << endl;

  MPI_Allreduce(MPI_IN_PLACE, &rms, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  cout << "rms1" << rms << endl;
  rms = sqrt(rms) * spectral.wgt;
  if (grid_.cells[0] == 1) rms = rms/2;
  cout << "rms2" << rms << endl;
  // exit(123);

  return rms;
}

void MechBase::gamma_convolution(TensorMap<Tensor<double, 5>> &field, Tensor<double, 2> &field_aim) {
  cout << "\n ... doing gamma convolution ..............................................." << endl;
  Eigen::Tensor<std::complex<double>, 5> tensorfield_fourier = spectral.tensorfield->forward(field);
  Eigen::Matrix<complex<double>, 3, 3> temp33_cmplx;
  temp33_cmplx.setZero();
  if (config.numerics.fft.memory_efficient) {
    Eigen::Matrix<double, 6, 6> A;
    Eigen::Matrix<double, 6, 6> A_inv;
    Eigen::Matrix<complex<double>, 3, 3> xiDyad_cmplx;
    // #pragma omp parallel for private(l, m, n, o, temp33_cmplx, xiDyad_cmplx, A, A_inv, err, gamma_hat)
    for (int j = 0; j < grid_.cells1_tensor; ++j) {
      for (int k = 0; k < grid_.cells[2]; ++k) {
        for (int i = 0; i < grid_.cells0_reduced; ++i) {
          if (i != 0  || j + grid_.cells1_offset_tensor != 0 || k != 0) { // singular point at xi=(0.0,0.0,0.0) i.e. i=j=k=1
            for (int l = 0; l < 3; ++l) {
              for (int m = 0; m < 3; ++m) {
                xiDyad_cmplx(l, m) = std::conj(-spectral.xi1st(l, i, k, j)) * spectral.xi1st(m, i, k, j);
              }
            }
            for (int l = 0; l < 3; ++l) {
              for (int m = 0; m < 3; ++m) {
                for (int n = 0; n < 3; ++n) {
                  for (int o = 0; o < 3; ++o) {
                    temp33_cmplx(l, m) += complex<double>(C_ref(l, n, m, o)) * xiDyad_cmplx(n, o);
                  }
                }
              }
            }
            A.block(0, 0, 3, 3) = temp33_cmplx.real();
            A.block(3, 3, 3, 3) = temp33_cmplx.real();
            A.block(0, 3, 3, 3) = temp33_cmplx.imag();
            A.block(3, 0, 3, 3) = -temp33_cmplx.imag();
            // TODO: if det(A(1:3,1:3)> ...)
            tensorfield_fourier.slice(Eigen::array<Eigen::Index, 5>({0, 0, i, k, j}),
                                      Eigen::array<Eigen::Index, 5>({3, 3, 1, 1, 1})).setConstant(complex<double>(0,0));
          }
        }
      }
    }
  } else {
    Eigen::Matrix<complex<double>, Eigen::Dynamic, Eigen::Dynamic> res;
    for (int j = 0; j < grid_.cells1_tensor; ++j) {
      for (int k = 0; k < grid_.cells[2]; ++k) {
        for (int i = 0; i < grid_.cells0_reduced; ++i) {
          for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {
              Tensor<complex<double>, 2> gamma_hat_slice =  gamma_hat.chip(j, 6).chip(k,5).chip(i,4).chip(m,1).chip(l,0)
                              .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
              Tensor<complex<double>, 2> gamma_hat_slice_ = gamma_hat_slice; //create copy to generate continuous storage of data
              Eigen::Map<const Eigen::Matrix<complex<double>, 3, 3>> gamma_hat_slice_mat(gamma_hat_slice_.data());
              Tensor<complex<double>, 2> tensorField_fourier_slice = tensorfield_fourier.chip(j, 4).chip(k,3).chip(i,2)
                              .slice(Eigen::array<Eigen::Index, 2>({0,0}), Eigen::array<Eigen::Index, 2>({3,3}));
              Tensor<complex<double>, 2> tensorField_fourier_slice_ = tensorField_fourier_slice; //create copy to generate continuous storage of data
              Eigen::Map<const Eigen::Matrix<complex<double>, 3, 3>> tensorField_fourier_slice_mat(tensorField_fourier_slice_.data());
              res = gamma_hat_slice_mat * tensorField_fourier_slice_mat;
              temp33_cmplx(l, m) = tensor_sum(res);
            }
          }
          for (int l = 0; l < 3; ++l) {
            for (int m = 0; m < 3; ++m) {
              (*spectral.tensorfield->field_fourier)(l, m, i, k, j) = temp33_cmplx(l, m);
            }
          }
        }
      }
    }
  }
  if (grid_.cells_local_offset_z==0) {
    for (int l = 0; l < 3; ++l) {
      for (int m = 0; m < 3; ++m) {
        tensorfield_fourier(l, m, 0, 0, 0) = complex<double>(field_aim(l, m), 0);
      }
    }
  }
  double neutral_wgt = 1;
  // PERFORMANCE TODO: look into directly modifying resulting tensorfield_fourier if possible
  field = spectral.tensorfield->backward(&tensorfield_fourier, neutral_wgt);
}

Eigen::Tensor<double, 5> MechBase::calculate_rate(bool heterogeneous, 
                                                  const Eigen::Tensor<double, 5>& field0, 
                                                  const Eigen::Tensor<double, 5>& field, 
                                                  double dt, 
                                                  const Eigen::Tensor<double, 2>& avRate) {
  Eigen::Tensor<double, 5> rate(field.dimensions());
  if (heterogeneous) {
    rate = (field - field0) / dt;
  } else {
    for (int i = 0; i < field.dimension(2); ++i) {
      for (int j = 0; j < field.dimension(3); ++j) {
        for (int k = 0; k < field.dimension(4); ++k) {
          rate.chip<4>(k).chip<3>(j).chip<2>(i) = avRate;
        }
      }
    }
  }
  return rate;
}
