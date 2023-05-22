#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <cstdio>
#include <fstream>

#include <mpi.h>
#include <petscsys.h>
#include <petsc.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "spectral/spectral.h"
#include "simple_grid_setup.hpp"
#include "init_environments.hpp"

#include <test/complex_concatenator.h>
#include <tensor_operations.h>
#include <helper.h>


TEST_F(SimpleGridSetup,SpectralTestInit) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);

  Eigen::Tensor<std::complex<double>, 4> expected_xi1st(3, 2, 1, 1);
  expected_xi1st.setValues({
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}},
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}},
   {{{ c( 0 ,  0 ) }},
    {{ c( 0 ,  0 ) }}}
  });

  Eigen::Tensor<std::complex<double>, 4> expected_xi2nd;
  expected_xi2nd.resize(3, 2, 1, 1);
  expected_xi2nd.setValues({
   {{{ c( 0               ,  0                ) }},
    {{ c( 0               ,  314159.2653589793) }}},
   {{{ c( 0               ,  0                ) }},
    {{ c( 0               ,  0                ) }}},
   {{{ c( 0               ,  0                ) }},
    {{ c( 0               ,  0                ) }}}
  });

  spectral.init();
  // print_f("xi1st", spectral.xi1st);
  EXPECT_TRUE(tensor_eq(spectral.xi1st, expected_xi1st));
  EXPECT_TRUE(tensor_eq(spectral.xi2nd, expected_xi2nd));
  // TODO: mock calls to set_up_fftw template function
}


// TEST(SpectralTestFFTRandom, BasicTest) {
//   MockDiscretization mock_discretization;
//   int cells_[] = {2, 1, 1};
//   double geom_size_[] = {2e-5, 1e-5, 1e-5};
//   MockDiscretizedGrid mock_grid(mock_discretization, &cells_[0], &geom_size_[0]);
//   Spectral spectral(mock_grid);

//   ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
//   spectral.set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, 9, 
//                        spectral.tensorField_real, spectral.tensorField_fourier, spectral.tensorField_fourier_fftw,
//                        FFTW_MEASURE, spectral.plan_tensor_forth, spectral.plan_tensor_back,
//                        "tensor");
//   mock_grid->cells1_tensor = cells1_fftw;
//   mock_grid->cells1_offset_tensor = cells1_offset;

//   fill_random(*spectral.tensorField_real);

//   spectral.tensorField_real->slice(Eigen::array<long, 5>({0, 0, mock_grid->cells[0], 0, 0}), 
//                                    Eigen::array<long, 5>({3, 3, mock_grid->cells0_reduced * 2 - mock_grid->cells[0], -1, -1})).setConstant(0);

//   auto tensorField_real_copy = *(spectral.tensorField_real);

//   fftw_mpi_execute_dft_r2c(spectral.plan_tensor_forth, spectral.tensorField_real->data(), spectral.tensorField_fourier_fftw);

//   std::complex<double> sum =   (*spectral.tensorField_fourier);
  // double tensorSumLocal = tensorField_real_copy.sum();
  // MPI_Allreduce(&tensorSumLocal, &tensorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  // double avg = tensorSum / tensorField_fourier->dimension(2) / tensorField_fourier->dimension(3) / tensorField_fourier->dimension(4);
  // double epsilon = 1e-12;

  // if (world_rank == 0) {
  //   if (std::abs(avg - 1.0) > epsilon) {
  //     std::cerr << "Mismatch avg tensorField FFT <-> real" << std::endl;
  //     MPI_Abort(MPI_COMM_WORLD, 1);
  //   }
  // }

  // fftw_execute(plan_tensor_back);

  // // Set the specified range to zero
  // tensorField_real->slice(Eigen::array<long, 5>({0, 0, cells(1), 0, 0}), Eigen::array<long, 5>({3, 3, cells1Red * 2 - cells(1), -1, -1})) = Eigen::Tensor<double, 5>::Zero(3, 3, cells1Red * 2 - cells(1), grid.cells[1], cells2_fftw);

  // double max_difference = (*tensorField_real - tensorField_real_copy * wgt).abs().
// }


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}