#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <fft.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <fftw3-mpi.h>

#include <tensor_operations.h>
#include "init_environments.hpp"


TEST(TestFFT, TestInitForwardBackwardCopy) {
  std::array<int, 3> cells = {2, 1, 1};
  int cells2 = 1;
  std::vector<int> extra_dims = {3, 3};
  int fft_flag = 0;
  ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;

  fftw_mpi_init();
  FFT<5> fft_obj(cells, cells2, extra_dims, fft_flag, &cells1_fftw, &cells1_offset, &cells2_fftw);
  Eigen::Tensor<double, 5> field_real_test(3,3,2,1,1);
  field_real_test.setRandom();
  Eigen::Tensor<std::complex<double>, 5> field_fourier_test = fft_obj.forward(field_real_test);
  for (int i = 0; i < field_fourier_test.size(); ++i) {
    EXPECT_TRUE(!(std::abs(field_fourier_test.data()[i].real() - 1.0) < 1e-12));
  }
  double wgt = 0.5;
  Eigen::Tensor<double, 5> field_real_test_ = fft_obj.backward(&field_fourier_test, wgt);
  EXPECT_TRUE(tensor_eq(field_real_test, field_real_test_));
}

TEST(TestFFT, TestInitForwardBackwardDirectAssignment) {
  std::array<int, 3> cells = {2, 1, 1};
  int cells2 = 1;
  std::vector<int> extra_dims = {3, 3};
  int fft_flag = 0;
  ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;

  fftw_mpi_init();
  FFT<5> fft_obj(cells, cells2, extra_dims, fft_flag, &cells1_fftw, &cells1_offset, &cells2_fftw);
  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<double> dist(1.0, 1000.0);
  for (int i = 0; i < fft_obj.field_real->size(); ++i) {
    fft_obj.field_real->data()[i] = dist(mt);
  }
  Eigen::TensorMap<Eigen::Tensor<double, 5>> field_real_map = *fft_obj.field_real;
  Eigen::Tensor<double, 5> field_real_copy = field_real_map.slice(
    Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}), 
    Eigen::array<Eigen::Index, 5>({3, 3, 2, 1, 1}));

  Eigen::Tensor<std::complex<double>, 5> field_fourier_test = fft_obj.forward(*fft_obj.field_real);
  for (int i = 0; i < field_fourier_test.size(); ++i) {
    EXPECT_TRUE(!(std::abs(field_fourier_test.data()[i].real() - 1.0) < 1e-12));
  }
  double wgt = 0.5;
  Eigen::Tensor<double, 5> field_real_test_ = fft_obj.backward(&field_fourier_test, wgt);
  EXPECT_TRUE(tensor_eq(field_real_test_, field_real_copy));
}

// TEST(TestFFT, TestInitForwardBackwardInPlace) {
//   fftw_mpi_init();
//   std::array<int, 3> cells = {2, 1, 1};
//   int cells2 = 1;
//   std::vector<int> extra_dims = {3, 3};
//   int fft_flag = 0;
//   ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
//   FFT<5> fft_obj(cells, cells2, extra_dims, fft_flag, &cells1_fftw, &cells1_offset, &cells2_fftw);

//   std::random_device rd;
//   std::mt19937 mt(rd());
//   std::uniform_real_distribution<double> dist(1.0, 1000.0);
//   for (int i = 0; i < fft_obj.field_real->size(); ++i) {
//     fft_obj.field_real->data()[i] = dist(mt);
//   }
//   Eigen::TensorMap<Eigen::Tensor<double, 5>> field_real_map = *fft_obj.field_real;
//   Eigen::Tensor<double, 5> field_real_copy = field_real_map.slice(
//     Eigen::array<Eigen::Index, 5>({0, 0, 0, 0, 0}),
//     Eigen::array<Eigen::Index, 5>({3, 3, 2, 1, 1}));

//   std::shared_ptr<Eigen::TensorMap<Eigen::Tensor<std::complex<double>, 5>>> field_fourier_test = 
//     fft_obj.forward(fft_obj.field_real.get(), true);
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}