#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array_matcher.h>
#include <iostream>
#include <cstdio>
#include <fstream>

#include <mpi.h>
#include <petscsys.h>
#include <petsc.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Core>

#include "mock_discretized_grid.hpp"
#include "spectral/spectral.h"
#include "helper.h"

class SpectralSetup : public ::testing::Test {
  protected:
    PetscErrorCode ierr;
  void SetUp() override {
    MPI_Init(NULL, NULL);
    int argc = 0;
    char **argv = NULL;
    ierr = PetscInitialize(&argc, &argv, (char *)0,"PETSc help message.");
    ASSERT_EQ(ierr, 0) << "Error initializing PETSc.";
  }
  void TearDown() override {
    ierr = PetscFinalize();
    ASSERT_EQ(ierr, 0) << "Error finalizing PETSc.";
  }
};

class PartialMockSpectral : public Spectral {
public:
    PartialMockSpectral(DiscretizationGrid& grid_) : Spectral(grid_) {}
    MOCK_METHOD((std::array<std::complex<double>, 3>), get_freq_derivative, (int*), (override));
};

TEST_F(SpectralSetup, TestInit) {
  MockDiscretization mock_discretization;
  int cells_[] = {2, 3, 4};
  double geom_size_[] = {2e-5, 3e-5, 4e-5};
  MockDiscretizedGrid mock_grid(mock_discretization, &cells_[0], &geom_size_[0]);
  PartialMockSpectral spectral(mock_grid);

  std::array<std::complex<double>, 3> res;
  for (int i = 0; i < 3; ++i) res[i] = std::complex<double>(1,0);
  EXPECT_CALL(spectral, get_freq_derivative(testing::_))
    .WillRepeatedly(testing::DoAll(testing::Return(res)));

  spectral.init();
  Eigen::DSizes<Eigen::DenseIndex, 4> expected_xi_dims(3, 2, 4, 3);
  ASSERT_EQ(spectral.xi1st.dimensions(), expected_xi_dims);
  ASSERT_EQ(spectral.xi2nd.dimensions(), expected_xi_dims);
  Eigen::DSizes<Eigen::DenseIndex, 6> expected_gamma_hat_dims(3, 3, 3, 2, 4, 3);
  ASSERT_EQ(spectral.gamma_hat.dimensions(), expected_gamma_hat_dims);
}

