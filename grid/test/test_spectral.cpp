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

TEST_F(SpectralSetup, TestUpdateCoordsInit) {
  MockDiscretization mock_discretization;
  int cells_[] = {2,1,1};
  double geom_size_[] = {2e-5,1e-5,1e-5};
  MockDiscretizedGrid mock_grid(mock_discretization, &cells_[0], &geom_size_[0]);
  PartialMockSpectral spectral(mock_grid);

  std::array<std::complex<double>, 3> res;
  for (int i = 0; i < 3; ++i) res[i] = std::complex<double>(1,0);
  EXPECT_CALL(spectral, get_freq_derivative(testing::_))
    .WillRepeatedly(testing::DoAll(testing::Return(res)));

  double expected_ip_coords_[3][12] = {
    {
        0, 1e-05, 2e-05,     0, 1e-05, 2e-05,     0, 1e-05, 2e-05,     0, 1e-05, 2e-05
    }, {
        0,     0,     0, 1e-05, 1e-05, 1e-05,     0,     0,     0, 1e-05, 1e-05, 1e-05
    }, {
        0,     0,     0,     0,     0,     0, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05, 1e-05
    }
  };
  Eigen::array<Eigen::Index, 2> dims_ip_coords = {3, 12};
  Eigen::Tensor<double, 2> expected_ip_coords = array_to_eigen_tensor<double, 2>(&expected_ip_coords_[0][0], dims_ip_coords);
  EXPECT_CALL(mock_discretization, set_node_coords(testing::_))
  .WillOnce(testing::Invoke([&](Eigen::Tensor<double, 2>* ip_coords){
      bool eq = tensor_eq<double, 2>(*ip_coords, expected_ip_coords);
      EXPECT_TRUE(eq);
  })); 

  double expected_node_coords_[3][2] = {
    {
      5e-06, 1.5e-05
    }, {
      5e-06, 1.000005
    }, {
    1.000005,   5e-06
  }
  };
  Eigen::array<Eigen::Index, 2> dims_node_coords = {3, 2};
  Eigen::Tensor<double, 2> expected_node_coords = array_to_eigen_tensor<double, 2>(&expected_node_coords_[0][0], dims_node_coords);

  EXPECT_CALL(mock_discretization, set_ip_coords(testing::_))
  .WillOnce(testing::Invoke([&](Eigen::Tensor<double, 2>* node_coords){
      bool eq = tensor_eq<double, 2>(*node_coords, expected_node_coords);
      EXPECT_TRUE(eq);
  })); 

  Eigen::Tensor<double, 5> F(3, 3, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2]);
  Eigen::Matrix3d math_I3 = Eigen::Matrix3d::Identity();
  for (int i = 0; i < 3; ++i)
  {
      for (int j = 0; j < 3; ++j)
      {
          F.slice(Eigen::array<Eigen::Index, 5>({i, j, 0, 0, 0}),
                  Eigen::array<Eigen::Index, 5>({1, 1, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2]}))
                  .setConstant(math_I3(i, j));
      }
    }
  spectral.init(); // TODO: recreate only the required parts in test constructor
  spectral.update_coords(F);
}

// TODO: add mpi test with initialized instead of mocked discretization
