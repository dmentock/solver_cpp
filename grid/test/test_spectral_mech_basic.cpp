#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array_matcher.h>
#include <iostream>
#include <cstdio>
#include <fstream>

// #include <mpi.h>
#include <petscsys.h>
#include <petsc.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "mock_discretized_grid.hpp"
#include "spectral/mech/basic.h"

class SpectralBasicSetup : public ::testing::Test {
  protected:
    PetscErrorCode ierr;

  void SetUp() override {
    MockDiscretizedGrid grid;
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

class MockSpectralBasic : public Basic {
public:
    MockSpectralBasic(DiscretizationGrid& grid_) : Basic(grid_) {}
    using Tensor2 = Eigen::Tensor<double, 2>;
    using Tensor4 = Eigen::Tensor<double, 4>;
    using Tensor5 = Eigen::Tensor<double, 5>;
    MOCK_METHOD(void, update_coords, (Tensor5&), (override));
    MOCK_METHOD(void, update_gamma, (Tensor4&), (override));
    MOCK_METHOD(void, constitutive_response, (Tensor5&, Tensor2&, Tensor4&, Tensor4&, Tensor5&, double), (override));
};

TEST_F(SpectralBasicSetup, TestInit) {
  MockDiscretization mock_discretization;
  int cells_[] = {2, 3, 4};
  double geom_size_[] = {2e-5, 3e-5, 4e-5};
  MockDiscretizedGrid mock_grid(mock_discretization, &cells_[0], &geom_size_[0]);
  MockSpectralBasic spectral_basic(mock_grid);
  spectral_basic.init();

}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}