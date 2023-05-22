#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <cstdio>
#include <fstream>

// #include <mpi.h>
#include <petscsys.h>
#include <petsc.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "spectral/solver/basic.h"
#include "simple_grid_setup.hpp"


class PetscMpiEnv : public ::testing::Environment {
protected:
  PetscErrorCode ierr;
public:
  virtual void SetUp() {
    int argc = 0;
    char **argv = NULL;
    ierr = PetscInitialize(&argc, &argv, (char *)0,"PETSc help message.");
    ASSERT_EQ(ierr, 0) << "Error initializing PETSc.";
  }
  virtual void TearDown() override {
    ierr = PetscFinalize();
    ASSERT_EQ(ierr, 0) << "Error finalizing PETSc.";
  }
};

class PartialMockSolverBasic : public SolverBasic {
public:
  PartialMockSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : SolverBasic(config_, grid_, spectral_) {};

  using Tensor2 = Eigen::Tensor<double, 2>;
  using Tensor4 = Eigen::Tensor<double, 4>;
  using Tensor5 = Eigen::Tensor<double, 5>;
  MOCK_METHOD(void, update_coords, (Tensor5&, Tensor2&, Tensor2&), (override));
  MOCK_METHOD(void, update_gamma, (Tensor4&), (override));
};

TEST_F(SimpleGridSetup, TestSolverBasicInit) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);
  PartialMockSolverBasic solver_basic(config, *mock_grid, spectral);
  solver_basic.init();
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}