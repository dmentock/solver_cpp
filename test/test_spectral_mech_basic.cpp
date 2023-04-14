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
  MockDiscretizedGrid grid;
  MockSpectralBasic spectral_basic(grid);
  spectral_basic.init();

}

//   MockDiscretizationGrid discretization_grid;
//   // assert calculate_nodes0 and calculate_ipCoordinates0 are called with expected values
//   EXPECT_CALL(discretization_grid, calculate_nodes0(
//     ArrayPointee(3, testing::ElementsAreArray(cells)), 
//     ArrayPointee(3, testing::ElementsAreArray(geom_size)), 
//     testing::Eq(0))).WillOnce(testing::DoDefault());
//   EXPECT_CALL(discretization_grid, calculate_ipCoordinates0(
//     ArrayPointee(3, testing::ElementsAreArray(cells)), 
//     ArrayPointee(3, testing::ElementsAreArray(geom_size)), 
//     testing::Eq(0))).WillOnce(testing::DoDefault());
  
//   // assert discretization_grid fortran function is called with expected values
//   std::vector<int> expected_grid(grid, grid + grid_size);
//   EXPECT_CALL(discretization_grid, f_discretization_init(
//     ArrayPointee(grid_size, testing::ElementsAreArray(expected_grid)), 
//     testing::Pointee(24),
//     testing::_, 
//     testing::_, 
//     testing::Pointee(48)))
//     .WillOnce(testing::Return());
//   discretization_grid.init(false, &mock_vti_reader);


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}