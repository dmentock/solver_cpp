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
#include "spectral/spectral.h"

class SpectralSetup : public ::testing::Test {
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

// TEST_F(SpectralSetup, TestInit) {
//   MockDiscretizedGrid grid;
//   Spectral spectral(grid);
//   spectral.init();
// }
// class MockSpectralBasic : public Basic {
// public:
//     MockSpectralBasic(DiscretizationGrid& grid_) : Basic(grid_) {}
//     MOCK_METHOD(void, update_coords, ((Eigen::Tensor<double, 5>)), (override));
// };

TEST_F(SpectralSetup, TestConstitutiveResponse) {
  MockDiscretizedGrid grid;
  Spectral spectral(grid);

  int cells[3] = {2,3,4};
  Eigen::Tensor<double, 5> P(3, 3, cells[0], cells[1], cells[2]);
  Eigen::Tensor<double, 2> P_av(3, 3);
  Eigen::Tensor<double, 4> C_volAvg(3, 3, 3, 3);
  Eigen::Tensor<double, 4> C_minMaxAvg(3, 3, 3, 3);
  Eigen::Tensor<double, 5> F(3, 3, cells[0], cells[1], cells[2]);
  Eigen::Matrix3d math_I3 = Eigen::Matrix3d::Identity();
  for (int i = 0; i < 3; ++i)
  {
      for (int j = 0; j < 3; ++j)
      {
          F.slice(Eigen::array<Eigen::Index, 5>({i, j, 0, 0, 0}),
                  Eigen::array<Eigen::Index, 5>({1, 1, cells[0], cells[1], cells[2]}))
              .setConstant(math_I3(i, j));
      }
  }
  spectral.constitutive_response(P,
                                  P_av,
                                  C_volAvg,
                                  C_minMaxAvg,
                                  F,
                                  0);


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
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}