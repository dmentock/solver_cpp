#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <test/array_matcher.h>
#include <iostream>
#include <cstdio>
#include <fstream>

#include <mpi.h>
#include <fftw3-mpi.h>

#include "discretization_grid.h"
#include "vti_reader.h"
class MockDiscretization : public Discretization {
  public:
    MOCK_METHOD(void, init, (int*, int*, double*, double*, int), (override));
};
class MockDiscretizationGrid : public DiscretizationGrid {
  public:
    MockDiscretizationGrid(Discretization& discretization)
      : DiscretizationGrid(discretization) {
    }
    MOCK_METHOD(double*, calculate_nodes0, (int[3], double[3], int), (override));
    MOCK_METHOD(double*, calculate_ipCoordinates0, (int[3], double[3], int), (override));
};
class DiscretizationGridSetup : public ::testing::Test {
  protected:
    //mock vti file output
    int cells[3] = {2, 3, 4};
    double geom_size[3] = {1.0, 1.0, 1.0};
    double origin[3] = {0.0, 0.0, 0.0};
    int grid_size = 2*3*4;
    int* grid = new int[grid_size];
  void SetUp() override {
    MPI_Init(NULL, NULL);
    // fill grid returned by read_vti module with unique ids from 1 to 24 
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
          for (int k = 0; k < 4; ++k) {
              grid[i*3*4+j*4+k] = 1+i*3*4+j*4+k;
          }
      }
    }
  }
  void TearDown() override {
    fftw_mpi_cleanup();
    MPI_Finalize();
  }
};
TEST_F(DiscretizationGridSetup, TestInit) {

  MockDiscretization mock_discretization;
  MockDiscretizationGrid discretization_grid(mock_discretization);

  // assert calculate_nodes0 and calculate_ipCoordinates0 are called with expected values
  EXPECT_CALL(discretization_grid, calculate_nodes0(
    ArrayPointee(3, testing::ElementsAreArray(cells)), 
    ArrayPointee(3, testing::ElementsAreArray(geom_size)), 
    testing::Eq(0))).WillOnce(testing::DoDefault());
  EXPECT_CALL(discretization_grid, calculate_ipCoordinates0(
    ArrayPointee(3, testing::ElementsAreArray(cells)), 
    ArrayPointee(3, testing::ElementsAreArray(geom_size)), 
    testing::Eq(0))).WillOnce(testing::DoDefault());

  // assert discretization_grid fortran function is called with expected values
  std::vector<int> expected_grid(grid, grid + grid_size);
  EXPECT_CALL(mock_discretization, init(
    ArrayPointee(grid_size, testing::ElementsAreArray(expected_grid)), 
    testing::Pointee(24),
    testing::_, 
    testing::_, 
    testing::Eq(48)))
    .WillOnce(testing::Return());

  // discretization_grid.init(false, &mock_vti_reader);
}
// test expected behaviour of calculate_ipCoordinates0 and calculate_nodes0
class CoordCalculationSetup : public ::testing::Test {
  protected:
    int cells[3] = {2,2,2};
    double geom_size[3] = {1,2,4};
};
TEST_F(CoordCalculationSetup, TestIpCoords) {
  MockDiscretization mock_discretization;
  DiscretizationGrid discretization_grid(mock_discretization);
  // memory layout corresponds to required column-major format for fortran code
  std::vector<double> expected_ipCoords = {
      -0.25, -0.5, -1, 
      0.25, -0.5, -1, 
      -0.25, 0.5, -1, 
      0.25, 0.5, -1, 
      -0.25, -0.5, 1, 
      0.25, -0.5, 1, 
      -0.25, 0.5, 1, 
      0.25, 0.5, 1,
  };
  ASSERT_THAT(discretization_grid.calculate_ipCoordinates0(cells, geom_size, 0), ArrayPointee(sizeof(expected_ipCoords), 
              testing::ElementsAreArray(expected_ipCoords)));
}
TEST_F(CoordCalculationSetup, TestNodes0) {
  MockDiscretization mock_discretization;
  DiscretizationGrid discretization_grid(mock_discretization);
  std::vector<double> expected_nodeCoords = {
      0, 0, 0, 
      0.5, 0, 0, 
      0, 1, 0, 
      0.5, 1, 0, 
      0, 0, 2, 
      0.5, 0, 2, 
      0, 1, 2, 
      0.5, 1, 2
  };
  ASSERT_THAT(discretization_grid.calculate_nodes0(cells, geom_size, 0), ArrayPointee(sizeof(expected_nodeCoords), 
              testing::ElementsAreArray(expected_nodeCoords)));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}