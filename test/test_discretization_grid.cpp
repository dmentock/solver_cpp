#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <array_matcher.h>
#include <iostream>
#include <cstdio>
#include <fstream>

#include <mpi.h>
#include <fftw3-mpi.h>

#include "discretization_grid.h"
#include "vti_reader.h"

// test discretization_grid.init and mock the calls to vti_reader, calculate_nodes0, calculate_ipCoordinates0
// to test functionality in isolation, verify functionality by checking args passed to mocked discretization_init fortran subroutine
class MockVtiReader : public VtiReader {
  public:
    MOCK_METHOD(int*, read_vti_material_data, (const char*, int (&)[3], double (&)[3], double (&)[3]), (override));
};
class MockDiscretizationGrid : public DiscretizationGrid {
  public:
    MOCK_METHOD(void, f_discretization_init, (int*, int*, double*, double*, int*), (override));
    MOCK_METHOD(double*, calculate_nodes0, (int[3], double[3], int), (override));
    MOCK_METHOD(double*, calculate_ipCoordinates0, (int[3], double[3], int), (override));
};
class DiscretizationGridSetup : public ::testing::Test {
  protected:
    //mock vti file output
    int cells[3] = {2, 3, 4};
    double geomSize[3] = {1.0, 1.0, 1.0};
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
  MockVtiReader mock_vti_reader;
  // mock vti_reader module to return specified data
  EXPECT_CALL(mock_vti_reader, read_vti_material_data(testing::_, testing::_, testing::_, testing::_))
    .WillOnce(testing::DoAll(testing::SetArrayArgument<1>(cells, cells + 3),
                             testing::SetArrayArgument<2>(geomSize, geomSize + 3),
                             testing::SetArrayArgument<3>(origin, origin + 3),
                             testing::Return(grid)));

  MockDiscretizationGrid discretization_grid;
  // assert calculate_nodes0 and calculate_ipCoordinates0 are called with expected values
  EXPECT_CALL(discretization_grid, calculate_nodes0(
    ArrayPointee(3, testing::ElementsAreArray(cells)), 
    ArrayPointee(3, testing::ElementsAreArray(geomSize)), 
    testing::Eq(0))).WillOnce(testing::DoDefault());
  EXPECT_CALL(discretization_grid, calculate_ipCoordinates0(
    ArrayPointee(3, testing::ElementsAreArray(cells)), 
    ArrayPointee(3, testing::ElementsAreArray(geomSize)), 
    testing::Eq(0))).WillOnce(testing::DoDefault());
  
  // assert discretization_grid fortran function is called with expected values
  std::vector<int> expected_grid(grid, grid + grid_size);
  EXPECT_CALL(discretization_grid, f_discretization_init(
    ArrayPointee(grid_size, testing::ElementsAreArray(expected_grid)), 
    testing::Pointee(24),
    testing::_, 
    testing::_, 
    testing::Pointee(48)))
    .WillOnce(testing::Return());
  discretization_grid.init(false, &mock_vti_reader);
}

// test expected behaviour of calculate_ipCoordinates0 and calculate_nodes0
class CoordCalculationSetup : public ::testing::Test {
  protected:
    int cells[3] = {2,2,2};
    double geomSize[3] = {1,2,4};
};
TEST_F(CoordCalculationSetup, TestIpCoords) {
  DiscretizationGrid discretization_grid;
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
  ASSERT_THAT(discretization_grid.calculate_ipCoordinates0(cells, geomSize, 0), ArrayPointee(sizeof(expected_ipCoords), 
              testing::ElementsAreArray(expected_ipCoords)));
}
TEST_F(CoordCalculationSetup, TestNodes0) {
  DiscretizationGrid discretization_grid;
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
  ASSERT_THAT(discretization_grid.calculate_nodes0(cells, geomSize, 0), ArrayPointee(sizeof(expected_nodeCoords), 
              testing::ElementsAreArray(expected_nodeCoords)));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}