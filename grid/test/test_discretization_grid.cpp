#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <mpi.h>
#include <fftw3-mpi.h>

#include "discretization_grid.h"

#include "tensor_operations.h"
#include "helper.h"

class PartialMockDiscretizationGrid : public DiscretizationGrid {
  public:
    PartialMockDiscretizationGrid(std::array<int, 3> cells_)
    : DiscretizationGrid(cells_) {}
    using Tensor1i = Eigen::Tensor<int, 1>;
    using Tensor2d = Eigen::Tensor<double, 2>;
    using array3i = std::array<int, 3>;
    using array3d = std::array<double, 3>;
    MOCK_METHOD(void, calculate_nodes0, (Tensor2d&, array3i&, array3d&, int), (override));
    MOCK_METHOD(void, calculate_ipCoordinates0, (Tensor2d&, array3i&, array3d&, int), (override));
    MOCK_METHOD(void, VTI_readDataset_int, (Tensor1i& materialAt));
    MOCK_METHOD(void, discretization_init, (int* materialAt, int n_materialpoints, 
                                            double* IPcoords0, int n_ips,
                                            double* NodeCoords0, int n_nodes, 
                                            int sharedNodesBegin), (override));
    // MOCK_METHOD(void, discretization_init, (array3i&, array3d&, int), (override));
};

class DiscretizationGridSetup : public ::testing::Test {
  void SetUp() override {
    MPI_Init(NULL, NULL);
  }
  void TearDown() override {
    fftw_mpi_cleanup();
    MPI_Finalize();
  }
};

ACTION_P(SetArg0, value) { arg0 = value; }
TEST_F(DiscretizationGridSetup, TestInit) {

  PartialMockDiscretizationGrid discretization_grid(std::array<int, 3>{2, 1, 1});

  // assert calculate_nodes0 and calculate_ipCoordinates0 are called with expected values
  std::array<int, 3> cells = {2,1,1};
  discretization_grid.cells = cells;
  std::array<double, 3> geom_size = {2e-5,1e-5,1e-5};
  discretization_grid.geom_size = geom_size;
  int grid_size = cells[0] * cells[1] * cells[2];

  Eigen::Tensor<int, 1> mocked_vti_response(2);
  mocked_vti_response.setValues({3, 4});
  EXPECT_CALL(discretization_grid, VTI_readDataset_int(
    testing::_)).WillOnce(SetArg0(mocked_vti_response));

  Eigen::Tensor<double, 2> nodes0(3, 12);
  nodes0.setValues({
  {  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2 },
  {  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,  3.2 },
  {  3.1,  3.2,  3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4.0,  4.1,  4.2 }
  });
  EXPECT_CALL(discretization_grid, calculate_nodes0(
    testing::_,
    cells, 
    geom_size, 
    testing::Eq(0))).WillOnce(SetArg0(nodes0));

  Eigen::Tensor<double, 2> IPcoordinates0(3, 2);
  IPcoordinates0.setValues({
  {  1.1,  1.2 },
  {  2.1,  2.2 },
  {  3.1,  3.2 }
  });
  EXPECT_CALL(discretization_grid, calculate_ipCoordinates0(
    testing::_,
    cells, 
    geom_size, 
    testing::Eq(0))).WillOnce(SetArg0(IPcoordinates0));
  
  EXPECT_CALL(discretization_grid, discretization_init(
    testing::_, 
    testing::Eq(2),
    testing::_,
    testing::Eq(2),
    testing::_, 
    testing::Eq(12),
    testing::Eq(6)))
    .WillOnce(testing::Return());
  discretization_grid.init(false);
}

class CoordCalculationSetup : public ::testing::Test {
  protected:
    std::array<int, 3> cells = {2,1,1};
    std::array<double, 3> geom_size = {2e-5,1e-5,1e-5};
};

TEST_F(CoordCalculationSetup, TestCalculateIpCoords0) {
  DiscretizationGrid discretization_grid(std::array<int, 3>{2, 1, 1});

  Eigen::Tensor<double, 2> expected_IPcoordinates0(3, 2);
  expected_IPcoordinates0.setValues({
   {  5.0000000000000004e-06,  1.5000000000000002e-05 },
   {  5.0000000000000004e-06,  5.0000000000000004e-06 },
   {  5.0000000000000004e-06,  5.0000000000000004e-06 }
  });

  Eigen::Tensor<double, 2> IPcoordinates0(3, 2);
  discretization_grid.calculate_ipCoordinates0(IPcoordinates0, cells, geom_size, 0);
  EXPECT_TRUE(tensor_eq(IPcoordinates0, expected_IPcoordinates0));
}

TEST_F(CoordCalculationSetup, TestCalculateNodes0) {
  DiscretizationGrid discretization_grid(std::array<int, 3>{2, 1, 1});

  Eigen::Tensor<double, 2> expected_nodes0(3, 12);
  expected_nodes0.setValues({
   {  0                     ,  1.0000000000000001e-05,  2.0000000000000002e-05,  0,  
      1.0000000000000001e-05,  2.0000000000000002e-05,  0                     ,  1.0000000000000001e-05,  
      2.0000000000000002e-05,  0                     ,  1.0000000000000001e-05,  2.0000000000000002e-05 },
   {  0                     ,  0                     ,  0                     ,  1.0000000000000001e-05,  
      1.0000000000000001e-05,  1.0000000000000001e-05,  0                     ,  0,  
      0                     ,  1.0000000000000001e-05,  1.0000000000000001e-05,  1.0000000000000001e-05 },
   {  0                     ,  0                     ,  0                     ,  0,  
      0                     ,  0                     ,  1.0000000000000001e-05,  1.0000000000000001e-05,  
      1.0000000000000001e-05,  1.0000000000000001e-05,  1.0000000000000001e-05,  1.0000000000000001e-05 }
  });
  Eigen::Tensor<double, 2> nodes0(3, 12);
  discretization_grid.calculate_nodes0(nodes0, cells, geom_size, 0);
  EXPECT_TRUE(tensor_eq(nodes0, expected_nodes0));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}