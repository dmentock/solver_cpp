#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <fstream> 

#include "init_environments.h"
#include "conftest.h"
#include "../utilities_tensor.h"

#include "../discretization_grid.h"

class PartialMockDiscretizationGrid : public DiscretizationGrid {
  public:
    using Tensor3i = Eigen::Tensor<int, 3>;
    using Tensor2d = Eigen::Tensor<double, 2>;
    using Tensor3d = Eigen::Tensor<double, 3>;
    using array3i = std::array<int, 3>;
    using array3d = std::array<double, 3>;
    MOCK_METHOD(Tensor2d, calculate_node_coords0, (array3i&, array3d&, int), (override));
    MOCK_METHOD(Tensor2d, calculate_ip_coords0, (array3i&, array3d&, int), (override));
    MOCK_METHOD(void, VTI_readCellsSizeOrigin, (std::string& file_content, 
                                                array3i& cells, 
                                                array3d& size,
                                                array3d& origin));
    MOCK_METHOD(Tensor3i, VTI_readDataset_int, (std::string& file_content, std::string& label, array3i& cells));
    MOCK_METHOD(void, discretization_init, (Tensor3i& material_at,
                                            Tensor2d& ip_coords0,
                                            Tensor2d& node_coords0,
                                            int shared_nodes_begin), (override));
};

TEST(DiscretizationGridSetup, TestInit) {

  PartialMockDiscretizationGrid discretization_grid;

  std::array<int, 3>cells_from_vti = {2,1,1};
  std::array<double, 3> geom_size_from_vti = {2e-5,1e-5,1e-5};
  std::array<double, 3> origin_from_vti = {0, 0, 0};
  EXPECT_CALL(discretization_grid, VTI_readCellsSizeOrigin(testing::_, testing::_, testing::_, testing::_))
    .WillOnce(testing::DoAll(
        testing::SetArgReferee<1>(cells_from_vti),
        testing::SetArgReferee<2>(geom_size_from_vti),
        testing::SetArgReferee<3>(origin_from_vti)
    ));

  std::string mock_vti_file_content = "mock vti file content";
  std::string vti_label = "material";
  Eigen::Tensor<int, 3> mocked_material_at(2, 1, 1);
  mocked_material_at.setValues({{{3}}, {{4}}});
  EXPECT_CALL(discretization_grid, VTI_readDataset_int(mock_vti_file_content, vti_label, cells_from_vti))
    .WillOnce(testing::Return(mocked_material_at));

  Eigen::Tensor<double, 2> ip_coords0(3, 2);
  ip_coords0.setValues({
  {  1.1,  1.2 },
  {  2.1,  2.2 },
  {  3.1,  3.2 }
  });
  EXPECT_CALL(discretization_grid, calculate_ip_coords0(
    cells_from_vti,
    geom_size_from_vti, 
    testing::Eq(0))).WillOnce(testing::Return(ip_coords0));
  
  Eigen::Tensor<double, 2> node_coords0(3, 12);
  node_coords0.setValues({
  {  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,  1.7,  1.8,  1.9,  2.0,  2.1,  2.2 },
  {  2.1,  2.2,  2.3,  2.4,  2.5,  2.6,  2.7,  2.8,  2.9,  3.0,  3.1,  3.2 },
  {  3.1,  3.2,  3.3,  3.4,  3.5,  3.6,  3.7,  3.8,  3.9,  4.0,  4.1,  4.2 }
  });
  EXPECT_CALL(discretization_grid, calculate_node_coords0(
    cells_from_vti, 
    geom_size_from_vti, 
    testing::Eq(0))).WillOnce(testing::Return(node_coords0));

  EXPECT_CALL(discretization_grid, discretization_init(
    TensorEq(mocked_material_at),
    TensorEq(ip_coords0),
    TensorEq(node_coords0),
    testing::Eq(6))).WillOnce(testing::DoDefault);
  fftw_mpi_init();
  discretization_grid.init(false, mock_vti_file_content);
}

TEST(CoordCalculationSetup, TestCalculateIpCoords0) {
  DiscretizationGrid discretization_grid;
  std::array<int, 3> cells{2, 1, 1};
  std::array<double, 3> geom_size = {2e-5,1e-5,1e-5};
  // discretization_grid.cells = cells;

  Eigen::Tensor<double, 2> expected_ip_coords0(3, 2);
  expected_ip_coords0.setValues({
   {  5.0000000000000004e-06,  1.5000000000000002e-05 },
   {  5.0000000000000004e-06,  5.0000000000000004e-06 },
   {  5.0000000000000004e-06,  5.0000000000000004e-06 }
  });

  Eigen::Tensor<double, 2> ip_coords0 = discretization_grid.calculate_ip_coords0(cells, geom_size, 0);
  EXPECT_TRUE(tensor_eq(ip_coords0, expected_ip_coords0));
}

TEST(CoordCalculationSetup, TestCalculateNodes0) {
  std::array<int, 3> cells{2, 1, 1};
  std::array<double, 3> geom_size = {2e-5,1e-5,1e-5};
  DiscretizationGrid discretization_grid;

  Eigen::Tensor<double, 2> expected_node_coords0(3, 12);
  expected_node_coords0.setValues({
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
  Eigen::Tensor<double, 2> node_coords0 = discretization_grid.calculate_node_coords0(cells, geom_size, 0);
  EXPECT_TRUE(tensor_eq(node_coords0, expected_node_coords0));
}


class VTISetup : public ::testing::Test {
 protected:
  std::string vti_path;
    std::string vti_file_content = (R""""(<?xml version="1.0"?>
  <VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
    <ImageData WholeExtent="0 16 0 16 0 16" Origin="0 0 0" Spacing="0.0625 0.0625 0.0625" Direction="1 0 0 0 1 0 0 0 1">
      <FieldData>
        <Array type="String" Name="comments" NumberOfTuples="2" format="binary">
          AQAAAACAAAAkAAAALAAAAA==eF7LzUwuyi8uKSpNLiktSi1WMDJgyMjPzU9PzcusSizJzM/jNGQAAASkDSY=
        </Array>
      </FieldData>
    <Piece Extent="0 16 0 16 0 16">
      <PointData>
      </PointData>
      <CellData>
        <DataArray type="Int64" Name="material" format="binary" RangeMin="0" RangeMax="19">
          AQAAAACAAAAAAAAADwYAAA==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
        </DataArray>
      </CellData>
    </Piece>
    </ImageData>
  </VTKFile>
    )"""");
};

TEST_F(VTISetup, TestVTI_readCellsSizeOrigin) {
  std::array<int, 3> cells;
  std::array<double, 3> size;
  std::array<double, 3> origin;
  std::array<int, 3> expected_cells = {16, 16, 16};
  std::array<double, 3> expected_origin = {0, 0, 0};
  std::array<double, 3> expected_size = {1, 1, 1};
  f_VTI_readCellsSizeOrigin(vti_file_content.c_str(), std::size(vti_file_content), cells.data(), size.data(), origin.data());
  EXPECT_EQ(cells, expected_cells);
  EXPECT_EQ(origin, expected_origin);
  EXPECT_EQ(size, expected_size);
}

TEST_F(VTISetup, TestVTI_readDataset_int) {
  int material_size = 16*16*16;
  std::array<int, 16*16*16> material_at;
  std::string label = "material";
  f_VTI_readDataset_int(vti_file_content.c_str(), std::size(vti_file_content),
                        label.c_str(), std::size(label),
                        material_at.data(), &material_size);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}
