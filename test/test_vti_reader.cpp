#include <gtest/gtest.h>
#include <fstream> 

#include <mpi.h>

#include "vti_reader.h"

class SetupVtiReadFromFile : public ::testing::Test {
 protected:
  // contains ordered 3x3x3 vti grid with unique points from 1 to 27, origin [0,0,0] and size [1,1,1] created with python preprocessing
  std::string tmpFilePath;
  void SetUp() override {
    MPI_Init(NULL, NULL);
    tmpFilePath = "tmp.vti";
    std::ofstream tmpFile(tmpFilePath.c_str());
    const char *vti_file = R""""(<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <ImageData WholeExtent="0 3 0 3 0 3" Origin="0 0 0" Spacing="0.3333333333333333 0.3333333333333333 0.3333333333333333" Direction="1 0 0 0 1 0 0 0 1">
    <FieldData>
      <Array type="String" Name="comments" NumberOfTuples="0" format="binary">
        AAAAAACAAAAAAAAA
      </Array>
    </FieldData>
  <Piece Extent="0 3 0 3 0 3">
    <PointData>
    </PointData>
    <CellData>
      <DataArray type="Int64" Name="material" format="binary" RangeMin="1" RangeMax="27">
        AQAAAACAAADYAAAAOQAAAA==eF4txbUBgDAAADAcikvx//9kIFmSJr/WmwsPPlx79u3MnaNLjz7dePHj3L13V558OXj16w+XCAF7
      </DataArray>
    </CellData>
  </Piece>
  </ImageData>
</VTKFile>)"""";
    tmpFile << vti_file;
    tmpFile.close();
  }

  void TearDown() override {
    MPI_Finalize();
    std::remove(tmpFilePath.c_str());
  }
};

// test expected functionality of VtiReader.read_vti_material_data
TEST_F(SetupVtiReadFromFile, VtiReadFromFile) {
    int cells[3];
    double geomSize[3];
    double origin[3];
    VtiReader vti_reader;
    int* result_grid = vti_reader.read_vti_material_data(tmpFilePath.c_str(), cells, geomSize, origin);
    int expected_grid[3][3][3] = {{{1, 10, 19}, {4, 13, 22}, {7, 16, 25}},
                                  {{2, 11, 20}, {5, 14, 23}, {8, 17, 26}},
                                  {{3, 12, 21}, {6, 15, 24}, {9, 18, 27}}};
    EXPECT_TRUE (std::equal(&expected_grid[0][0][0], &expected_grid[0][0][0] + 27, result_grid));
    int expected_cells[3] = {3, 3, 3};
    EXPECT_TRUE (std::equal(cells, cells + 3, expected_cells));
    int expected_geomSize[3] = {1, 1, 1};
    EXPECT_TRUE (std::equal(geomSize, geomSize + 3, expected_geomSize));
    int expected_origin[3] = {0, 0, 0};
    EXPECT_TRUE (std::equal(origin, origin + 3, expected_origin));
}

//TODO: Add test where vti file is created and read directly from memory