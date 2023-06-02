#ifndef MOCK_DISCRETIZED_GRID_H
#define MOCK_DISCRETIZED_GRID_H
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <discretization_grid.h>
#include <spectral/spectral.h>
#include "spectral/mech/utilities.h"
#include <config.h>

extern "C" {
  void f_homogenization_init (double* homogenization_F, int* dims_homog_F, 
                              double* homogenization_P, int* dims_homog_P,
                              void** terminally_ill);
}

using Tensor2 = Eigen::Tensor<double, 2>;
using array3i = std::array<int, 3>;
using array3d = std::array<double, 3>;
class MockDiscretizedGrid : public DiscretizationGrid {
  public:
  MockDiscretizedGrid(std::array<int, 3> cells_) : DiscretizationGrid() {
      world_rank = 0;
      world_size = 1;

      cells = cells_;

      cells0_reduced = cells[0] / 2 + 1;

      cells1_tensor = cells[1];
      cells1_offset_tensor = 0;
      
      cells2 = cells[2];
      cells2_offset = 0;  

      geom_size[0] = cells_[0]*1e-5; geom_size[1] = cells_[1]*1e-5; geom_size[2] = cells_[2]*1e-5;
      scaled_geom_size[0] = cells_[0]; scaled_geom_size[1] = cells_[1]; scaled_geom_size[2] = cells_[2];
      size2 = geom_size[2] * cells2 / cells[2];
      size2Offset = geom_size[2] * cells2_offset / cells[2];

    }  
    MOCK_METHOD(void, calculate_nodes0, (Tensor2&, array3i&, array3d&, int), (override));
    MOCK_METHOD(void, calculate_ipCoordinates0, (Tensor2&, array3i&, array3d&, int), (override));
};

class SimpleGridSetup : public ::testing::Test {
protected:
  std::unique_ptr<MockDiscretizedGrid> mock_grid;
  Config config;
public:
  void init_grid(std::array<int, 3> dims) {
    mock_grid = std::make_unique<MockDiscretizedGrid>(dims);
  }
  void init_tensorfield(Spectral& spectral, MockDiscretizedGrid& mock_grid) {
    ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
    spectral.set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, 9, 
                       spectral.tensorField_real, spectral.tensorField_fourier, spectral.tensorField_fourier_fftw,
                       FFTW_MEASURE, spectral.plan_tensor_forth, spectral.plan_tensor_back,
                       "tensor");
    mock_grid.cells1_tensor = cells1_fftw;
    mock_grid.cells1_offset_tensor = cells1_offset;
  }
  void init_vectorfield(Spectral& spectral, MockDiscretizedGrid& mock_grid){
    ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
    spectral.set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, 9, 
                       spectral.vectorField_real, spectral.vectorField_fourier, spectral.vectorField_fourier_fftw,
                       FFTW_MEASURE, spectral.plan_vector_forth, spectral.plan_vector_back,
                       "vector");
  }

  void* raw_void_pointer;
  void init_minimal_homogenization(Spectral &spectral) {
    spectral.homogenization_F.resize(3,3,2);
    spectral.homogenization_F.setZero();
    spectral.homogenization_P.resize(3,3,2);
    spectral.homogenization_P.setZero();
    std::array<int, 3> shape = {3,3,2};
    f_homogenization_init(spectral.homogenization_F.data(), shape.data(),
                          spectral.homogenization_P.data(), shape.data(),
                          &raw_void_pointer);
    spectral.terminally_ill = static_cast<bool*>(raw_void_pointer);
    std::cout << "QQ " << *spectral.terminally_ill << std::endl;
  }
};

#endif // MOCK_DISCRETIZED_GRID_H