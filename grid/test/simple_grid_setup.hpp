#ifndef MOCK_DISCRETIZED_GRID_H
#define MOCK_DISCRETIZED_GRID_H
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <discretization.h>
#include <discretization_grid.h>
#include <spectral/spectral.h>
#include <config.h>

class MockDiscretization : public Discretization {
  public:
    using Tensor2 = Eigen::Tensor<double, 2>;
    MOCK_METHOD(void, init, (int*, int*, double*, double*, int), (override));
    MOCK_METHOD(void, set_ip_coords, (Tensor2*));
    MOCK_METHOD(void, set_node_coords, (Tensor2*));
};

class MockDiscretizedGrid : public DiscretizationGrid {
  public:
    MockDiscretizedGrid(Discretization& discretization, 
                        std::array<int, 3> cells_) : DiscretizationGrid(discretization) {
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
    MOCK_METHOD(double*, calculate_nodes0, (int[3], double[3], int), (override));
    MOCK_METHOD(double*, calculate_ipCoordinates0, (int[3], double[3], int), (override));
};

class SimpleGridSetup : public ::testing::Test {
protected:
  MockDiscretization mock_discretization;
  std::unique_ptr<MockDiscretizedGrid> mock_grid;
  Config config;
public:
  void init_grid(std::array<int, 3> dims) {
    mock_grid = std::make_unique<MockDiscretizedGrid>(mock_discretization, dims);
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
};

#endif // MOCK_DISCRETIZED_GRID_H