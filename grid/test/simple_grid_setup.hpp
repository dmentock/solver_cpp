#ifndef MOCK_DISCRETIZED_GRID_H
#define MOCK_DISCRETIZED_GRID_H
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <discretization_grid.h>
#include <spectral/spectral.h>
#include "spectral/mech/utilities.h"
#include <config.h>
#include <petsc.h>
#include <unsupported/Eigen/CXX11/Tensor>


extern "C" {
  void f_homogenization_init();
  void f_deallocate_resources();
}

using Tensor2 = Eigen::Tensor<double, 2>;
using array3i = std::array<int, 3>;
using array3d = std::array<double, 3>;

class MockDiscretizedGrid : public DiscretizationGrid {
  public:
  MockDiscretizedGrid(std::array<int, 3> cells_)
    : DiscretizationGrid(cells_) {
    world_rank = 0;
    world_size = 1;

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
};

class SimpleGridSetup : public ::testing::Test {
protected:
  std::unique_ptr<MockDiscretizedGrid> mock_grid;
  Config config;
  PetscErrorCode ierr;
public:
  void gridSetup_init_grid(std::array<int, 3> dims) {
    mock_grid = std::make_unique<MockDiscretizedGrid>(dims);
  }
  void gridSetup_init_discretization() {
    Eigen::Tensor<int, 1> materialAt(mock_grid->n_cells_global);
    for (size_t i = 0; i < mock_grid->n_cells_global; i++) {
      materialAt[i] = i;
    }
    Eigen::Tensor<double, 2> ipCoordinates0;
    mock_grid->calculate_ipCoordinates0(ipCoordinates0, mock_grid->cells, mock_grid->geom_size, 0);
    Eigen::Tensor<double, 2> nodes0;
    mock_grid->calculate_nodes0(nodes0, mock_grid->cells, mock_grid->geom_size, 0);
    int sharedNodesBegin = (mock_grid->world_rank+1 == mock_grid->world_size) ? 
      (mock_grid->cells[0]+1) * (mock_grid->cells[1]+1) * mock_grid->cells2 : 
      (mock_grid->cells[0]+1) * (mock_grid->cells[1]+1) * (mock_grid->cells2+1);
    mock_grid->discretization_init(materialAt.data(), mock_grid->n_cells_global,
                        ipCoordinates0.data(), ipCoordinates0.dimension(1),
                        nodes0.data(), nodes0.dimension(1),
                        sharedNodesBegin);
  }
  void gridSetup_init_tensorfield(Spectral& spectral, MockDiscretizedGrid& mock_grid) {
    ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
    spectral.set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, 9, 
                       spectral.tensorField_real, spectral.tensorField_fourier, spectral.tensorField_fourier_fftw,
                       FFTW_MEASURE, spectral.plan_tensor_forth, spectral.plan_tensor_back,
                       "tensor");
    mock_grid.cells1_tensor = cells1_fftw;
    mock_grid.cells1_offset_tensor = cells1_offset;
  }
  void gridSetup_init_vectorfield(Spectral& spectral, MockDiscretizedGrid& mock_grid){
    ptrdiff_t cells1_fftw, cells1_offset, cells2_fftw;
    spectral.set_up_fftw(cells1_fftw, cells1_offset, cells2_fftw, 9, 
                       spectral.vectorField_real, spectral.vectorField_fourier, spectral.vectorField_fourier_fftw,
                       FFTW_MEASURE, spectral.plan_vector_forth, spectral.plan_vector_back,
                       "vector");
  }
};

#endif // MOCK_DISCRETIZED_GRID_H