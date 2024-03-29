#ifndef MOCK_DISCRETIZED_GRID_H
#define MOCK_DISCRETIZED_GRID_H
#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <discretization_grid.h>
#include <spectral.h>
#include <mech_base.h>
#include <config.h>
#include <petsc.h>
#include <unsupported/Eigen/CXX11/Tensor>


extern "C" {
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

    n_cells_global = cells[0] * cells[1] * cells[2];
    n_cells_local = cells[0] * cells[1] * cells2;
    n_nodes_local = (cells[0]+1) * (cells[1]+1) * (cells2+1);

    Tensor<double, 2> ip_coords_(3, n_cells_local);
    Tensor<double, 2> node_coords_(3, n_nodes_local);
    ip_coords = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 2>>>(ip_coords_.data(), 3, n_cells_local);
    node_coords = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 2>>>(node_coords_.data(), 3, n_nodes_local);
  }
};

class GridTestSetup : public ::testing::Test {
protected:
  Config config;
  PetscErrorCode ierr;
public:
  void gridTestSetup_init_discretization(MockDiscretizedGrid& mock_grid) {
    Eigen::Tensor<int, 1> materialAt(mock_grid.n_cells_global);
    for (size_t i = 0; i < mock_grid.n_cells_global; i++) {
      materialAt[i] = i;
    }
    Eigen::Tensor<double, 2> ipCoordinates0;
    mock_grid.calculate_ipCoordinates0(ipCoordinates0, mock_grid.cells, mock_grid.geom_size, 0);
    Eigen::Tensor<double, 2> nodes0;
    mock_grid.calculate_nodes0(nodes0, mock_grid.cells, mock_grid.geom_size, 0);
    int sharedNodesBegin = (mock_grid.world_rank+1 == mock_grid.world_size) ? 
      (mock_grid.cells[0]+1) * (mock_grid.cells[1]+1) * mock_grid.cells2 : 
      (mock_grid.cells[0]+1) * (mock_grid.cells[1]+1) * (mock_grid.cells2+1);
    mock_grid.discretization_init(materialAt.data(), mock_grid.n_cells_global,
                                  ipCoordinates0.data(), ipCoordinates0.dimension(1),
                                  nodes0.data(), nodes0.dimension(1),
                                  sharedNodesBegin);
  }

  void gridTestSetup_mock_homogenization_tensors(Spectral& spectral, int n_cells_local) {
    cout << "ww  " << n_cells_local << endl;
    mock_homogenization_F0.resize(3, 3, n_cells_local);
    mock_homogenization_F.resize(3, 3, n_cells_local);
    mock_homogenization_P.resize(3, 3, n_cells_local);
    mock_homogenization_dPdF.resize(3, 3, 3, 3, n_cells_local);
    spectral.homogenization_F0 = std::make_unique<TensorMap<Tensor<double, 3>>>(mock_homogenization_F0.data(), 3, 3, n_cells_local);
    spectral.homogenization_F = std::make_unique<TensorMap<Tensor<double, 3>>>(mock_homogenization_F.data(), 3, 3, n_cells_local);
    spectral.homogenization_P = std::make_unique<TensorMap<Tensor<double, 3>>>(mock_homogenization_P.data(), 3, 3, n_cells_local);
    spectral.homogenization_dPdF = std::make_unique<TensorMap<Tensor<double, 5>>>(mock_homogenization_dPdF.data(), 3, 3, 3, 3, n_cells_local);
    spectral.terminally_ill = static_cast<bool*>(&mock_terminally_ill);
  }

  template <int Rank>
  FFT<Rank>* gridTestSetup_init_fft (MockDiscretizedGrid& mock_grid) {
    ptrdiff_t cells2_fftw, cells1_fftw, cells1_offset;
    std::vector<int> extra_dims;
    for (Eigen::Index i = 0; i < Rank-3; ++i) {
      extra_dims.push_back(3);
    }
    FFT<Rank>* fft_obj = new FFT<Rank>(mock_grid.cells, mock_grid.cells2, extra_dims, 0, &cells1_fftw, &cells1_offset, &cells2_fftw);
    return fft_obj;
  }

  void gridTestSetup_set_up_dm(DM& da, Vec& vec, MockDiscretizedGrid& mock_grid) {
    DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2],
                1, 1, 1,
                9, 0,
                &mock_grid.cells[0], &mock_grid.cells[1], &mock_grid.cells[2], &da);
    DMSetFromOptions(da);
    DMSetUp(da);
    DMCreateGlobalVector(da, &vec);
  }

  template <typename TensorType>
  void gridTestSetup_set_solution_vec(TensorType& tensor, Vec& vec) {
    int ni = 1;
    for (int i = 0; i < tensor.NumDimensions; ++i) {
      ni *= tensor.dimension(i);
    }
    std::vector<int> ix(ni);
    std::iota(ix.begin()+1, ix.end(), 1);
    VecSetValues(vec, ni, ix.data(), tensor.data(), INSERT_VALUES);
    VecAssemblyBegin(vec);
    VecAssemblyEnd(vec);
  }

  void gridTestSetup_set_up_snes(SNES& snes) {
    SNESCreate(PETSC_COMM_WORLD, &snes);
    SNESSetOptionsPrefix(snes, "mechanical_");
  }

  bool mock_terminally_ill = false;
  Tensor<double, 3> mock_homogenization_F0;
  Tensor<double, 3> mock_homogenization_F;
  Tensor<double, 3> mock_homogenization_P;
  Tensor<double, 5> mock_homogenization_dPdF;
};

#endif // MOCK_DISCRETIZED_GRID_H