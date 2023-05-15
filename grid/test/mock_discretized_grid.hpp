#ifndef MOCK_DISCRETIZED_GRID_H
#define MOCK_DISCRETIZED_GRID_H

#include <gmock/gmock.h>
#include <discretization.h>
#include <discretization_grid.h>

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
                        std::array<int, 3> cells_, 
                        std::array<double, 3> geom_size_) : DiscretizationGrid(discretization) {
      world_rank = 0;
      world_size = 1;
      cells = cells_;
      scaled_geom_size[0] = cells_[0]; scaled_geom_size[1] = cells_[1]; scaled_geom_size[2] = cells_[2];
      cells2 = cells[2];
      cells2_offset = 0;  
      geom_size = geom_size_;
      size2 = geom_size[2] * cells2 / cells[2];
      size2Offset = geom_size[2] * cells2_offset / cells[2];
      cells0_reduced = cells[0] / 2 + 1;
    }  
    MOCK_METHOD(double*, calculate_nodes0, (int[3], double[3], int), (override));
    MOCK_METHOD(double*, calculate_ipCoordinates0, (int[3], double[3], int), (override));
};

#endif // MOCK_DISCRETIZED_GRID_H