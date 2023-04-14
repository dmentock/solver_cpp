#ifndef MOCK_DISCRETIZED_GRID_H
#define MOCK_DISCRETIZED_GRID_H

#include <gmock/gmock.h>
#include <discretization_grid.h>


class MockDiscretizedGrid : public DiscretizationGrid {
  public:
    MockDiscretizedGrid() 
      : DiscretizationGrid() {
          world_rank = 0;
          world_size = 1;
          cells[0] = 2; cells[1] = 3; cells[2] = 4;
          cells2 = 4;
          cells2Offset = 0;
          geom_size[0] = 1; geom_size[1] = 1; geom_size[2] = 1;
          size2 = 1;
          size2Offset = 0;
    }  

    MOCK_METHOD(void, f_discretization_init, (int*, int*, double*, double*, int*), (override));
    MOCK_METHOD(double*, calculate_nodes0, (int[3], double[3], int), (override));
    MOCK_METHOD(double*, calculate_ipCoordinates0, (int[3], double[3], int), (override));
};
#endif // MOCK_DISCRETIZED_GRID_H