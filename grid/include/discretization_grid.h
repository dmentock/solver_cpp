#ifndef DISCRETIZATION_GRID_H
#define DISCRETIZATION_GRID_H

#include <vti_reader.h>
#include <discretization.h>
#include <unsupported/Eigen/CXX11/Tensor>

class DiscretizationGrid {
public:
    DiscretizationGrid(Discretization& discretization) : discretization_(discretization) {}
    int world_rank, world_size;

    int cells[3];
    int cells2;
    int cells2_offset;

    double geom_size[3];
    double scaled_geom_size[3];
    double size2;
    double size2Offset;

    int cells0_reduced;
    int cells1_tensor;
    int cells1_offset_tensor;

    void init(bool restart, VtiReader* vti_reader);
    virtual double* calculate_nodes0(int cells[3], double geom_size[3], int cells2_offset);
    virtual double* calculate_ipCoordinates0(int cells[3], double geom_size[3], int cells2_offset);
    static int modulo(int x,int N);

    Discretization& discretization_;
};
#endif // DISCRETIZATION_GRID_H