#ifndef DISCRETIZATION_GRID_H
#define DISCRETIZATION_GRID_H

#include <vti_reader.h>

extern "C" {
    void __discretization_MOD_discretization_init(int *materialAt, int *discretization_Nelems, double *IPcoords0, double *NodeCoords0, int *sharedNodesBegin);
}

class DiscretizationGrid {
public:
    virtual void f_discretization_init(int *materialAt, int *discretization_Nelems, double *IPcoords0, double *NodeCoords0, int *sharedNodesBegin) {
      return __discretization_MOD_discretization_init(materialAt, discretization_Nelems, IPcoords0, NodeCoords0, sharedNodesBegin);
    }
    void init(bool restart, VtiReader* vti_reader);
    virtual double* calculate_nodes0(int cells[3], double geomSize[3], int cells2Offset);
    virtual double* calculate_ipCoordinates0(int cells[3], double geomSize[3], int cells2Offset);
};
#endif // DISCRETIZATION_GRID_H