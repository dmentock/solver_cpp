#ifndef DISCRETIZATION_H
#define DISCRETIZATION_H

#include <unsupported/Eigen/CXX11/Tensor>


// extern "C" {
//     void __discretization_MOD_discretization_init(int *materialAt, int *discretization_Nelems, double *IPcoords0, double *NodeCoords0, int *sharedNodesBegin);
// }

class Discretization {
public:
    virtual void init(int *materialAt, int *discretization_Nelems, double *IPcoords0, double *NodeCoords0, int sharedNodesBegin);
    virtual void set_ip_coords(Eigen::Tensor<double, 2>* node_coords);
    virtual void set_node_coords(Eigen::Tensor<double, 2>* ip_coords);
    // virtual void f_discretization_init(int *materialAt, int *discretization_Nelems, double *IPcoords0, double *NodeCoords0, int *sharedNodesBegin) {
    //   return __discretization_MOD_discretization_init(materialAt, discretization_Nelems, IPcoords0, NodeCoords0, sharedNodesBegin);
    // }
};
#endif // DISCRETIZATION_H