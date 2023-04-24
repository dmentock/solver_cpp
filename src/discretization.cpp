#include <unsupported/Eigen/CXX11/Tensor>
#include <discretization.h>
#include <cstdio>
#include <helper.h>


//// make wrapper around fortran functions here where needed

void Discretization::init(int *materialAt, int *discretization_Nelems, double *IPcoords0, double *NodeCoords0, int sharedNodesBegin){

}
void Discretization::set_ip_coords(Eigen::Tensor<double, 2>* node_coords){

}
void Discretization::set_node_coords(Eigen::Tensor<double, 2>* ip_coords){

}

