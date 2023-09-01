#ifndef DISCRETIZATION_GRID_H
#define DISCRETIZATION_GRID_H

#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

extern "C" {
  void f_discretization_init (int* materialAt, int* n_materialpoints, 
                              double* IPcoords0, int* n_ips,
                              double* NodeCoords0, int* n_nodes, 
                              int* sharedNodesBegin);
  void f_discretization_setIPcoords(double* IPcoords0);
  void f_discretization_setNodeCoords(double* NodeCoords);
  void f_CLI_get_geomFilename(char* filename);
  void f_VTI_readDataset_int(int* dataset, int* size);
  void f_discretization_fetch_ip_node_coord_pointers(double** ip_coords_ptr, double** node_coords_ptr);
}

class DiscretizationGrid {
public:
  DiscretizationGrid(std::array<int, 3> cells_)
      : cells(cells_) {}
  int world_rank, world_size;

  std::array<int, 3> cells;
  int cells2;
  int cells2_offset;
  int n_cells_global = cells[0] * cells[1] * cells[2];
  int n_cells_local = cells[0] * cells[1] * cells2;
  int n_nodes_local = (cells[0]+1) * (cells[1]+1) * (cells2+1);

  std::array<double, 3> geom_size;
  std::array<double, 3> scaled_geom_size;
  double size2;
  double size2Offset;

  int cells0_reduced;
  int cells1_tensor;
  int cells1_offset_tensor;

  void init(bool restart);
  virtual void calculate_nodes0(Tensor<double, 2>& nodes0,
                                std::array<int, 3>& cells,
                                std::array<double, 3>& geom_size,
                                int cells2_offset);
  virtual void calculate_ipCoordinates0(Tensor<double, 2>& ipCoordinates0, 
                                        std::array<int, 3>& cells, 
                                        std::array<double, 3>& geom_size, 
                                        int cells2_offset);
  static int modulo(int x,int N);

  // template <int n_materialpoints, int n_ips, int n_nodes>
  // virtual void discretization_init (std::array<int, n_materialpoints> materialAt, 
  //                           std::array<double, n_ips> IPcoords0,
  //                           std::array<double, n_nodes> NodeCoords0,
  //                           int sharedNodesBegin) {
  //   f_discretization_init(materialAt.data(), &n_materialpoints, 
  //                         IPcoords0.data(), &n_ips, 
  //                         NodeCoords0.data(), &n_nodes, 
  //                         &sharedNodesBegin);
  // }

  virtual void discretization_init (int* materialAt, int n_materialpoints,
                                    double* IPcoords0, int n_ips,
                                    double* NodeCoords0, int n_nodes,
                                    int sharedNodesBegin) {
  f_discretization_init(materialAt, &n_materialpoints, 
                        IPcoords0, &n_ips,
                        NodeCoords0, &n_nodes,
                        &sharedNodesBegin);

  }
  virtual void VTI_readDataset_int(Tensor<int, 1>& materialAt_global) {
    int size = materialAt_global.dimension(1);
    f_VTI_readDataset_int(materialAt_global.data(), &size);
  }
  void discretization_setIPcoords(double* IPcoords0);
  void discretization_setNodeCoords(double* NodeCoords);

  virtual void fetch_ip_node_coord_pointers(int n_ip_coords, int n_node_coords) {
    double* ip_coords_ptr;
    double* node_coords_ptr;
    f_discretization_fetch_ip_node_coord_pointers(&ip_coords_ptr, &node_coords_ptr);
    ip_coords = std::make_unique<TensorMap<Tensor<double, 2>>>(ip_coords_ptr, 3, n_ip_coords);
    node_coords = std::make_unique<TensorMap<Tensor<double, 2>>>(node_coords_ptr, 3, n_node_coords);
  }
  std::unique_ptr<TensorMap<Tensor<double, 2>>> ip_coords;
  std::unique_ptr<TensorMap<Tensor<double, 2>>> node_coords;
};
#endif // DISCRETIZATION_GRID_H