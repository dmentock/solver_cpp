#ifndef DISCRETIZATION_GRID_H
#define DISCRETIZATION_GRID_H

#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;

extern "C" {
  void f_discretization_init (int* materialAt, int* n_materialpoints, 
                              double* IPcoords0, int* n_ips,
                              double* NodeCoords0, int* n_nodes, 
                              int* sharedNodesBegin);
  void f_discretization_setIPcoords(double* IPcoords0);
  void f_discretization_setNodeCoords(double* NodeCoords);
  void f_CLI_get_geomFilename(char* filename);
  void f_VTI_readDataset_int(int* dataset, int* size);
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

  std::array<double, 3> geom_size;
  std::array<double, 3> scaled_geom_size;
  double size2;
  double size2Offset;

  int cells0_reduced;
  int cells1_tensor;
  int cells1_offset_tensor;

  void init(bool restart);
  virtual void calculate_nodes0(Eigen::Tensor<double, 2>& nodes0,
                                std::array<int, 3>& cells,
                                std::array<double, 3>& geom_size,
                                int cells2_offset);
  virtual void calculate_ipCoordinates0(Eigen::Tensor<double, 2>& ipCoordinates0, 
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
  virtual void VTI_readDataset_int(Eigen::Tensor<int, 1>& materialAt_global) {
    int size = materialAt_global.dimension(1);
    f_VTI_readDataset_int(materialAt_global.data(), &size);
  }
  void discretization_setIPcoords(double* IPcoords0);
  void discretization_setNodeCoords(double* NodeCoords);
    
};
#endif // DISCRETIZATION_GRID_H