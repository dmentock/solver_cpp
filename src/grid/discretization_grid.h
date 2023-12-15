#ifndef DISCRETIZATION_GRID_H
#define DISCRETIZATION_GRID_H

#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;
using namespace Eigen;

extern "C" {
  void f_discretization_init (int* material_at, int* n_materialpoints, 
                              double* IPcoords0, int* n_ips,
                              double* NodeCoords0, int* n_nodes, 
                              int* shared_nodes_begin);
  void f_discretization_setIPcoords(double* IPcoords0);
  void f_discretization_setNodeCoords(double* NodeCoords);
  void f_CLI_get_geomFilename(char* filename);
  // void f_VTI_readDataset_int(int* dataset, int* size);
  void f_discretization_fetch_ip_node_coord_pointers(double** ip_coords_ptr, double** node_coords_ptr);
  void f_VTI_readCellsSizeOrigin(const char* fileContent, int string_size, int* cells, double* size, double* origin);
  void f_VTI_readDataset_int(const char* fileContent, int string_size, const char* label, int label_length, 
                             int* dataset_ptr, int* dataset_size);
}

class DiscretizationGrid {
public:
  // class variables
  std::array<int, 3> cells;
  std::array<double, 3> geom_size;
  std::array<double, 3> origin;

  int n_cells_global;

  // mpi variables
  int world_rank, world_size;

  std::array<int, 3> cells_local;
  int cells_local_offset_z;
  int cells2;

  std::array<double, 3> geom_size_local;
  double geom_size_local_offset_z;
  double size2;

  int n_cells_local;
  int n_nodes_local;

  // variebles depending on divergence_correction
  std::array<double, 3> scaled_geom_size;

  // // fourier field dimensions
  int cells0_reduced;
  int cells1_tensor;
  int cells1_offset_tensor;

  void init(bool restart, std::string& vti_file_content);
  virtual Tensor<double, 2> calculate_node_coords0(std::array<int, 3>& cells,
                                             std::array<double, 3>& geom_size,
                                             int cells_local_offset_z);
  virtual Tensor<double, 2> calculate_ip_coords0(std::array<int, 3>& cells, 
                                                 std::array<double, 3>& geom_size, 
                                                 int cells_local_offset_z);
  static int modulo(int x,int N);

  virtual void VTI_readCellsSizeOrigin(std::string& file_content, 
                                       std::array<int, 3>& cells_, std::array<double, 3>& size_, std::array<double, 3>& origin_) {
    f_VTI_readCellsSizeOrigin(file_content.c_str(), std::size(file_content), cells_.data(), size_.data(), origin_.data());
  }

  virtual Tensor<int, 3> VTI_readDataset_int(std::string& file_content, std::string& label, std::array<int, 3>& cells) {
    Tensor<int, 3> material_at_global(cells[0], cells[1], cells[2]);
    int n_cells = cells[0] * cells[1] * cells[2];
    f_VTI_readDataset_int(file_content.c_str(), std::size(file_content), 
                          label.c_str(), std::size(label), 
                          material_at_global.data(), &n_cells);
    return material_at_global;
  };

  virtual void discretization_init(Tensor<int, 3>& material_at,
                                   Tensor<double, 2>& ip_coords0,
                                   Tensor<double, 2>& node_coords0,
                                   int shared_nodes_begin) {
    int n_materialpoints = material_at.dimension(0) * material_at.dimension(1) * material_at.dimension(2);
    int n_ip_coords = ip_coords0.dimension(1);
    int n_node_coords = node_coords0.dimension(1);
    f_discretization_init(material_at.data(), &n_materialpoints, 
                          ip_coords0.data(), &n_ip_coords,
                          node_coords0.data(), &n_node_coords,
                          &shared_nodes_begin);
  }

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
