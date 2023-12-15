#include <iostream>

#include <mpi.h>
#include <fftw3-mpi.h>

#include "discretization_grid.h"

void DiscretizationGrid::init(bool restart, std::string& vti_file_content) {
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  cout << "World Size: " << world_size << "   Rank: " << world_rank << endl;

  Tensor<int, 3>material_at_global;
  if (world_rank == 0) {
    VTI_readCellsSizeOrigin(vti_file_content, cells, geom_size, origin);
    n_cells_global = cells[0] * cells[1] * cells[2];
    cells0_reduced = cells[0]/2+1;

    std::string label = "material";
    material_at_global = VTI_readDataset_int(vti_file_content, label, cells);

    // TODO
    // call result_openJobFile(parallel=.false.)
    // call result_writeDataset_str(fileContent,'setup',fname,'geometry definition (grid solver)')
    // call result_closeJobFile()

  }
  MPI_Bcast(cells.data(), 3, MPI_INTEGER, 0, MPI_COMM_WORLD);
  if (cells[0] < 2) {
    cerr << "cells[0] must be larger than 1" << endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Bcast(geom_size.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(origin.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  cout << "cells:  " << cells[0] << " x " << cells[1] << " x " << cells[2] << endl;
  cout << "size:   " << geom_size[0] << " x " << geom_size[1] << " x " << geom_size[2] << " m³" << endl;
  cout << "origin: " << origin[0] << " " << origin[1] << " " << origin[2] << " m" << endl;

  if (world_size > cells[2]) {
      cerr << "number of processes exceeds number of cells in z-direction" << endl;
  }
  ptrdiff_t z, z_offset;
  fftw_mpi_local_size_3d(cells[2], cells[1], cells[0]/2+1, MPI_COMM_WORLD, &z, &z_offset);
  if (z == 0) {
      cerr << "Cannot distribute MPI processes" << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
  }

  cells2 = (int)z;
  cells_local_offset_z = (int)z_offset;
  cells_local = {cells[0], cells[1], cells2};
  size2 = geom_size[2] * cells2 / cells[2];
  geom_size_local_offset_z = geom_size[2] * cells_local_offset_z / cells[2];

  n_cells_local = cells[0] * cells[1] * cells2;
  n_nodes_local = (cells[0]+1) * (cells[1]+1) * (cells2+1);

  geom_size_local = {geom_size[0], geom_size[1], size2};

  int local_displacement = cells[0] * cells[1] * cells_local_offset_z;
  std::vector<int> displacements(world_size);
  MPI_Gather(&local_displacement, 1, MPI_INT, displacements.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  int n_materialpoints_local = cells_local[0] * cells_local[1] * cells_local[2];
  std::vector<int> gridpoint_numbers(world_size);
  MPI_Gather(&n_materialpoints_local, 1, MPI_INT, gridpoint_numbers.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

  Tensor<int, 3> material_at(cells_local[0], cells_local[1], cells_local[2]);
  MPI_Scatterv(material_at_global.data(), 
               gridpoint_numbers.data(), 
               displacements.data(), 
               MPI_INTEGER, 
               material_at.data(), 
               n_materialpoints_local, 
               MPI_INTEGER, 0, MPI_COMM_WORLD);

  Tensor<double, 2> ip_coords0;
  ip_coords0 = calculate_ip_coords0(cells_local, geom_size_local, cells_local_offset_z);
  Tensor<double, 2> node_coords0;
  node_coords0 = calculate_node_coords0(cells_local, geom_size_local, cells_local_offset_z);
  int shared_nodes_begin = (world_rank+1 == world_size) ? (cells[0]+1) * (cells[1]+1) * cells2 : (cells[0]+1) * (cells[1]+1) * (cells2+1);
  discretization_init(material_at, ip_coords0, node_coords0, shared_nodes_begin);
  fetch_ip_node_coord_pointers(n_cells_local, n_nodes_local);
}

Tensor<double, 2> DiscretizationGrid::calculate_ip_coords0(std::array<int, 3>& cells, 
                                                           std::array<double, 3>& geom_size, 
                                                           int cells_local_offset_z) {
  int N = cells[0] * cells[1] * cells[2];
  Tensor<double, 2> ip_coords0(3, N);
  int i = 0;
  for (int c = 0; c < cells[2]; ++c) {
    for (int b = 0; b < cells[1]; ++b) {
      for (int a = 0; a < cells[0]; ++a) {
        ip_coords0(0, i) = geom_size[0] / cells[0] * (a + 0.5);
        ip_coords0(1, i) = geom_size[1] / cells[1] * (b + 0.5);
        ip_coords0(2, i) = geom_size[2] / cells[2] * (c + cells_local_offset_z + 0.5);
        i++;
      }
    }
  }
  return ip_coords0;
}

Tensor<double, 2> DiscretizationGrid::calculate_node_coords0(std::array<int, 3>& cells, 
                                                       std::array<double, 3>& geom_size, 
                                                       int cells_local_offset_z) {
  int N = (cells[0]+1) * (cells[1]+1) * (cells[2]+1);
  Tensor<double, 2> node_coords0(3, N);
  node_coords0.setZero();
  int i = 0;
  for (int c = 0; c <= cells[2]; ++c) {
    for (int b = 0; b <= cells[1]; ++b) {
      for (int a = 0; a <= cells[0]; ++a) {
        node_coords0(0, i) = geom_size[0] / cells[0] * a;
        node_coords0(1, i) = geom_size[1] / cells[1] * b;
        node_coords0(2, i) = geom_size[2] / cells[2] * (c + cells_local_offset_z);
        i++;
      }
    }
  }
  return node_coords0;
}

int DiscretizationGrid::modulo(int x,int N) {
    return (x % N + N) %N;
}
