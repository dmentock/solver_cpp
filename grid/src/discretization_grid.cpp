#include <iostream>
#include <vector>
#include <numeric>

#include <mpi.h>
#include <fftw3-mpi.h>

#include "discretization_grid.h"

#include <iostream>
#include <fstream>

void DiscretizationGrid::init(bool restart) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    cout << "World Size: " << world_size << "   Rank: " << world_rank << endl;

    double origin[3];
    Eigen::Tensor<int, 1> materialAt_global;
    if (world_rank == 0) {
      // array<int, n_cells> materialAt_global;
      materialAt_global.resize(cells[0] * cells[1] * cells[2]);
      VTI_readDataset_int(materialAt_global);
      // call result_openJobFile(parallel=.false.)
      // call result_writeDataset_str(fileContent,'setup',fname,'geometry definition (grid solver)')
      // call result_closeJobFile()
      
    } else {
      materialAt_global.resize(0);
    }
    MPI_Bcast(cells.data(), 3, MPI_INTEGER, 0, MPI_COMM_WORLD);
    if (cells[0] < 2) {
      cerr << "cells[0] must be larger than 1" << endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(geom_size.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(origin, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    cout << "cells:  " << cells[0] << " x " << cells[1] << " x " << cells[2] << endl;
    cout << "size:   " << geom_size[0] << " x " << geom_size[1] << " x " << geom_size[2] << " m³" << endl;
    cout << "origin: " << origin[0] << " " << origin[1] << " " << origin[2] << " m" << endl;

    if (world_size > cells[2]) {
        cerr << "number of processes exceeds cells[2]" << endl;
    }
    fftw_mpi_init();
    ptrdiff_t z, z_offset;
    fftw_mpi_local_size_3d(cells[2], cells[1], cells[0]/2+1, MPI_COMM_WORLD, &z, &z_offset);
    if (z == 0) {
        cerr << "Cannot distribute MPI processes" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cells2 = (int)z;
    cells2_offset = z_offset;
    array<int, 3> local_grid = {cells[0], cells[1], cells2};
    size2 = geom_size[2] * cells2 / cells[2];
    size2Offset = geom_size[2] * cells2_offset / cells[2];
    array<double, 3> local_size = {geom_size[0], geom_size[1], size2};

    int local_displacement = cells[0] * cells[1] * cells2_offset;
    int displacements[world_size];
    MPI_Gather(&local_displacement, 1, MPI_INT, displacements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int n_materialpoints_local = local_grid[0] * local_grid[1] * local_grid[2];
    int gridpoint_numbers[world_size];
    MPI_Gather(&n_materialpoints_local, 1, MPI_INT, gridpoint_numbers, 1, MPI_INT, 0, MPI_COMM_WORLD);

    Eigen::Tensor<int, 1> materialAt(n_materialpoints_local);
    int a = MPI_Scatterv(materialAt_global.data(), 
                        gridpoint_numbers, 
                        displacements, 
                        MPI_INTEGER, 
                        materialAt.data(), 
                        n_materialpoints_local, 
                        MPI_INTEGER, 0, MPI_COMM_WORLD);

    Eigen::Tensor<double, 2> ipCoordinates0;
    calculate_ipCoordinates0(ipCoordinates0, local_grid, local_size, cells2_offset);
    Eigen::Tensor<double, 2> nodes0;
    calculate_nodes0(nodes0, local_grid, local_size, cells2_offset);
    int sharedNodesBegin = (world_rank+1 == world_size) ? (cells[0]+1) * (cells[1]+1) * cells2 : (cells[0]+1) * (cells[1]+1) * (cells2+1);
    discretization_init(materialAt.data(), n_materialpoints_local,
                        ipCoordinates0.data(), ipCoordinates0.dimension(1),
                        nodes0.data(), nodes0.dimension(1),
                        sharedNodesBegin);
}

void DiscretizationGrid::calculate_ipCoordinates0(Eigen::Tensor<double, 2>& ipCoordinates0, 
                                                  array<int, 3>& cells, 
                                                  array<double, 3>& geom_size, 
                                                  int cells2_offset){
  int N = cells[0] * cells[1] * cells[2];
  ipCoordinates0.resize(3, N);
  int i = 0;
  for (int c = 0; c < cells[2]; ++c) {
    for (int b = 0; b < cells[1]; ++b) {
      for (int a = 0; a < cells[0]; ++a) {
        ipCoordinates0(0, i) = geom_size[0] / cells[0] * (a - 0.5);
        ipCoordinates0(1, i) = geom_size[1] / cells[1] * (b - 0.5);
        ipCoordinates0(2, i) = geom_size[2] / cells[2] * (c + cells2_offset - 0.5);
        i++;
      }
    }
  }
}

void DiscretizationGrid::calculate_nodes0(Eigen::Tensor<double, 2>& nodes0, 
                                          array<int, 3>& cells, 
                                          array<double, 3>& geom_size, 
                                          int cells2_offset) {
  int N = (cells[0]+1) * (cells[1]+1) * (cells[2]+1);
  nodes0.resize(3, N);
  nodes0.setZero();
  int i = 0;
  for (int c = 0; c < cells[2]; ++c) {
    for (int b = 0; b < cells[1]; ++b) {
      for (int a = 0; a < cells[0]; ++a) {
        nodes0(0, i) = geom_size[0] / cells[0] * a;
        nodes0(1, i) = geom_size[1] / cells[1] * b;
        nodes0(2, i) = geom_size[2] / cells[2] * c + cells2_offset;
        i++;
      }
    }
  }
}

int DiscretizationGrid::modulo(int x,int N){
    return (x % N + N) %N;
}