#include <iostream>
#include <vector>
#include <numeric>

#include <mpi.h>
#include <fftw3-mpi.h>
#include <vti_reader.h>

#include "discretization_grid.h"

void DiscretizationGrid::init(bool restart, VtiReader* vti_reader) {
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::cout << "World Size: " << world_size << "   Rank: " << world_rank << std::endl;

    double origin[3];
    // int* materialAt_global = vti_reader->read_vti_material_data("17grains.vti", cells, geom_size, origin);
    int* materialAt_global;
    if (world_rank != 0) {
    }
    MPI_Bcast(cells.data(), 3, MPI_INTEGER, 0, MPI_COMM_WORLD);
    if (cells[0] < 2) {
        std::cerr << "cells(1) must be larger than 1" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(geom_size.data(), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(origin, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    std::cout << "cells:  " << cells[0] << " x " << cells[1] << " x " << cells[2] << std::endl;
    std::cout << "size:   " << geom_size[0] << " x " << geom_size[1] << " x " << geom_size[2] << " m³" << std::endl;
    std::cout << "origin: " << origin[0] << " " << origin[1] << " " << origin[2] << " m" << std::endl;

    if (world_size > cells[2]) {
        std::cerr << "number of processes exceeds cells[2]" << std::endl;
    }
    fftw_mpi_init();
    ptrdiff_t z, z_offset;
    fftw_mpi_local_size_3d(cells[2], cells[1], cells[1]/2+1, MPI_COMM_WORLD, &z, &z_offset);
    if (z == 0) {
        std::cerr << "Cannot distribute MPI processes" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    cells2 = (int)z;
    cells2_offset = z_offset;
    int local_grid[3] = {cells[0], cells[1], cells2};

    size2 = geom_size[2] * cells2 / cells[2];
    size2Offset = geom_size[2] * cells2_offset / cells[2];
    double local_size[3] = {geom_size[0], geom_size[1], size2};

    int local_displacement = cells[0] * cells[1] * cells2_offset;
    int displacements[world_size];
    MPI_Gather(&local_displacement, 1, MPI_INT, displacements, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_gridpoint_number = local_grid[0] * local_grid[1] * local_grid[2];
    int gridpoint_numbers[world_size];
    MPI_Gather(&local_gridpoint_number, 1, MPI_INT, gridpoint_numbers, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int materialAt[local_gridpoint_number];
    int a = MPI_Scatterv(materialAt_global, 
                        gridpoint_numbers, 
                        displacements, 
                        MPI_INTEGER, 
                        materialAt, 
                        local_gridpoint_number, 
                        MPI_INTEGER, 0, MPI_COMM_WORLD);

    double* ipCoordinates0 = this->calculate_ipCoordinates0(local_grid, local_size, cells2_offset);
    double* nodes0 = this->calculate_nodes0(local_grid, local_size, cells2_offset);
    int sharedNodesBegin = (world_rank+1 == world_size) ? (cells[0]+1) * (cells[1]+1) * cells2 : (cells[0]+1) * (cells[1]+1) * (cells2+1);
    this->discretization_.init(&materialAt[0],
                               &local_gridpoint_number,
                               ipCoordinates0, 
                               nodes0,
                               sharedNodesBegin);
}

double* DiscretizationGrid::calculate_ipCoordinates0(int cells[3], double geom_size[3], int cells2_offset){
    int N = cells[0]*cells[1]*cells[2];
    int i = 0;
    double* ipCoordinates0 = new double[3*N];
    for (int c = 0; c < cells[2]; ++c) {
        for (int b = 0; b < cells[1]; ++b) {
            for (int a = 0; a < cells[0]; ++a) {
                ipCoordinates0[3*i] = geom_size[0] / cells[0] * (a - 0.5);
                ipCoordinates0[3*i + 1] = geom_size[1] / cells[1] * (b - 0.5);
                ipCoordinates0[3*i + 2] = geom_size[2] / cells[2] * (c + cells2_offset - 0.5);
                i++;
            }
        }
    }
    return ipCoordinates0;
}

double* DiscretizationGrid::calculate_nodes0(int cells[3], double geom_size[3], int cells2_offset) {
    int N = cells[0]*cells[1]*cells[2];
    int n = 0;
    double* nodes0 = new double[3*N];
    for (int c = 0; c < cells[2]; ++c) {
        for (int b = 0; b < cells[1]; ++b) {
            for (int a = 0; a < cells[0]; ++a) {
                nodes0[3*n] = geom_size[0] / cells[0] * a;
                nodes0[3*n + 1] = geom_size[1] / cells[1] * b;
                nodes0[3*n + 2] = geom_size[2] / cells[2] * c + cells2_offset;
                n++;
            }
        }
    }
    return nodes0;
}

int DiscretizationGrid::modulo(int x,int N){
    return (x % N + N) %N;
}