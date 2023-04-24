#include <petscsnes.h>
#include <petscdmda.h>
#include <petscdm.h>
#include <mpi.h>
#include <vector>
#include <array>
#include <string>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "spectral/mech/basic.h"
#include "spectral/spectral.h"

void Basic::init()
{

    std::cout << "\n <<<+-  grid_mechanical_spectral_basic init  -+>>>\n";
    std::cout << "P. Eisenlohr et al., International Journal of Plasticity 46:37–53, 2013\n";
    std::cout << "https://doi.org/10.1016/j.ijplas.2012.09.012\n\n";

    std::cout << "P. Shanthraj et al., International Journal of Plasticity 66:31–45, 2015\n";
    std::cout << "https://doi.org/10.1016/j.ijplas.2014.02.006\n";

    // read numerical parameters and do sanity checks 132-149

    PetscOptionsInsertString(NULL, "-mechanical_snes_type ngmres");
    // insert more options from yaml 157-58

    SNES SNES_mechanical;
    SNESCreate(PETSC_COMM_WORLD, &SNES_mechanical);
    SNESSetOptionsPrefix(SNES_mechanical, "mechanical_");

    int localK[grid.world_size];
    localK[grid.world_rank] = grid.cells2;

    std::cout << "hohohoh " << grid.world_size << " " << localK[0] << std::endl;
    MPI_Allreduce(MPI_IN_PLACE, localK, grid.world_size, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    DM da;
    int lx[1] = {grid.cells[0]};
    int ly[1] = {grid.cells[1]};
    DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX,
                 grid.cells[0], grid.cells[1], grid.cells[2],
                 1, 1, grid.world_size,
                 9, 0,
                 lx, ly, localK, &da);

    DMSetFromOptions(da);
    DMSetUp(da);
    Vec solution_vec;
    DMCreateGlobalVector(da, &solution_vec);
    DMDASNESSetFunctionLocal(da, INSERT_VALUES, this->formResidual, PETSC_NULLPTR);
    // Specify custom convergence check function "converged"
    SNESSetConvergenceTest(SNES_mechanical, this->converged, PETSC_NULL, NULL);
    // Set the DMDA context
    SNESSetDM(SNES_mechanical, da);
    // Pull it all together with additional CLI arguments
    SNESSetFromOptions(SNES_mechanical);

    // if CLI_restartInc == 0 ... 216-242
    F_lastInc.resize(Eigen::array<Eigen::Index, 5>{3, 3, grid.cells[0], grid.cells[1], grid.cells[2]});
    F_dot.resize(Eigen::array<Eigen::Index, 5>{3, 3, grid.cells[0], grid.cells[1], grid.cells[2]});

    // Initialize the tensor with the 3x3 identity matrix repeated across the last three dimensions
    Eigen::Matrix3d math_I3 = Eigen::Matrix3d::Identity();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            F_lastInc.slice(Eigen::array<Eigen::Index, 5>({i, j, 0, 0, 0}),
                            Eigen::array<Eigen::Index, 5>({1, 1, grid.cells[0], grid.cells[1], grid.cells[2]}))
                .setConstant(math_I3(i, j));
        }
    }

    Eigen::Tensor<double, 4> F(9, grid.cells[0], grid.cells[1], grid.cells[2]);
    F = F_lastInc.reshape(Eigen::array<int, 4>({9, grid.cells[0], grid.cells[1], grid.cells[2]}));
    PetscScalar *F_data = F.data();

    DMDAVecGetArray(da, solution_vec, &F_data); //moved down because its recommended to do this on already initialized arrays

    Eigen::Tensor<double, 3> homogenization_F0(3, 3, grid.cells[0] * grid.cells[1] * grid.cells[2]);
    homogenization_F0 = F_lastInc.reshape(Eigen::array<int, 3>({3, 3, grid.cells[0] * grid.cells[1] * grid.cells[2]}));
    Spectral::update_coords(F_lastInc); // fix after implementing restart

    Eigen::Tensor<double, 5> P(3, 3, grid.cells[0], grid.cells[1], grid.cells[2]);
    Basic::P_av(3, 3);
    Basic::C_volAvg(3, 3, 3, 3);
    Basic::C_minMaxAvg(3, 3, 3, 3);
    Spectral::constitutive_response(P,
                                    Basic::P_av,
                                    Basic::C_volAvg,
                                    Basic::C_minMaxAvg, // stress field, stress avg, global average of stiffness and (min+max)/2
                                    F_lastInc,          // target F
                                    0);
    DMDAVecRestoreArray(da, solution_vec, &F_data);
    // add restart calls 253-268
    Spectral::update_gamma(Basic::C_minMaxAvg);
    Basic::C_minMaxAvgRestart = Basic::C_minMaxAvg;
}

PetscErrorCode Basic::converged(SNES snes_local,
                                PetscInt PETScIter,
                                PetscReal devNull1,
                                PetscReal devNull2,
                                PetscReal devNull3,
                                SNESConvergedReason *reason,
                                void *ctx)
{
    return 0;
    // double divTol = std::max(std::max(std::abs(P_av)) * num.eps_div_rtol, num.eps_div_atol);
    // double BCTol = std::max(std::max(std::abs(P_av)) * num.eps_stress_rtol, num.eps_stress_atol);

    // if ((totalIter >= num.itmin && (err_div / divTol < 1.0 && err_BC / BCTol < 1.0)) || terminallyIll) {
    //     reason = 1;
    // } else if (totalIter >= num.itmax) {
    //     reason = -1;
    // } else {
    //     reason = 0;
    // }

    // std::cout << std::endl << " ... reporting ............................................................." << std::endl;
    // std::cout << std::endl
    //           << " error divergence = " << std::fixed << std::setprecision(2) << err_div / divTol
    //           << " (" << std::scientific << std::setprecision(2) << err_div << " / m, tol = " << std::scientific << std::setprecision(2) << divTol << ")"
    //           << std::endl;
    // std::cout << " error stress BC  = " << std::fixed << std::setprecision(2) << err_BC / BCTol
    //           << " (" << std::scientific << std::setprecision(2) << err_BC << " Pa,  tol = " << std::scientific << std::setprecision(2) << BCTol << ")"
    //           << std::endl;
    // std::cout << std::endl << "===========================================================================" << std::endl;
    // std::cout.flush();
    // err_PETSc = 0;
}
PetscErrorCode Basic::formResidual(DMDALocalInfo *residual_subdomain,
                                   void *F_void,
                                   void *r_void,
                                   void *dummy)
{
    Vec *F = static_cast<Vec *>(F_void);
    Vec *r = static_cast<Vec *>(r_void);
    return 0;
    // std::vector<std::vector<double>> deltaF_aim(3, std::vector<double>(3));
    // PetscInt PETScIter, nfuncs;
    // int err_MPI;

    // SNESGetNumberFunctionEvals(SNES_mechanical, &nfuncs, &err_PETSc);
    // CHKERRQ(err_PETSc);
    // SNESGetIterationNumber(SNES_mechanical, &PETScIter, &err_PETSc);
    // CHKERRQ(err_PETSc);

    // if (nfuncs == 0 && PETScIter == 0) {
    //     totalIter = -1; // new increment
    // }

    // if (totalIter <= PETScIter) {
    //     totalIter += 1;
    //     // Print statements and other code from the original function
    //     // ...
    // }

    // // Call your other functions with appropriate parameters
    // utilities_constitutiveResponse(/*...*/);
    // utilities_divergenceRMS(/*...*/);
    // // ...

    // err_MPI = MPI_Allreduce(MPI_IN_PLACE, terminallyIll, 1, MPI_INTEGER_KIND, MPI_LOGICAL, MPI_LOR, MPI_COMM_WORLD);
    // if (err_MPI != MPI_SUCCESS) {
    //     throw std::runtime_error("MPI error");
    // }

    // math_mul3333xx33(/*...*/);

    // // Update F_aim, err_BC, and r
    // // ...
}