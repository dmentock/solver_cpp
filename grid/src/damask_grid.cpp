#include <damask_grid.h>
#include <config.h>
#include <spectral.h>
#include <cstdio>
#include <mech_solver_basic.h>

void DamaskGrid::init(Config& config, DiscretizationGrid& grid, Spectral& spectral) {

  // write to resultfile

  if (config.fields.find("mechanical") != config.fields.end()) {
    if (config.fields["mechanical"] == "spectral_basic") {
      mech_field_ptr = std::make_unique<MechSolverBasic>(config, grid, spectral);      
    } else {
      throw std::runtime_error("unkown key for mechanical field specified in loadfile: " + config.fields["mechanical"]);
    }
  } else {
    throw std::runtime_error("mandatory mechanical field key in loadfile is not specified");
  }

  mech_field_ptr->init();
}

