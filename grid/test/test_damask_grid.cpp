#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <damask_grid.h>
#include <mech_solver_basic.h>

#include <iostream>

#include <unsupported/Eigen/CXX11/Tensor>

#include "simple_grid_setup.hpp"
#include "init_environments.hpp"
#include <helper.h>

TEST_F(GridTestSetup, TestDamaskGridInit) {
  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});
  Config config;
  config.fields["mechanical"] = "spectral_basic";

  gridSetup_init_discretization(mock_grid);
  f_materialpoint_initAll();

  Spectral spectral;
  spectral.init(0, mock_grid);

  DamaskGrid damask_grid;
  damask_grid.init(config, mock_grid, spectral);
}
