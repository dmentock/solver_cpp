#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <mech_solver_basic.h>

#include <iostream>
#include <cstdio>
#include <fstream>
#include <optional>

#include <unsupported/Eigen/CXX11/Tensor>
#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

#include "simple_grid_setup.hpp"
#include "init_environments.hpp"
#include <tensor_operations.h>

using Tensor2d = Eigen::Tensor<double, 2>;
using Tensor4d = Eigen::Tensor<double, 4>;
using Tensor5d = Eigen::Tensor<double, 5>;
using TensorMap5d = Eigen::TensorMap<Eigen::Tensor<double, 5>>;
using optional_q = std::optional<Eigen::Quaterniond>;
class PartialMockMechSolverBasic : public MechSolverBasic {
public:
  PartialMockMechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : MechSolverBasic(config_, grid_, spectral_) {};
  MOCK_METHOD(void, update_coords, (Tensor5d&, Tensor2d&, Tensor2d&), (override));
  MOCK_METHOD(void, update_gamma, (Tensor4d&), (override));
};
  class PartialMockSpectral : public Spectral {
    public:
    PartialMockSpectral(Config& config_, DiscretizationGrid& grid_)
    : Spectral(config_, grid_) {}
    MOCK_METHOD(void, constitutive_response, (TensorMap5d&, Tensor2d&, Tensor4d&, Tensor4d&, TensorMap5d&, double, optional_q), (override));
  };
TEST_F(GridTestSetup, TestMechSolverBasicInit) {
  gridSetup_init_grid(std::array<int, 3>{2,1,1});
  PartialMockSpectral spectral(config, *mock_grid);
  PartialMockMechSolverBasic mech_basic(config, *mock_grid, spectral);

  Eigen::Tensor<double, 5> expected_F_lastInc(3, 3, 2, 1, 1);
  expected_F_lastInc.setValues({
   {{{{  1 }}, {{  1 }}},
    {{{  0 }}, {{  0 }}},
    {{{  0 }}, {{  0 }}}},
   {{{{  0 }}, {{  0 }}},
    {{{  1 }}, {{  1 }}},
    {{{  0 }}, {{  0 }}}},
   {{{{  0 }}, {{  0 }}},
    {{{  0 }}, {{  0 }}},
    {{{  1 }}, {{  1 }}}}
  });

  Eigen::Tensor<double, 3> expected_homogenization_F0(3, 3, 2);
  expected_homogenization_F0.setValues({
   {{  1,  1 }, {  0,  0 }, {  0,  0 }},
   {{  0,  0 }, {  1,  1 }, {  0,  0 }},
   {{  0,  0 }, {  0,  0 }, {  1,  1 }}
  });

  EXPECT_CALL(mech_basic, update_coords(testing::_, testing::_, testing::_)).WillOnce(testing::DoDefault());
  EXPECT_CALL(mech_basic, update_gamma(testing::_)).WillOnce(testing::DoDefault());
  EXPECT_CALL(spectral, constitutive_response(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_)).WillOnce(testing::DoDefault());

  mech_basic.init();

  EXPECT_TRUE(tensor_eq(mech_basic.F_lastInc, expected_F_lastInc));
  EXPECT_TRUE(tensor_eq(mech_basic.homogenization_F0, expected_homogenization_F0));
}

ACTION_P(SetArg0, value) { arg0 = value; }
TEST_F(GridTestSetup, TestFormResidual) {
  gridSetup_init_grid(std::array<int, 3>{2,1,1});
  PartialMockSpectral spectral(config, *mock_grid);
  PartialMockMechSolverBasic mech_basic(config, *mock_grid, spectral);

  DMDALocalInfo residual_subdomain;

  Eigen::Tensor<double, 5> F(3, 3, 2, 1, 1);
  F.setValues({
   {{{{  1.1002000000000001 }}, {{  1.1002000000000001 }}},
    {{{  0                  }}, {{  0                  }}},
    {{{  0                  }}, {{  0                  }}}},
   {{{{  0                  }}, {{  0                  }}},
    {{{  1                  }}, {{  1                  }}},
    {{{  0                  }}, {{  0                  }}}},
   {{{{  0                  }}, {{  0                  }}},
    {{{  0                  }}, {{  0                  }}},
    {{{  1                  }}, {{  1                  }}}}
  });
  Eigen::Tensor<double, 5> r(3, 3, 2, 1, 1);
  r.setZero();

  Eigen::Tensor<double, 5> mocked_constitutive_response_P(3, 3, 2, 1, 1);
  mocked_constitutive_response_P.setValues({
   {{{{  11823499031.8272   }},
     {{  11822773973.893665 }}},
    {{{  6593774.9170823097 }},
     {{  7412068.1404895782 }}},
    {{{ -15122969.537146807 }},
     {{  9211038.9774441719 }}}},
   {{{{  5993251.1516828537 }},
     {{  6737018.8515625    }}},
    {{{  11709066032.578241 }},
     {{  11725937266.480543 }}},
    {{{ -1647942.0415525436 }},
     {{  7174876.1495118141 }}}},
   {{{{ -13745654.91469276  }},
     {{  8372149.5886597633 }}},
    {{{ -1647942.0415520668 }},
     {{  7174876.1495125294 }}},
    {{{  11722169413.974548 }},
     {{  11705689816.819708 }}}}
  });
  EXPECT_CALL(spectral, constitutive_response
    (testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
    .WillOnce(SetArg0(mocked_constitutive_response_P));

  void* mech_basic_raw = static_cast<void*>(&mech_basic);

  mech_basic.params.rot_bc_q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);

  SNESCreate(PETSC_COMM_WORLD, &(mech_basic.SNES_mechanical));
  
  mech_basic.formResidual(&residual_subdomain, &F, &r, mech_basic_raw);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}