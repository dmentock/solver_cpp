#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <cstdio>
#include <fstream>

// #include <mpi.h>
#include <petscsys.h>
#include <petsc.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include "tensor_operations.h"

#include "simple_grid_setup.hpp"

#include "spectral/mech/basic.h"
#include "init_environments.hpp"

class PartialMockMechBasic : public MechBasic {
public:
  PartialMockMechBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : MechBasic(config_, grid_, spectral_) {};

  using Tensor2 = Eigen::Tensor<double, 2>;
  using Tensor4 = Eigen::Tensor<double, 4>;
  using Tensor5 = Eigen::Tensor<double, 5>;
  MOCK_METHOD(void, update_coords, (Tensor5&, Tensor2&, Tensor2&), (override));
  MOCK_METHOD(void, update_gamma, (Tensor4&), (override));
};

TEST_F(SimpleGridSetup, TestMechBasicInit) {
  init_grid(std::array<int, 3>{2,1,1});
  Spectral spectral(config, *mock_grid);
  PartialMockMechBasic mech_basic(config, *mock_grid, spectral);

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

  mech_basic.init();
  EXPECT_TRUE(tensor_eq(mech_basic.F_lastInc, expected_F_lastInc));
  EXPECT_TRUE(tensor_eq(mech_basic.homogenization_F0, expected_homogenization_F0));
}

// TEST_F(SimpleGridSetup, TestSpectralMechBasicInitFull) {
//   init_grid(std::array<int, 3>{2,1,1});
//   Basic spectral_basic(*mock_grid);
//   spectral_basic.init();
//   spectral_basic.init_basic();
// }

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}