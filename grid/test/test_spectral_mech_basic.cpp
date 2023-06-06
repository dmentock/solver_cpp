#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "spectral/mech/basic.h"

#include <iostream>
#include <cstdio>
#include <fstream>

#include <unsupported/Eigen/CXX11/Tensor>
#include <optional>

#include "simple_grid_setup.hpp"
#include "init_environments.hpp"
#include "tensor_operations.h"

using Tensor2 = Eigen::Tensor<double, 2>;
using Tensor4 = Eigen::Tensor<double, 4>;
using Tensor5 = Eigen::Tensor<double, 5>;
using optional_q = std::optional<Eigen::Quaterniond>;
class PartialMockMechBasic : public MechBasic {
public:
  PartialMockMechBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : MechBasic(config_, grid_, spectral_) {};
  MOCK_METHOD(void, update_coords, (Tensor5&, Tensor2&, Tensor2&), (override));
  MOCK_METHOD(void, update_gamma, (Tensor4&), (override));
};
  class PartialMockSpectral : public Spectral {
    public:
    PartialMockSpectral(Config& config_, DiscretizationGrid& grid_)
    : Spectral(config_, grid_) {}
    MOCK_METHOD(void, constitutive_response, (Tensor5&, Tensor2&, Tensor4&, Tensor4&, Tensor5&, double, optional_q), (override));
  };
TEST_F(SimpleGridSetup, TestMechBasicInit) {
  gridSetup_init_grid(std::array<int, 3>{2,1,1});
  PartialMockSpectral spectral(config, *mock_grid);
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
  EXPECT_CALL(spectral, constitutive_response(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_)).WillOnce(testing::DoDefault());

  mech_basic.init();

  EXPECT_TRUE(tensor_eq(mech_basic.F_lastInc, expected_F_lastInc));
  EXPECT_TRUE(tensor_eq(mech_basic.homogenization_F0, expected_homogenization_F0));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}