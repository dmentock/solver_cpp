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
#include <petscsys.h>

#include "simple_grid_setup.hpp"
#include "init_environments.hpp"
#include <tensor_operations.h>
#include <helper.h>

using Tensor2d = Eigen::Tensor<double, 2>;
using Tensor4d = Eigen::Tensor<double, 4>;
using Tensor5d = Eigen::Tensor<double, 5>;
using TensorMap5d = Eigen::TensorMap<Eigen::Tensor<double, 5>>;
using Matrix33b = Eigen::Matrix<bool, 3, 3>;
using Matrix33d = Eigen::Matrix<double, 3, 3>;
using optional_q = std::optional<Eigen::Quaterniond>;

MATCHER_P(TensorEq, other, "") {
  return tensor_eq(arg, other);
}
ACTION_P(SetArg0, value) { arg0 = value; }
ACTION_P(SetArg1, value) { arg1 = value; }


class PartialMockSpectral : public Spectral {
  public:
  PartialMockSpectral(Config& config_, DiscretizationGrid& grid_)
  : Spectral() {}
  MOCK_METHOD(void, constitutive_response, (TensorMap5d&, Tensor2d&, Tensor4d&, Tensor4d&, TensorMap5d&, double, optional_q), (override));
};

TEST_F(GridTestSetup, TestMechSolverBasicInit) {
  class PartialMockMechSolverBasic : public MechSolverBasic {
  public:
    PartialMockMechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
        : MechSolverBasic(config_, grid_, spectral_) {};
    MOCK_METHOD(void, base_init, (), (override));
    MOCK_METHOD(void, base_update_coords, (TensorMap5d&, Tensor2d&, Tensor2d&), (override));
    MOCK_METHOD(void, update_gamma, (Tensor4d&), (override));
  };

  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});

  PartialMockSpectral spectral(config, mock_grid);
  PartialMockMechSolverBasic mech_basic(config, mock_grid, spectral);

  gridTestSetup_set_up_dm(mech_basic.da, mech_basic.solution_vec, mock_grid);
  Eigen::Tensor<PetscScalar, 4> F_(9, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2]);
  F_.setValues({
    {{{0}},  {{1}}},  {{{2}},  {{3}}},  {{{4}},  {{5}}},
    {{{6}},  {{7}}},  {{{8}},  {{9}}},  {{{10}}, {{11}}},
    {{{12}}, {{13}}}, {{{14}}, {{15}}}, {{{16}}, {{17}}},
  });
  gridTestSetup_set_solution_vec(F_, mech_basic.solution_vec);

  Eigen::Tensor<double, 5> expected_F_last_inc(3, 3, 2, 1, 1);
  expected_F_last_inc.setValues({
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

  EXPECT_CALL(mech_basic, base_init());
  EXPECT_CALL(mech_basic, base_update_coords(testing::_, testing::_, testing::_)).WillOnce(testing::DoDefault());
  EXPECT_CALL(mech_basic, update_gamma(testing::_)).WillOnce(testing::DoDefault());
  EXPECT_CALL(spectral, constitutive_response(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_)).WillOnce(testing::DoDefault());

  Eigen::Tensor<double, 3> homogenization_F0_(3, 3, 2);
  spectral.homogenization_F0 = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(homogenization_F0_.data(), 3, 3, 2);

  mech_basic.init();

  EXPECT_TRUE(tensor_eq(mech_basic.F_last_inc, expected_F_last_inc));
  EXPECT_TRUE(tensor_eq(*spectral.homogenization_F0, expected_homogenization_F0));
}

TEST_F(GridTestSetup, TestConverged) {
  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});
  PartialMockSpectral spectral(config, mock_grid);
  MechSolverBasic mech_basic(config, mock_grid, spectral);
  SNESConvergedReason reason;
  void* mech_basic_raw = static_cast<void*>(&mech_basic);

  mech_basic.total_iter = 1;
  mech_basic.converged(mech_basic.SNES_mechanical, 4, 0, 0, 0, &reason, mech_basic_raw);

  EXPECT_EQ(reason, 0);
}


TEST_F(GridTestSetup, TestMechSolverBasicSolution) {

  class PartialMockMechSolverBasic : public MechSolverBasic {
  public:
    PartialMockMechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
        : MechSolverBasic(config_, grid_, spectral_) {};
    MOCK_METHOD(Tensor4d, calculate_masked_compliance, (Tensor4d&, Eigen::Quaterniond&, Matrix33b&), (override));
    MOCK_METHOD(void, update_gamma, (Tensor4d&), (override));
    static PetscErrorCode formResidual(DMDALocalInfo* residual_subdomain, void* F_raw, void* r_raw, void *ctx) { return 0; }  
    static PetscErrorCode converged(SNES snes_local, PetscInt PETScIter, 
                                    PetscReal devNull1, PetscReal devNull2, 
                                    PetscReal fnorm, SNESConvergedReason *reason, 
                                    void *ctx) { return 0; }
  };
  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});

  PartialMockSpectral spectral(config, mock_grid);
  PartialMockMechSolverBasic mech_basic(config, mock_grid, spectral);

  gridTestSetup_set_up_dm(mech_basic.da, mech_basic.solution_vec, mock_grid);
  gridTestSetup_set_up_snes(mech_basic.SNES_mechanical);

  Eigen::Tensor<PetscScalar, 4> F_(9, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2]);
  F_.setValues({
    {{{0}},  {{1}}},  {{{2}},  {{3}}},  {{{4}},  {{5}}},
    {{{6}},  {{7}}},  {{{8}},  {{9}}},  {{{10}}, {{11}}},
    {{{12}}, {{13}}}, {{{14}}, {{15}}}, {{{16}}, {{17}}},
  });
  gridTestSetup_set_solution_vec(F_, mech_basic.solution_vec);

  DMDASNESSetFunctionLocal(mech_basic.da, INSERT_VALUES, mech_basic.formResidual, this);
  SNESSetConvergenceTest(mech_basic.SNES_mechanical, mech_basic.converged, PETSC_NULL, NULL);
  SNESSetDM(mech_basic.SNES_mechanical, mech_basic.da);
  SNESSetFromOptions(mech_basic.SNES_mechanical);
  bool terminally_ill;
  spectral.terminally_ill = &terminally_ill;

  Spectral::SolutionState solution = mech_basic.calculate_solution("test");
  // cout << "ss " << solution << endl;
  EXPECT_EQ(*spectral.terminally_ill, false);
}

class PartialMockMechSolverBasicForward : public MechSolverBasic {
public:
  PartialMockMechSolverBasicForward(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
      : MechSolverBasic(config_, grid_, spectral_) {};
  MOCK_METHOD(Tensor5d, forward_field, (double, Tensor5d&, Tensor5d&, Matrix33d*), (override));
};
TEST_F(GridTestSetup, TestBasicForward) {

  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});
  PartialMockSpectral spectral(config, mock_grid);
  PartialMockMechSolverBasicForward mech_basic(config, mock_grid, spectral);

  gridTestSetup_set_up_dm(mech_basic.da, mech_basic.solution_vec, mock_grid);


  Config::BoundaryCondition deformation_bc;
  deformation_bc.type = "dot_F";
  deformation_bc.values << 1e-3, 0,    0,
                           0,    0,    0,
                           0,    0,    0;
  deformation_bc.mask <<  false, false, false,
                          false, true , false,
                          false, false, true;

  Config::BoundaryCondition stress_bc;
  stress_bc.type = "P";
  stress_bc.mask << true, true, true,
                    true, false, true,
                    true, true, false;

  Eigen::Quaterniond rot_bc_q(1.0, 0.0, 0.0, 0.0);

  Eigen::Tensor<double, 5> expected_field_last_inc(3, 3, 2, 1, 1);
  expected_field_last_inc.setValues({
   {{{{  1 }}, {{  1 }}}, {{{  0 }}, {{  0 }}}, {{{  0 }}, {{  0 }}}},
   {{{{  0 }}, {{  0 }}}, {{{  1 }}, {{  1 }}}, {{{  0 }}, {{  0 }}}},
   {{{{  0 }}, {{  0 }}}, {{{  0 }}, {{  0 }}}, {{{  1 }}, {{  1 }}}}
  });

  Eigen::Matrix<double, 3, 3> expected_aim;
  expected_aim << 1.1002000000000001, 0, 0,
                  0                 , 1, 0,
                  0                 , 0, 1;

  Eigen::Tensor<double, 5> forwarded_field(3, 3, 2, 1, 1);
  forwarded_field.setValues({
   {{{{  1.1002000000000001 }}, {{  1.1002000000000001 }}}, {{{  0                  }}, {{  0                  }}}, {{{  0                  }}, {{  0                  }}}},
   {{{{  0                  }}, {{  0                  }}}, {{{  1                  }}, {{  1                  }}}, {{{  0                  }}, {{  0                  }}}},
   {{{{  0                  }}, {{  0                  }}}, {{{  0                  }}, {{  0                  }}}, {{{  1                  }}, {{  1                  }}}}
  });

  Eigen::Tensor<PetscScalar, 4> F_(9, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2]);
  F_.setValues({
   {{{  1 }}, {{  1 }}}, {{{  0 }}, {{  0 }}}, {{{  0 }}, {{  0 }}}, 
   {{{  0 }}, {{  0 }}}, {{{  1 }}, {{  1 }}}, {{{  0 }}, {{  0 }}}, 
   {{{  0 }}, {{  0 }}}, {{{  0 }}, {{  0 }}}, {{{  1 }}, {{  1 }}}
  });
  gridTestSetup_set_solution_vec(F_, mech_basic.solution_vec);

  EXPECT_CALL(mech_basic, forward_field(100.2, 
                                        TensorEq(expected_field_last_inc), 
                                        testing::_, 
                                        testing::_))
                                        .WillOnce(testing::Return(forwarded_field));
  Eigen::Tensor<double, 3> homogenization_F0_(3, 3, 2);
  spectral.homogenization_F0 = std::make_unique<Eigen::TensorMap<Eigen::Tensor<double, 3>>>(homogenization_F0_.data(), 3, 3, 2);
  mech_basic.forward(false, false, 100.2, 1, 1002, deformation_bc, stress_bc, rot_bc_q);

  double* F_res;
  ierr = VecGetArray(mech_basic.solution_vec, &F_res);
  CHKERRABORT(PETSC_COMM_WORLD, ierr);
  TensorMap<Tensor<double, 5>> F(reinterpret_cast<double*>(F_res),  3, 3, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells2);
  EXPECT_TRUE(tensor_eq(F, forwarded_field));
  EXPECT_EQ(mech_basic.params.delta_t, 100.2);
  EXPECT_EQ(mech_basic.params.rot_bc_q, rot_bc_q);
}

TEST_F(GridTestSetup, TestFormResidual) {

  class PartialMockMechSolverBasic : public MechSolverBasic {
  public:
    PartialMockMechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
        : MechSolverBasic(config_, grid_, spectral_) {};
    MOCK_METHOD(double, calculate_divergence_rms, (const Tensor5d&), (override));
    MOCK_METHOD(void, gamma_convolution, (TensorMap5d&, Tensor2d&), (override));
  };

  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});

  PartialMockSpectral spectral(config, mock_grid);
  PartialMockMechSolverBasic mech_basic(config, mock_grid, spectral);

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

  mech_basic.S.setValues({
   {{{  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      }},
    {{  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      }},
    {{  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      }}},
   {{{  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      }},
    {{  0                     ,  0                     ,  0                      },
     {  0                     ,  1.0088275068222855e-11,  0                      },
     {  0                     ,  0                     , -6.176010751524149e-12  }},
    {{  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      }}},
   {{{  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      }},
    {{  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      },
     {  0                     ,  0                     ,  0                      }},
    {{  0                     ,  0                     ,  0                      },
     {  0                     , -6.1760107515241474e-12,  0                      },
     {  0                     ,  0                     ,  1.0143787719914657e-11 }}}
  });

  mech_basic.P_av.setValues({
   {  11823136502.860432,  7002921.528785944 , -2955965.2798513174 },
   {  6365135.0016226768,  11717501649.529392,  2763467.0539796352 },
   { -2686752.6630164981,  2763467.0539802313,  11713929615.397129 }
  });

  mech_basic.F_aim << 1.1002000000000001, 0, 0,
                      0                 , 1, 0,
                      0                 , 0, 1;

  mech_basic.params.stress_mask <<  true, true, true,
                                    true, false, true,
                                    true, true, false;

  mech_basic.params.rot_bc_q = Eigen::Quaterniond(1.0, 0.0, 0.0, 0.0);

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
  Eigen::TensorMap<Eigen::Tensor<double, 5>> mocked_constitutive_response_P_map(
    mocked_constitutive_response_P.data(), 3, 3, 2, 1, 1);
  EXPECT_CALL(spectral, constitutive_response
    (testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
    .WillOnce(SetArg0(mocked_constitutive_response_P_map));

  EXPECT_CALL(mech_basic, calculate_divergence_rms(testing::_)).WillOnce(testing::Return(0));

  Eigen::TensorMap<Tensor<double, 5>> r_map(r.data(), 3, 3, 2, 1, 1);
  Eigen::Tensor<double, 2> expected_rotated(3, 3);
  expected_rotated.setValues({
   {  0                  ,  0                   ,  0                   },
   {  0                  ,  0.045864024505517742,  0                   },
   {  0                  ,  0                   ,  0.04645619921611456 }
  });
  EXPECT_CALL(mech_basic, gamma_convolution(TensorEq(r_map), TensorEq(expected_rotated)));

  Eigen::Tensor<double, 2> expected_F_aim(3, 3);
  expected_F_aim.setValues({
   {  1.1002000000000001,  0                  ,  0                   },
   {  0                 ,  0.95413597549448226,  0                   },
   {  0                 ,  0                  ,  0.95354380078388545 }
  });

  SNESCreate(PETSC_COMM_WORLD, &(mech_basic.SNES_mechanical));
  void* mech_basic_raw = static_cast<void*>(&mech_basic);

  mech_basic.formResidual(&residual_subdomain, F.data(), r.data(), mech_basic_raw);

  EXPECT_EQ(mech_basic.err_div, 0);
  EXPECT_EQ(mech_basic.err_BC, 11717501649.529392);
  TensorMap<Tensor<double, 2>> F_aim_map(mech_basic.F_aim.data(), 3, 3);
  EXPECT_TRUE(tensor_eq(F_aim_map, expected_F_aim));
}

TEST_F(GridTestSetup, TestMechSolverBasicUpdateCoords) {
  class PartialMockMechSolverBasic : public MechSolverBasic {
  public:
    PartialMockMechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
        : MechSolverBasic(config_, grid_, spectral_) {};
    MOCK_METHOD(void, base_update_coords, (TensorMap5d&, Tensor2d&, Tensor2d&), (override));
  };

  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});
  PartialMockSpectral spectral(config, mock_grid);
  PartialMockMechSolverBasic mech_basic(config, mock_grid, spectral);

  gridTestSetup_set_up_dm(mech_basic.da, mech_basic.solution_vec, mock_grid);

  Eigen::Tensor<PetscScalar, 4> F_(9, mock_grid.cells[0], mock_grid.cells[1], mock_grid.cells[2]);
  F_.setValues({
   {{{  1 }}, {{  2 }}}, {{{  3 }}, {{  4 }}}, {{{  5 }}, {{  6 }}}, 
   {{{  7 }}, {{  8 }}}, {{{  9 }}, {{ 10 }}}, {{{ 11 }}, {{ 12 }}}, 
   {{{ 13 }}, {{ 14 }}}, {{{ 15 }}, {{ 16 }}}, {{{ 17 }}, {{ 18 }}}
  });
  gridTestSetup_set_solution_vec(F_, mech_basic.solution_vec);

  EXPECT_CALL(mech_basic, base_update_coords(testing::_, testing::_, testing::_)).WillOnce(testing::DoDefault());
  mech_basic.update_coords();
}


int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}