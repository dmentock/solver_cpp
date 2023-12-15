#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

#include "conftest.h"
#include "init_environments.h"
#include "../utilities_tensor.h"

#include "../damask_grid.h"
#include "../mech_solver_basic.h"

class PartialMockSpectral : public Spectral {
public:
  using Tensor2d = Eigen::Tensor<double, 2>;
  using Tensor4d = Eigen::Tensor<double, 4>;
  using Tensor5d = Eigen::Tensor<double, 5>;
  using TensorMap5d = Eigen::TensorMap<Eigen::Tensor<double, 5>>;

  MOCK_METHOD(Tensor5d, constitutive_response, (Tensor2d&, Tensor4d&, Tensor4d&, TensorMap5d&, 
                                                double, int, std::optional<Eigen::Quaterniond>), (override));
};

TEST_F(GridTestSetup, TestDamaskGridInit) {
  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});
  
  Config config;
  config.load.fields["mechanical"] = "spectral_basic";

  gridTestSetup_init_discretization(mock_grid);

  PartialMockSpectral spectral;
  spectral.init(0, mock_grid, config.numerics.fft.fftw_planner_flag, config.numerics.fft.fftw_timelimit);
  gridTestSetup_mock_homogenization_tensors(spectral, mock_grid.n_cells_global);

  DamaskGrid damask_grid;
  EXPECT_CALL(spectral, constitutive_response
    (testing::_, testing::_, testing::_, testing::_, testing::_, testing::_, testing::_));
  damask_grid.init(config, mock_grid, spectral);
  // TODO: assert..
}


TEST_F(GridTestSetup, TestDamaskGridSolve) {
  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});
  Config config;
  Spectral spectral;

  class MockMechSolverBasic : public MechSolverBasic {
  public:
    MockMechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
        : MechSolverBasic(config_, grid_, spectral_) {};
    Config::SolutionState calculate_solution(std::string& inc_info) {
      Config::SolutionState converged_mech_state;
      return converged_mech_state;
    }
  };

  MockMechSolverBasic mech_basic(config, mock_grid, spectral);

  DamaskGrid damask_grid;
  damask_grid.fields.mech = std::make_unique<MockMechSolverBasic>(std::move(mech_basic));

  std::string inc_info;
  double delta_t;
  int stag_it_max = 10;
  DamaskGrid::SolutionStates solution_states;
  bool converged = damask_grid.solve_fields(solution_states, delta_t, inc_info, stag_it_max);
  EXPECT_EQ(converged, true);
  // TODO: verify with different field configurations
}

TEST_F(GridTestSetup, TestDamaskGridLoopOverLoadcase) {
  
  class MockMechSolverBasic : public MechSolverBasic {
  public:
    MockMechSolverBasic(Config& config_, DiscretizationGrid& grid_, Spectral& spectral_)
        : MechSolverBasic(config_, grid_, spectral_) {};
    void update_coords() {};
  };

class PartialMockDamaskGrid : public DamaskGrid {
public:  // make the mock methods public
    MOCK_METHOD(bool, solve_fields, (SolutionStates&, double&, std::string&, int&), (override));
    MOCK_METHOD(void, forward_fields, (Config::LoadStep&, bool&, bool&, double&, double&, double&), (override));
    MOCK_METHOD(void, materialpoint_result, (int&, double&), (override));
};

  MockDiscretizedGrid mock_grid(std::array<int, 3>{2,1,1});
  Spectral spectral;
  Config config;
  config.load.fields["mechanical"] = "spectral_basic";
  Config::LoadStep load_step;

  Config::BoundaryCondition deformation_bc;
  deformation_bc.type = "dot_F";
  deformation_bc.values << 1e-3, 0,    0,
                           0,    0,    0,
                           0,    0,    0;
  deformation_bc.mask <<  false, false, false,
                          false, true , false,
                          false, false, true;
  load_step.deformation = deformation_bc;
  load_step.rot_bc_q = Quaterniond(1.0, 0.0 ,0.0 ,0.0);
  load_step.t = 1;
  load_step.N = 1;

  config.load.steps.push_back(load_step);

  gridTestSetup_init_discretization(mock_grid);

  MockMechSolverBasic mech_basic(config, mock_grid, spectral);
  PartialMockDamaskGrid damask_grid;
  damask_grid.fields.mech = std::make_unique<MockMechSolverBasic>(std::move(mech_basic));

  int stag_it_max = 1;
  int max_cut_back = 0;

  EXPECT_CALL(damask_grid, solve_fields(testing::_, testing::_, testing::_, testing::_))
    .WillOnce(testing::Return(true));
  EXPECT_CALL(damask_grid, forward_fields(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
    .WillOnce(testing::DoDefault());
  EXPECT_CALL(damask_grid, materialpoint_result(testing::Eq(1), testing::Eq(1))).WillOnce(testing::DoDefault());
  damask_grid.loop_over_loadcase(config.load.steps, stag_it_max, max_cut_back);

  EXPECT_CALL(damask_grid, solve_fields(testing::_, testing::_, testing::_, testing::_))
    .WillOnce(testing::Return(false));
  EXPECT_CALL(damask_grid, forward_fields(testing::_, testing::_, testing::_, testing::_, testing::_, testing::_))
    .WillOnce(testing::DoDefault());
  EXPECT_THROW(damask_grid.loop_over_loadcase(config.load.steps, stag_it_max, max_cut_back), std::runtime_error);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new PetscMpiEnv);
    return RUN_ALL_TESTS();
}
