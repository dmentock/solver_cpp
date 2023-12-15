#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fstream> 
#include <stdexcept>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <yaml.h>

#include "../utilities_tensor.h"

#include "../config.h"

using namespace Eigen;

class NumYamlSetup : public ::testing::Test {
 protected:
  std::string numerics_path;
  void SetUp() override {
    numerics_path = "numerics.yaml";
  }

  void TearDown() override {
    std::remove(numerics_path.c_str());
  }

  void write_to_file(const std::string& content) {
    std::ofstream tmpFile(numerics_path);
    tmpFile << content;
    tmpFile.close();
  }
};

TEST_F(NumYamlSetup, TestYamlReadSuccess) {
  write_to_file(R""""(
solver:
  grid:
    N_staggered_iter_max: 10            # max number of field-level staggered iterations
    N_cutback_max:        3             # maximum cutback level (0: 1, 1: 0.5, 2: 0.25, etc)

    mechanical:
      N_iter_min: 1                     # minimum iteration number
      N_iter_max: 100                   # maximum iteration number
      eps_abs_div(P): 1.0e-4            # absolute tolerance for fulfillment of stress equilibrium
      eps_rel_div(P): 5.0e-4            # relative tolerance for fulfillment of stress equilibrium
      eps_abs_P: 1.0e3                  # absolute tolerance for fulfillment of stress BC
      eps_rel_P: 1.0e-3                 # relative tolerance for fulfillment of stress BC
      update_gamma: false               # update Gamma-operator with current dPdF (not possible if FFT: memory_efficient == true)

    FFT:
      memory_efficient: true            # precalculate Gamma-operator (81 doubles per point)
      divergence_correction: size+grid  # use size-independent divergence criterion {none, size, size+grid}
      derivative: continuous            # approximation used for derivatives in Fourier space {continuous, central_difference, FWBW_difference}
      FFTW_plan_mode: FFTW_MEASURE      # planning-rigor flags, see manual at https://www.fftw.org/fftw3_doc/Planner-Flags.html
      FFTW_timelimit: -1.0              # time limit of plan creation for FFTW, see manual on www.fftw.org. (-1.0: disable time limit)
      PETSc_options: -snes_type ngmres -snes_ngmres_anderson   # PETSc solver options
      alpha: 1.0                        # polarization scheme parameter 0.0 < alpha < 2.0 (1.0: AL scheme, 2.0: accelerated scheme)
      beta: 1.0                         # polarization scheme parameter 0.0 <  beta < 2.0 (1.0: AL scheme, 2.0: accelerated scheme)
      eps_abs_curl(F): 1.0e-10          # absolute tolerance for fulfillment of strain compatibility
      eps_rel_curl(F): 5.0e-4           # relative tolerance for fulfillment of strain compatibility
)"""");

  Config config;

  config.numerics = config.parse_numerics(numerics_path);
  ASSERT_EQ(config.numerics.stag_iter_max, 10);
  ASSERT_EQ(config.numerics.max_cut_back, 3);
  ASSERT_EQ(config.numerics.grid_mechanical.itmin, 1);
  ASSERT_EQ(config.numerics.grid_mechanical.itmax, 100);
  ASSERT_DOUBLE_EQ(config.numerics.grid_mechanical.eps_abs_div, 1.0e-4);
  ASSERT_DOUBLE_EQ(config.numerics.grid_mechanical.eps_rel_div, 5.0e-4);
  ASSERT_DOUBLE_EQ(config.numerics.grid_mechanical.eps_abs_P, 1.0e3);
  ASSERT_DOUBLE_EQ(config.numerics.grid_mechanical.eps_rel_P, 1.0e-3);
  ASSERT_FALSE(config.numerics.grid_mechanical.update_gamma);
}

class LoadYamlSetup : public ::testing::Test {
 protected:
  std::string loadfile_path;
  void SetUp() override {
    loadfile_path = "load.yaml";
  }

  void TearDown() override {
    std::remove(loadfile_path.c_str());
  }

  void write_to_file(const std::string& content) {
    std::ofstream tmpFile(loadfile_path);
    tmpFile << content;
    tmpFile.close();
  }
};

TEST_F(LoadYamlSetup, TestYamlReadBasic) {
  write_to_file(R""""(
---

solver:
  mechanical: spectral_basic

loadstep:
  - boundary_conditions:
      mechanical:
        dot_F: [[1.0e-3, 1, 0],
                [0,      x, 0],
                [2,      0, x]]
        P: [[x, x, x],
            [x, 0, x],
            [x, x, 0]]
        R: [1, 2, 3, 4]
    discretization:
      t: 60
      N: 120
      r: 2
    estimate_rate: false
    f_out: 4
    f_restart: 10
)"""");

  Config config;
  config.load = config.parse_load(loadfile_path);

  std::map<std::string, std::string> expected_fields;
  expected_fields["mechanical"] = "spectral_basic";
  EXPECT_EQ(config.load.fields, expected_fields);

  EXPECT_EQ(config.load.steps[0].r, 2);
  EXPECT_EQ(config.load.steps[0].t, 60);
  EXPECT_EQ(config.load.steps[0].N, 120);
  EXPECT_EQ(config.load.steps[0].estimate_rate, false);
  EXPECT_EQ(config.load.steps[0].rot_bc_q, Quaterniond(1, 2, 3, 4));
  EXPECT_EQ(config.load.steps[0].f_out, 4);
  EXPECT_EQ(config.load.steps[0].f_restart, 10);

  Matrix<double, 3, 3> expected_deformation_values;
  Matrix<bool, 3, 3> expected_deformation_mask;
  expected_deformation_values <<  0.001, 1,    0,
                                  0,     0,    0,
                                  2,     0,    0;
  expected_deformation_mask <<  false, false, false,
                                false, true , false,
                                false, false, true;
  EXPECT_EQ(config.load.steps[0].deformation.values, expected_deformation_values);
  EXPECT_EQ(config.load.steps[0].deformation.mask, expected_deformation_mask);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
