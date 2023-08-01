#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <fstream> 
#include <stdexcept>

#include "config.h"
#include "helper.h"

#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
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
grid:
  itmin: 2
  itmax: 5
  eps_div_rtol: 1.23
  eps_div_atol: 4
  update_gamma: true
  )"""");

  Config config;

  Config::parse_numerics_yaml(numerics_path, config.numerics);
  ASSERT_EQ(config.numerics.itmin, 2);
  ASSERT_EQ(config.numerics.itmax, 5);
  ASSERT_EQ(config.numerics.eps_div_rtol, 1.23);
  ASSERT_EQ(config.numerics.eps_div_atol, 4);
  ASSERT_EQ(config.numerics.update_gamma, true);
}

TEST_F(NumYamlSetup, TestYamlReadInvalidIntValue) {
  write_to_file(R""""(
grid:
  itmin: 0
  itmax: 1
  divergence_correction: 3
  eps_div_atol: 0
  )"""");
  Config config;
  try {
    Config::parse_numerics_yaml(numerics_path, config.numerics);
  } catch (const std::runtime_error& e) {
      EXPECT_STREQ(e.what(), R"""(errors when parsing numerics yaml: 
itmin must be >= 1
itmax must be > 1
divergence_correction must be => 0 and <= 2
eps_div_atol must be > 0
)""");
  }
}

TEST_F(NumYamlSetup, TestYamlReadIntBoolMismatch) {
  write_to_file(R""""(
grid:
  itmin: true
  )"""");
  Config config;
  try {
    Config::parse_numerics_yaml(numerics_path, config.numerics);
  } catch (const std::exception& e) {
    // TODO: create wrapper with more useful error message
    EXPECT_STREQ(e.what(), "yaml-cpp: error at line 3, column 10: bad conversion");
  }
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
        dot_F: [ [1, 2.3, 4.5], [1.e-3, x, 0], [x, 0, x] ]
        P: [ [0, 1.2, 3.4], [1.e-3, 0, x], [0, x, 0] ]
        R: [ 1, 2, 3, 4]
    discretization:
      t: 60
      N: 120
      r: 2
    estimate_rate: false
    f_out: 3
    f_restart: 10

  )"""");

  Config config;
  std::map<std::string, std::string> fields;
  std::vector<Config::LoadStep> load_steps;

  Config::parse_load_yaml(loadfile_path, fields, load_steps);

  std::map<std::string, std::string> expected_fields;
  expected_fields["mechanical"] = "spectral_basic";
  EXPECT_EQ(fields, expected_fields);

  EXPECT_EQ(load_steps.size(), 1);
  EXPECT_EQ(load_steps[0].r, 2);
  EXPECT_EQ(load_steps[0].t, 60);
  EXPECT_EQ(load_steps[0].N, 120);
  EXPECT_EQ(load_steps[0].estimate_rate, false);
  EXPECT_EQ(load_steps[0].rot_bc_q, Quaterniond(1, 2, 3, 4));
  EXPECT_EQ(load_steps[0].f_out, 3);
  EXPECT_EQ(load_steps[0].f_restart, 10);

  Matrix<double, 3, 3> expected_deformation_values;
  expected_deformation_values <<  1,     2.3,  4.5,
                                  0.001,   0,    0,
                                  0,       0,    0;
  EXPECT_EQ(load_steps[0].deformation.values, expected_deformation_values);

  Matrix<double, 3, 3> expected_stress_values;
  expected_stress_values << 0,     1.2,  3.4,
                            0.001,   0,    0,
                            0,       0,    0;
  EXPECT_EQ(load_steps[0].stress.values, expected_stress_values);
}

TEST_F(LoadYamlSetup, TestYamlReadMultistep) {
  write_to_file(R""""(
---

solver:
  mechanical: spectral_basic
  thermal: spectral

loadstep:
  - boundary_conditions:
      mechanical:
        dot_F: [ [1, 2.3, 4.5], [1.e-3, x, 0], [x, 0, x] ]
        P: [ [0, 1.2, 3.4], [1.e-3, 0, x], [0, x, 0] ]
    discretization:
      t: 60
      N: 120
      r: 2
    f_out: 20
    estimate_rate: true
  - boundary_conditions:
      mechanical:
        dot_F: [[1.0e-3, 0, 0],
                [1.2,    x, 3.4],
                [0,      0, x]]
        P: [[x, x, x],
            [x, 0, x],
            [x, x, 0]]
    discretization:
      t: 600
      N: 6
    f_out: 4
    estimate_rate: true

  )"""");

  Config config;
  std::map<std::string, std::string> fields;
  std::vector<Config::LoadStep> load_steps;

  Config::parse_load_yaml(loadfile_path, fields, load_steps);

  std::map<std::string, std::string> expected_fields;
  expected_fields["mechanical"] = "spectral_basic";
  expected_fields["thermal"] = "spectral";
  EXPECT_EQ(fields, expected_fields);

  EXPECT_EQ(load_steps.size(), 2);
  EXPECT_EQ(load_steps[0].t, 60);
  EXPECT_EQ(load_steps[0].estimate_rate, false);
  Matrix<double, 3, 3> expected_deformation_values_1;
  expected_deformation_values_1 <<  1,     2.3,  4.5,
                                    0.001,   0,    0,
                                    0,       0,    0;
  EXPECT_EQ(load_steps[0].deformation.values, expected_deformation_values_1);

  EXPECT_EQ(load_steps[1].t, 600);
  EXPECT_EQ(load_steps[1].estimate_rate, true);
  Matrix<double, 3, 3> expected_deformation_values_2;
  expected_deformation_values_2 <<  0.001,   0,    0,
                                    1.2,     0,  3.4,
                                    0,       0,    0;
  EXPECT_EQ(load_steps[1].deformation.values, expected_deformation_values_2);
}

TEST_F(LoadYamlSetup, TestYamlMissingDeformation) {
  write_to_file(R""""(
---

solver:
  mechanical: spectral_basic

loadstep:
  - boundary_conditions:
      mechanical:
        P: [ [0, 1.2, 3.4], [1.e-3, 0, x], [0, x, 0] ]
    discretization:
      t: 60
      N: 120
  )"""");

  Config config;
  std::map<std::string, std::string> fields;
  std::vector<Config::LoadStep> load_steps;

  try {
    Config::parse_load_yaml(loadfile_path, fields, load_steps);
  } catch (const std::exception& e) {
    // TODO: create wrapper with more useful error message
    EXPECT_STREQ(e.what(), "Mandatory key {dot_F/L/F} missing");
  }
}


TEST(TestParseMechanical, TestBasic) {
  std::string yamlContent = R""""(
---

mechanical:
  dot_F: [ [1, 2.3, 4.5], [1.e-3, x, 0], [x, 0, x] ]
  P: [ [0, 1.2, 3.4], [1.e-3, 0, x], [0, x, 0] ]
  )"""";
  YAML::Node rootNode = YAML::Load(yamlContent);
  YAML::Node mechNode = rootNode["mechanical"];

  std::vector<std::string> deformation_key_variations = {"dot_F", "L", "F"};
  Config::BoundaryCondition expected_deformation;
  expected_deformation.type = "dot_F";
  expected_deformation.mask <<  false, false, false,
                                false, true , false,
                                true, false, true;
  expected_deformation.values <<  1,     2.3,  4.5,
                                  0.001,   0,    0,
                                  0,       0,    0;
  Config::BoundaryCondition deformation = Config::parse_boundary_condition(mechNode, deformation_key_variations);
  EXPECT_EQ(deformation.type, expected_deformation.type);
  EXPECT_TRUE(deformation.values.isApprox(expected_deformation.values));
  EXPECT_TRUE(deformation.mask.isApprox(expected_deformation.mask));
}

TEST(TestParseMechanical, TestIllegalCharacter) {
  std::string yamlContent = R""""(
mechanical:
  dot_F: [ [1, 2.3, i], [1.e-3, x, 0], [x, 0, x] ]
  )"""";
  YAML::Node rootNode = YAML::Load(yamlContent);
  YAML::Node mechNode = rootNode["mechanical"];

  std::vector<std::string> deformation_key_variations = {"dot_F", "L", "F"};
  try {
    Config::BoundaryCondition deformation = Config::parse_boundary_condition(mechNode, deformation_key_variations);
  } catch (const std::exception& e) {
    // TODO: create wrapper with more useful error message
    EXPECT_STREQ(e.what(), "yaml-cpp: error at line 3, column 21: bad conversion");
  }

}

TEST(TestParseMechanical, TestDuplicateDefinition) {
  std::string yamlContent = R""""(
mechanical:
  dot_F: [ [1, 2.3, 4.5], [1.e-3, x, 0], [x, 0, x] ]
  F: [ [0, 1.2, 3.4], [1.e-3, 0, x], [0, x, 0] ]
  )"""";
  YAML::Node rootNode = YAML::Load(yamlContent);
  YAML::Node mechNode = rootNode["mechanical"];

  std::vector<std::string> deformation_key_variations = {"dot_F", "L", "F"};
  try {
    Config::BoundaryCondition deformation = Config::parse_boundary_condition(mechNode, deformation_key_variations);
  } catch (const std::exception& e) {
    // TODO: create wrapper with more useful error message
    EXPECT_STREQ(e.what(), 
    "Redundant definition in loadstep boundary condition definition, only one of the following set of keys can be defined: {dot_F, L, F}");
  }

}
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
