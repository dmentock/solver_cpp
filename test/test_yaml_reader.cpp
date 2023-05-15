#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <fstream> 
#include <stdexcept>

#include "yaml_reader.h"
#include "helper.h"



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
  write_to_file(R""""(grid:
  itmin: 2
  itmax: 5
  eps_div_rtol: 1.23
  eps_div_atol: 4
  update_gamma: true
  )"""");

  YamlReader config;

  config.parse_num_grid_yaml(numerics_path);
  ASSERT_EQ(config.num_grid.itmin, 2);
  ASSERT_EQ(config.num_grid.itmax, 5);
  ASSERT_EQ(config.num_grid.eps_div_rtol, 1.23);
  ASSERT_EQ(config.num_grid.eps_div_atol, 4);
  ASSERT_EQ(config.num_grid.update_gamma, true);
}

TEST_F(NumYamlSetup, TestYamlReadInvalidIntValue) {
  write_to_file(R""""(grid:
  itmin: 0
  itmax: 1
  divergence_correction: 3
  eps_div_atol: 0
  )"""");
  YamlReader config;
  try {
    config.parse_num_grid_yaml(numerics_path);
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
  write_to_file(R""""(grid:
  itmin: true
  )"""");
  YamlReader config;
  try {
    config.parse_num_grid_yaml(numerics_path);
  } catch (const std::exception& e) {
    // TODO: create wrapper with more useful error message
    EXPECT_STREQ(e.what(), "yaml-cpp: error at line 2, column 10: bad conversion");
  }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}