#ifndef YAML_READER_H
#define YAML_READER_H

#include <yaml-cpp/yaml.h>
#include <yaml.h>

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Geometry> 
#include <Eigen/Dense>
#include <fftw3.h>

#include <unordered_map>
#include <string>
#include <map> 
#include <iostream>

using namespace std;
using namespace Eigen;

class Config {
public:

  enum divergence_correction_ids { 
    DIVERGENCE_CORRECTION_NONE_ID,
    DIVERGENCE_CORRECTION_SIZE_ID,
    DIVERGENCE_CORRECTION_SIZE_GRID_ID
  };

  enum derivative_ids { 
    DERIVATIVE_CONTINUOUS_ID,
    DERIVATIVE_CENTRAL_DIFF_ID,
    DERIVATIVE_FWBW_DIFF_ID
  };

  struct SolutionState {
    int required_iterations = 0;
    bool converged = true;
    bool stag_converged = true;
    bool terminally_ill = false;
  };

  struct SolutionParams {
    Eigen::Matrix<double, 3, 3> stress_bc;
    Eigen::Matrix<bool, 3, 3> stress_mask;
    Eigen::Quaterniond rot_bc_q;
    double delta_t;
  };


  struct NumGridMechanical {
    int itmin = 1;
    int itmax = 100;
    bool update_gamma = false;
    double eps_abs_div = 1e-4;
    double eps_rel_div = 5e-4;
    double eps_abs_P = 1e+3;
    double eps_rel_P =  1e-3;
  };

  struct NumFFT{
    int memory_efficient = 1;
    int divergence_correction = DIVERGENCE_CORRECTION_SIZE_GRID_ID;
    int derivative = DERIVATIVE_CONTINUOUS_ID;
    int fftw_planner_flag = FFTW_MEASURE;
    double fftw_timelimit = 300;
    std::string petsc_options = "";
    double alpha = 1;
    double beta = 1;
    double eps_abs_curl_F = 1e-10;
    double eps_rel_curl_F = 5e-4;
  };

  struct Numerics {
    int stag_iter_max = 10;
    int max_cut_back = 3;
    NumGridMechanical grid_mechanical;
    NumFFT fft;
  };

  struct BoundaryCondition {
    Matrix<double, 3, 3> values = Eigen::Matrix<double, 3, 3>::Zero();
    Matrix<bool, 3, 3> mask = Eigen::Matrix<bool, 3, 3>::Constant(false);
    std::string type = "";
  };

  struct LoadStep {
    BoundaryCondition stress;
    BoundaryCondition deformation;
    Quaterniond rot_bc_q = Quaterniond(1, 0, 0, 0);
    int t;
    double N;
    int r = 1;
    bool estimate_rate = true;
    int f_out = 1;
    int f_restart = 0;
  };

  struct Load {
    std::map<std::string, std::string> fields;
    std::vector<LoadStep> steps;
  };

  Numerics numerics;
  Load load;

  std::string vti_file_content;

  int n_total_load_steps = 0;

  static BoundaryCondition parse_boundary_condition(YAML::Node& mechanicalNode, std::vector<std::string>& key_variations);
  Load parse_load(const std::string& yaml_path);
  Numerics parse_numerics(const std::string& yaml_path);
};
#endif // YAML_READER_H
