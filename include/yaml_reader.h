#ifndef YAML_READER_H
#define YAML_READER_H
#include <unordered_map>
#include <string>

struct NumGridParams {
    int maxStaggeredIter = 10;
    int maxCutBack = 3;

    int itmin = 1;
    int itmax = 250;
    int memory_efficient = 1;
    int divergence_correction = 2;
    bool update_gamma = false;

    double eps_div_atol = 1e-4;
    double eps_div_rtol = 5e-4;
    double eps_stress_atol = 1e+3;
    double eps_stress_rtol = 1e-3;
    double eps_curl_atol = 1e-10;
    double eps_curl_rtol = 5e-4;

    double alpha = 1;
    double beta = 1;
    
    double eps_thermal_atol = 1e-2;
    double eps_thermal_rtol = 1e-6;

    double eps_damage_atol = 1e-2;
    double eps_damage_rtol = 1e-6;
    double phi_min = 1e-6;

    std::string petsc_options = "";
    std::string derivative = "continuous";
    std::string fftw_plan_mode = "FFTW_MEASURE";
};

class YamlReader {
public:
  NumGridParams num_grid;
  void parse_num_grid_yaml(std::string yamlFilePath);
};


#endif // YAML_READER_H