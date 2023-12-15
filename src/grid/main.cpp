#include "damask_grid.h"
#include "test/conftest.h"
#include "spectral.h"
#include <iostream>
#include <string>
#include <fstream>

extern "C" {
  void f_materialpoint_initBase(const char* material_path, int material_path_len,
                                const char* load_path, int load_path_len,
                                const char* grid_path, int grid_path_len,
                                const char* numerics_path, int numerics_path_len);
  void f_materialpoint_initDamask();
}

bool fileExists(const std::string& filename) {
  std::ifstream file(filename);
  return file.good();
}

std::string read_file(const std::string& file_path) {
    std::ifstream fstream(file_path);
        if (fstream.fail()) {
        throw std::runtime_error("Error: Unable to open file " + file_path);
    }
    std::stringstream buffer;
    buffer << fstream.rdbuf();
    return buffer.str();
};

int main(int argc, char *argv[]) {
  std::cout << std::fixed << std::setprecision(16);
  Config config;

  std::string material_path, grid_path, load_path, numerics_path;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if ((arg == "-m") && (i + 1 < argc)) {
      material_path = argv[++i];
    } else if ((arg == "-g") && (i + 1 < argc)) {
      grid_path = argv[++i];
    } else if ((arg == "-l") && (i + 1 < argc)) {
      load_path = argv[++i];
    } else if ((arg == "-n") && (i + 1 < argc)) {
      numerics_path = argv[++i];
    } else {
      std::cerr << "Unknown argument or missing value: " << arg << std::endl;
    }
  }

  if (numerics_path.empty()) {
    numerics_path = "numerics.yaml";
  }
  if (fileExists(numerics_path)) {
    config.numerics = config.parse_numerics(numerics_path);
  } else {
    std::cerr << "Warning: Numerics file not specified and not found in current directory, using default values." << std::endl;
  }

  if (material_path.empty()) {
    material_path = "material.yaml";
  }
  if (!fileExists(material_path)) {
    throw std::runtime_error("Material file not specified and not found in current directory.");
  }

  if (!load_path.empty()) {
    config.load = config.parse_load(load_path);
  } else {
    throw std::runtime_error("Tension file not specified.");
  }

  if (!grid_path.empty()) {
    config.vti_file_content = read_file(grid_path);
  } else {
    throw std::runtime_error("Grid file not specified.");
  }


  f_materialpoint_initBase(material_path.c_str(), std::size(material_path),
                           load_path.c_str(), std::size(load_path),
                           grid_path.c_str(), std::size(grid_path),
                           numerics_path.c_str(), std::size(numerics_path));
  DiscretizationGrid grid_;
  grid_.init(false, config.vti_file_content);
  f_materialpoint_initDamask();

  Spectral spectral;
  // TODO: use spectral derivative from numerics yaml
  spectral.init(0, grid_, config.numerics.fft.fftw_planner_flag, config.numerics.fft.fftw_timelimit);

  DamaskGrid damask_grid;
  damask_grid.init(config, grid_, spectral);

  int stag_it_max = 10;
  int max_cut_back = 3;

  damask_grid.loop_over_loadcase(config.load.steps, stag_it_max, max_cut_back);

  return 0;
}
