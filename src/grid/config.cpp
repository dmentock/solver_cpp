#include "config.h"
#include <fftw3.h>
#include "utilities_tensor.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <variant>
#include <map>
#include <optional>
#include <Eigen/Geometry>

struct YamlNode;
using YamlVariant = std::variant<std::string, int, double, bool, std::shared_ptr<std::vector<YamlNode>>, std::shared_ptr<std::map<std::string, YamlNode>>>;
struct YamlNode { YamlVariant data; };

using MapNodePointer = std::shared_ptr<std::map<std::string, YamlNode>>;
using MapNodeIterator = std::map<std::string, YamlNode>::iterator;
using MapNodePointerOptional = std::optional<MapNodePointer>;

using ListNodePointer = std::shared_ptr<std::vector<YamlNode>>;
using ListNodePointerOptional = std::optional<ListNodePointer>;

// helper method
void printYamlNode(const YamlNode& node, int depth = 0) {
  const auto& data = node.data;
  std::string indent(depth * 2, ' ');

  if (std::holds_alternative<std::shared_ptr<std::vector<YamlNode>>>(data)) {
      const auto& vec = *std::get<std::shared_ptr<std::vector<YamlNode>>>(data);
      std::cout << indent << "List:" << std::endl;
      for (const auto& item : vec) {
          printYamlNode(item, depth + 1);
      }
  } else if (std::holds_alternative<std::shared_ptr<std::map<std::string, YamlNode>>>(data)) {
      const auto& map = *std::get<std::shared_ptr<std::map<std::string, YamlNode>>>(data);
      std::cout << indent << "Map:" << std::endl;
      for (const auto& [key, value] : map) {
          std::cout << indent << "  " << key << ": ";
          printYamlNode(value, depth + 2);
      }
  } else if (std::holds_alternative<std::string>(data)) {
      std::cout << indent << "String: " << std::get<std::string>(data) << std::endl;
  } else if (std::holds_alternative<int>(data)) {
      std::cout << indent << "Int: " << std::get<int>(data) << std::endl;
  } else if (std::holds_alternative<double>(data)) {
      std::cout << indent << "Double: " << std::get<double>(data) << std::endl;
  } else {
      std::cout << indent << "Unknown type" << std::endl;
  }
}

YamlVariant parseScalar(yaml_event_t& event) {
  std::string value(reinterpret_cast<char*>(event.data.scalar.value));
  std::string lower_case_value = value;
  std::transform(lower_case_value.begin(), lower_case_value.end(), lower_case_value.begin(),
                  [](unsigned char c){ return std::tolower(c); });
  if (lower_case_value == "true") {
      return true;
  } else if (lower_case_value == "false") {
      return false;
  }
  try {
      size_t idx;
      int int_val = std::stoi(value, &idx);
      if (idx == value.size()) return int_val;

      double double_val = std::stod(value, &idx);
      if (idx == value.size()) return double_val;
  } catch (std::exception&) {}
  return value;
}

YamlNode parse_flow(yaml_parser_t& parser, bool& done) {
  yaml_event_t event;
  YamlNode node;

  while (!done) {
    if (!yaml_parser_parse(&parser, &event)) {
        std::cerr << "Failed to parse YAML." << std::endl;
        done = true;
        return node;
    }
    switch (event.type) {
      case YAML_SCALAR_EVENT: {
        node.data = parseScalar(event);
        yaml_event_delete(&event);
        return node;
      }
      case YAML_SEQUENCE_START_EVENT: {
        auto seq = std::make_shared<std::vector<YamlNode>>();
        bool seq_done = false;
        while (!seq_done) {
          YamlNode item = parse_flow(parser, seq_done);
          if (seq_done) {
            seq_done = false;
            break;
          }
          seq->push_back(item);
        }
        node.data = seq;
        return node;
      }
      case YAML_SEQUENCE_END_EVENT:
        done = true;
        return node;
      case YAML_MAPPING_START_EVENT: {
        auto map = std::make_shared<std::map<std::string, YamlNode>>();
        bool map_done = false;
        std::string key;
        while (!map_done) {
          YamlNode value = parse_flow(parser, map_done);
          if (!key.empty()) {
            (*map)[key] = value;
            key.clear();
          } else if (std::holds_alternative<std::string>(value.data)) {
            key = std::get<std::string>(value.data);
          }
        }
        node.data = map;
        yaml_event_delete(&event);
        return node;
      }
      case YAML_MAPPING_END_EVENT:
      case YAML_DOCUMENT_END_EVENT:
        done = true;
        yaml_event_delete(&event);
        return node;
      default:
        yaml_event_delete(&event);
        break;
    }
  }
  return node;
}

MapNodePointer get_map_node(const YamlNode& node) {
  auto result = std::get_if<MapNodePointer>(&node.data);
  return result ? *result : nullptr;
}

ListNodePointer get_list_node(const YamlNode& node) {
  auto result = std::get_if<ListNodePointer>(&node.data);
  return result ? *result : nullptr;
}

template<typename T>
const T* get_scalar(const YamlNode& node) {
  return std::get_if<T>(&node.data);
}

MapNodePointer get_mandatory_map_node(const MapNodePointer& parent_map, const std::string& key, const std::string& error_msg) {
  auto iter = parent_map->find(key);
  if (iter == parent_map->end()) {
    throw std::runtime_error("YAML Error: " + error_msg);
  }
  return get_map_node(iter->second);
}

std::optional<MapNodePointer> get_optional_map_node(const MapNodePointer& parent_map, const std::string& key) {
  auto iter = parent_map->find(key);
  if (iter != parent_map->end()) {
    return get_map_node(iter->second);
  }
  return std::nullopt;
}

ListNodePointer get_mandatory_list_node(const MapNodePointer& parent_map, const std::string& key, const std::string& error_msg) {
  auto iter = parent_map->find(key);
  if (iter == parent_map->end()) {
    throw std::runtime_error("YAML Error: " + error_msg);
  }
  return get_list_node(iter->second);
}

std::optional<ListNodePointer> get_optional_list_node(const MapNodePointer& parent_map, const std::string& key) {
  auto iter = parent_map->find(key);
  if (iter != parent_map->end()) {
    return get_list_node(iter->second);
  }
  return std::nullopt;
}

template<typename T>
T get_mandatory_scalar(const MapNodePointer& map, const std::string& key, const std::string& error_msg) {
  auto iter = map->find(key);
  if (iter != map->end()) {
    if (const T* value = get_scalar<T>(iter->second)) {
      return *value;
    }
  }
  throw std::runtime_error(error_msg);
}

template<typename T>
std::optional<T> get_optional_scalar(const MapNodePointer& map, const std::string& key) {
  auto iter = map->find(key);
  if (iter != map->end()) {
    if (const T* value = get_scalar<T>(iter->second)) {
      return *value;
    }
  }
  return std::nullopt;
}

void parse_boundary_condition_matrices(const ListNodePointer& vec, Matrix<double, 3, 3>& values, Matrix<bool, 3, 3>& mask) {
    if (vec->size() != 3) {
        throw std::runtime_error("Expected a 3x3 matrix, but got a different size");
    }
    size_t rowIndex = 0;
    for (const auto& rowItem : *vec) {
        auto rowVector = get_list_node(rowItem);
        if (!rowVector || rowVector->size() != 3) {
            throw std::runtime_error("Expected a 3x3 matrix, but one of the rows has a different size");
        }
        size_t colIndex = 0;
        for (const auto& scalarItem : *rowVector) {
            if (auto strValue = get_scalar<std::string>(scalarItem)) {
                if (*strValue == "x") {
                    mask(rowIndex, colIndex) = true;
                } else {
                    throw std::runtime_error("Invalid string value in matrix: expected 'x'");
                }
            } else if (auto doubleVal = get_scalar<double>(scalarItem)) {
                values(rowIndex, colIndex) = static_cast<double>(*doubleVal);
            } else if (auto intVal = get_scalar<int>(scalarItem)) {
                values(rowIndex, colIndex) = static_cast<double>(*intVal);
            } else {
                throw std::runtime_error("Non-numeric value found in matrix");
            }
            colIndex++;
        }
        rowIndex++;
    }
}

FILE* open_file(const std::string& path) {
  FILE* file = fopen(path.c_str(), "r");
  if (!file) {
    throw runtime_error("Failed to open file: " + path);
  }
  return file;
};

Config::Load Config::parse_load(const std::string& path) {
  Config::Load load;
  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  FILE* file = open_file(path);
  yaml_parser_set_input_file(&parser, file);

  bool done = false;
  YamlNode root_node = parse_flow(parser, done);
  printYamlNode(root_node);

  MapNodePointer top_level_map = get_map_node(root_node);
  if (!top_level_map) throw std::runtime_error("YAML Error: numerics yaml contains no data");

  MapNodePointer solver_map = get_mandatory_map_node(top_level_map, "solver", "loadcase dict needs to define 'solver'");
  std::optional<std::string> mech_solver = get_optional_scalar<std::string>(solver_map, "mechanical");
  if (mech_solver.has_value()) load.fields["mechanical"] = mech_solver.value();
  std::optional<std::string> thermal_solver = get_optional_scalar<std::string>(solver_map, "thermal");
  if (thermal_solver.has_value()) load.fields["thermal"] = thermal_solver.value();
  std::optional<std::string> damage_solver = get_optional_scalar<std::string>(solver_map, "damage");
  if (damage_solver.has_value()) load.fields["damage"] = damage_solver.value();

  ListNodePointer loadstep_list = get_mandatory_list_node(top_level_map, "loadstep", "Load definition is missing 'loadstep' entry");
  int loadstep_index = 0;
  for (const auto& list_item : *loadstep_list) {
    LoadStep load_step;
    
    MapNodePointer list_item_map = get_map_node(list_item);
    if (!list_item_map) throw std::runtime_error("YAML Error: Item in loadstep list contains no data");

    load_step.f_out = get_mandatory_scalar<int>(list_item_map, "f_out", "'f_out' is missing in loadstep");

    MapNodePointer discretization_map = get_mandatory_map_node(list_item_map, "discretization", "loadstep entry needs to define 'discretization'");
    load_step.t = get_mandatory_scalar<int>(discretization_map, "t", "'t' is missing in loadstep.discretization");
    load_step.N = get_mandatory_scalar<int>(discretization_map, "N", "'N' is missing in loadstep.discretization");
    n_total_load_steps+=load_step.N;
    auto r = get_optional_scalar<int>(discretization_map, "r");
    if (r.has_value()) load_step.r = r.value();

    MapNodePointer bc_map = get_mandatory_map_node(list_item_map, "boundary_conditions", "loadstep entry needs to define 'boundary_conditions'");
    MapNodePointer mechanical_map = get_mandatory_map_node(bc_map, "mechanical", "loadstep.boundary_conditions entry needs to define 'mechanical'");


    std::array<std::string, 3> deformation_formulations = {"F", "dot_F", "L"};
    bool deformation_found = false;
    for (const auto& deformation : deformation_formulations) {
      std::optional<ListNodePointer> deformation_list = get_optional_list_node(mechanical_map, deformation);
      if (deformation_list.has_value()) {
        if (load_step.deformation.type!="") {
          throw std::runtime_error("Only one deformation specification allowed, found more than one in loadstep at index " + loadstep_index);
        }
        parse_boundary_condition_matrices(deformation_list.value(), load_step.deformation.values, load_step.deformation.mask);
        load_step.deformation.type = deformation;
      }
    }
    if (load_step.deformation.type=="") {
      throw std::runtime_error("Deformation specification required for each loadstep, not specified for loadstep at index" + loadstep_index);
    }

    std::array<std::string, 3> stress_formulations = {"P", "dot_P"};
    for (const auto& stress : stress_formulations) {
      std::optional<ListNodePointer> stress_list = get_optional_list_node(mechanical_map, stress);
      if (stress_list.has_value()) {
        if (load_step.stress.type!="") {
          throw std::runtime_error("Only one stress specification allowed, found more than one in loadstep at index " + loadstep_index);
        }
        parse_boundary_condition_matrices(stress_list.value(), load_step.stress.values, load_step.stress.mask);
        load_step.stress.type = stress;
      };
    }
    if (load_step.stress.type=="") {
      throw std::runtime_error("Stress specification required for each loadstep, not specified for loadstep at index" + loadstep_index);
    }

    auto estimate_rate = get_optional_scalar<bool>(list_item_map, "estimate_rate");
    if (estimate_rate.has_value()) load_step.estimate_rate = estimate_rate.value();
    auto f_out = get_optional_scalar<int>(list_item_map, "f_out");
    if (f_out.has_value()) load_step.f_out = f_out.value();
    auto f_restart = get_optional_scalar<int>(list_item_map, "f_restart");
    if (f_restart.has_value()) load_step.f_restart = f_restart.value();

    std::optional<ListNodePointer> R_list = get_optional_list_node(mechanical_map, "R");
     if (R_list.has_value()) {
        if ((*R_list)->size() == 4) {
            std::vector<double> quaternionValues;
            quaternionValues.reserve(4);
            for (const auto& r_item : **R_list) {
                double val;
                if (auto doubleValue = get_scalar<double>(r_item)) {
                    val = *doubleValue;
                } else if (auto intValue = get_scalar<int>(r_item)) {
                    val = static_cast<double>(*intValue);
                } else {
                    throw std::runtime_error("Invalid quaternion value: must be a number");
                }
                quaternionValues.push_back(val);
            }
            load_step.rot_bc_q = Eigen::Quaterniond(quaternionValues[0],
                                                    quaternionValues[1],
                                                    quaternionValues[2],
                                                    quaternionValues[3]);
        } else {
            throw std::runtime_error("YAML Error: R entry in loadstep must have exactly 4 elements");
        }
    }
    load.steps.push_back(load_step);
    loadstep_index++;
  }
  yaml_parser_delete(&parser);
  fclose(file);
  return load;
}

Config::Numerics Config::parse_numerics(const std::string& path) {
  Config::Numerics numerics;
  std::string errors = "";

  yaml_parser_t parser;
  yaml_parser_initialize(&parser);
  FILE* file = open_file(path);
  yaml_parser_set_input_file(&parser, file);

  bool done = false;
  YamlNode root_node = parse_flow(parser, done);
  printYamlNode(root_node);

  MapNodePointer top_level_map = get_map_node(root_node);
  if (!top_level_map) throw std::runtime_error("YAML Error: loadcase yaml contains no data");

  MapNodePointerOptional solver_map = get_optional_map_node(top_level_map, "solver");
  if (solver_map.has_value()) {
    MapNodePointerOptional grid_map = get_optional_map_node(solver_map.value(), "grid");
    if (grid_map.has_value()) {

      std::optional<int> stag_iter_max = get_optional_scalar<int>(grid_map.value(), "N_staggered_iter_max");
      if (stag_iter_max.has_value()) numerics.stag_iter_max = stag_iter_max.value();
      std::optional<int> max_cut_back = get_optional_scalar<int>(grid_map.value(), "N_cutback_max");
      if (max_cut_back.has_value()) numerics.max_cut_back = max_cut_back.value();

      MapNodePointerOptional mechanical_map = get_optional_map_node(grid_map.value(), "mechanical");
      if (mechanical_map.has_value()) {
        std::optional<int> itmin = get_optional_scalar<int>(mechanical_map.value(), "N_iter_min");
        if (itmin.has_value()) numerics.grid_mechanical.itmin = itmin.value();
        std::optional<int> itmax = get_optional_scalar<int>(mechanical_map.value(), "N_iter_max");
        if (itmax.has_value()) numerics.grid_mechanical.itmax = itmax.value();
        std::optional<double> eps_abs_div = get_optional_scalar<double>(mechanical_map.value(), "eps_abs_div(P)");
        if (eps_abs_div.has_value()) numerics.grid_mechanical.eps_abs_div = eps_abs_div.value();
        std::optional<double> eps_rel_div = get_optional_scalar<double>(mechanical_map.value(), "eps_rel_div(P)");
        if (eps_rel_div.has_value()) numerics.grid_mechanical.eps_rel_div = eps_rel_div.value();
        std::optional<double> eps_abs_P = get_optional_scalar<double>(mechanical_map.value(), "eps_abs_P");
        if (eps_abs_P.has_value()) numerics.grid_mechanical.eps_abs_P = eps_abs_P.value();
        std::optional<double> eps_rel_P = get_optional_scalar<double>(mechanical_map.value(), "eps_rel_P");
        if (eps_rel_P.has_value()) numerics.grid_mechanical.eps_rel_P = eps_rel_P.value();
        std::optional<bool> update_gamma = get_optional_scalar<bool>(mechanical_map.value(), "update_gamma");
        if (update_gamma.has_value()) numerics.grid_mechanical.update_gamma = update_gamma.value();
      }

      MapNodePointerOptional fft_map = get_optional_map_node(grid_map.value(), "FFT");
      if (fft_map.has_value()) {
        std::optional<bool> memory_efficient = get_optional_scalar<bool>(fft_map.value(), "memory_efficient");
        if (memory_efficient.has_value()) numerics.fft.memory_efficient = memory_efficient.value();
        std::optional<std::string> divergence_correction = get_optional_scalar<std::string>(fft_map.value(), "divergence_correction");
        if (divergence_correction.has_value()) {
          if (divergence_correction.value() == "none") {
              numerics.fft.divergence_correction = Config::DIVERGENCE_CORRECTION_NONE_ID;
          } else if (divergence_correction.value() == "size") {
              numerics.fft.divergence_correction = Config::DIVERGENCE_CORRECTION_SIZE_ID;
          } else if (divergence_correction.value() == "size+grid" || divergence_correction.value() == "grid+size") {
              numerics.fft.divergence_correction = Config::DIVERGENCE_CORRECTION_SIZE_GRID_ID;
          } else {
              errors += "fft.divergence_correction must be either 'none', 'size' or 'size+grid/grid+size'\n";
          }
        }
        std::optional<std::string> derivative = get_optional_scalar<std::string>(fft_map.value(), "derivative");
        if (fft_map.has_value()) {
          if (derivative.value() == "continuous") {
              numerics.fft.derivative = Config::DERIVATIVE_CONTINUOUS_ID;
          } else if (derivative.value() == "central_difference") {
              numerics.fft.derivative = Config::DERIVATIVE_CENTRAL_DIFF_ID;
          } else if (derivative.value() == "FWBW_difference") {
              numerics.fft.derivative = Config::DERIVATIVE_FWBW_DIFF_ID;
          } else {
              errors += "fft.derivative must be either 'continuous', 'central_difference' or 'FWBW_difference'\n";
          }
        }
        std::optional<std::string> fftw_planner_flag = get_optional_scalar<std::string>(fft_map.value(), "FFTW_plan_mode");
        if (fftw_planner_flag.has_value()) {
          if (fftw_planner_flag.value() == "fftw_estimate") {
              numerics.fft.fftw_planner_flag = FFTW_ESTIMATE;
          } else if (fftw_planner_flag.value() == "fftw_measure") {
              numerics.fft.fftw_planner_flag = FFTW_MEASURE;
          } else if (fftw_planner_flag.value() == "fftw_patient") {
              numerics.fft.fftw_planner_flag = FFTW_PATIENT;
          } else if (fftw_planner_flag.value() == "fftw_exhaustive") {
              numerics.fft.fftw_planner_flag = FFTW_EXHAUSTIVE;
          } else {
              errors += "Invalid value for FFTW_plan_mode: " + fftw_planner_flag.value() + "\n";
          }
        }
        std::optional<double> fftw_timelimit = get_optional_scalar<double>(fft_map.value(), "FFTW_timelimit");
        if (fftw_timelimit.has_value()) numerics.fft.fftw_timelimit = fftw_timelimit.value();
        std::optional<std::string> petsc_options = get_optional_scalar<std::string>(fft_map.value(), "PETSc_options");
        if (petsc_options.has_value()) numerics.fft.petsc_options = petsc_options.value();
        std::optional<double> alpha = get_optional_scalar<double>(fft_map.value(), "alpha");
        if (alpha.has_value()) numerics.fft.alpha = alpha.value();
        std::optional<double> beta = get_optional_scalar<double>(fft_map.value(), "beta");
        if (beta.has_value()) numerics.fft.beta = beta.value();
        std::optional<double> eps_abs_curl_F = get_optional_scalar<double>(fft_map.value(), "eps_abs_curl(F)");
        if (eps_abs_curl_F.has_value()) numerics.fft.eps_abs_curl_F = eps_abs_curl_F.value();
        std::optional<double> eps_rel_curl_F = get_optional_scalar<double>(fft_map.value(), "eps_rel_curl(F)");
        if (eps_rel_curl_F.has_value()) numerics.fft.eps_rel_curl_F = eps_rel_curl_F.value();
      }
    }
  }
  yaml_parser_delete(&parser);
  fclose(file);
  return numerics;
}
