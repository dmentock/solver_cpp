#include <iostream>
#include <fstream>
#include <sstream>
#include <yaml-cpp/yaml.h>

#include <yaml_reader.h>

std::string readFileContent(const std::string& filePath) {
	std::ifstream inFile(filePath);
	std::stringstream buffer;
	buffer << inFile.rdbuf();
	return buffer.str();
}

// std::string yamlNodeToString(const YAML::Node& node) {
//     YAML::Emitter out;
//     out << node;
//     return out.c_str();
// }
// std::string getTypeName(const std::type_info& typeInfo) {
//     if (typeInfo == typeid(int)) {
//         return "int";
//     } else if (typeInfo == typeid(double)) {
//         return "double";
//     } else if (typeInfo == typeid(bool)) {
//         return "bool";
//     } else {
//         // Return the raw type name if not found in the list above
//         return typeInfo.name();
//     }
// }




void YamlReader::parse_num_grid_yaml(std::string yamlFilePath) {
	std::string yamlContent = readFileContent(yamlFilePath);

	YAML::Node rootNode = YAML::Load(yamlContent);
	YAML::Node gridNode = rootNode["grid"];

	std::string errors = "";

  for (const auto& key_ : gridNode) {
    std::string key = key_.first.as<std::string>();
		if (key == "itmin") {
				int itmin = key_.second.as<int>();
				if (itmin < 1)  {
					errors += "itmin must be >= 1\n";
				} else {
					num_grid.itmin = itmin;
				}
		} else if (key == "itmax") {
				int itmax = key_.second.as<int>();
				if (itmax <= 1)  {
					errors += "itmax must be > 1\n";
				} else {
					num_grid.itmax = itmax;
				}
		} else if (key == "memory_efficient") {
				num_grid.memory_efficient = key_.second.as<int>();
		} else if (key == "divergence_correction") {
				int divergence_correction = key_.second.as<int>();
				if (divergence_correction < 0 || divergence_correction > 2) 
					errors += "divergence_correction must be => 0 and <= 2\n";
				num_grid.divergence_correction = divergence_correction;
		} else if (key == "update_gamma") {
				num_grid.update_gamma = key_.second.as<bool>();
		} else if (key == "eps_div_atol") {
				double eps_div_atol = key_.second.as<double>();
				if (eps_div_atol <= 0)  {
					errors += "eps_div_atol must be > 0\n";
				} else {
					num_grid.eps_div_atol = eps_div_atol;
				}
		} else if (key == "eps_div_rtol") {
				double eps_div_rtol = key_.second.as<double>();
				if (eps_div_rtol <= 0)  {
					errors += "eps_div_rtol must be >= 0\n";
				} else {
					num_grid.eps_div_rtol = eps_div_rtol;
				}
		} else if (key == "eps_stress_atol") {
				double eps_stress_atol = key_.second.as<double>();
				if (eps_stress_atol <= 0)  {
					errors += "eps_stress_atol must be > 0\n";
				} else {
					num_grid.eps_stress_atol = eps_stress_atol;
				}
		} else if (key == "eps_stress_rtol") {
				double eps_stress_rtol = key_.second.as<double>();
				if (eps_stress_rtol <= 0)  {
					errors += "eps_stress_rtol must be >= 0\n";
				} else {
					num_grid.eps_stress_rtol = eps_stress_rtol;
				}
		} else if (key == "eps_curl_atol") {
				double eps_curl_atol = key_.second.as<double>();
				if (eps_curl_atol <= 0)  {
					errors += "eps_curl_atol must be > 0\n";
				} else {
					num_grid.eps_curl_atol = eps_curl_atol;
				}
		} else if (key == "eps_curl_rtol") {
				double eps_curl_rtol = key_.second.as<double>();
				if (eps_curl_rtol <= 0)  {
					errors += "eps_curl_rtol must be >= 0\n";
				} else {
					num_grid.eps_curl_rtol = eps_curl_rtol;
				}
		} else if (key == "alpha") {
				double alpha = key_.second.as<double>();
				if (alpha < 0 || alpha > 2)  {
					errors += "alpha must be > 0 and <= 2\n";
				} else {
					num_grid.alpha = alpha;
				}
		} else if (key == "beta") {
				double beta = key_.second.as<double>();
				if (beta < 0 || beta > 2)  {
					errors += "beta must be => 0 and <= 2\n";
				} else {
					num_grid.beta = beta;
				}
		} else if (key == "eps_thermal_atol") {
				double eps_thermal_atol = key_.second.as<double>();
				if (eps_thermal_atol <= 0)  {
					errors += "eps_thermal_atol must be > 0\n";
				} else {
					num_grid.eps_thermal_atol = eps_thermal_atol;
				}
		} else if (key == "eps_thermal_rtol") {
				double eps_thermal_rtol = key_.second.as<double>();
				if (eps_thermal_rtol <= 0)  {
					errors += "eps_thermal_rtol must be > 0\n";
				} else {
					num_grid.eps_thermal_rtol = eps_thermal_rtol;
				}
		} else if (key == "eps_damage_atol") {
				double eps_damage_atol = key_.second.as<double>();
				if (eps_damage_atol <= 0)  {
					errors += "eps_damage_atol must be > 0\n";
				} else {
					num_grid.eps_damage_atol = eps_damage_atol;
				}
		} else if (key == "eps_damage_rtol") {
				double eps_damage_rtol = key_.second.as<double>();
				if (eps_damage_rtol <= 0)  {
					errors += "eps_damage_rtol must be > 0\n";
				} else {
					num_grid.eps_damage_rtol = eps_damage_rtol;
				}
		} else if (key == "phi_min") {
				double phi_min = key_.second.as<double>();
				if (phi_min <= 0)  {
					errors += "phi_min must be => 0\n";
				} else {
					num_grid.phi_min = phi_min;
				}
		} else if (key == "petsc_options") {
				num_grid.petsc_options = key_.second.as<std::string>();
		} else if (key == "derivative") {
				std::string derivative = key_.second.as<std::string>();
				if (derivative != "continuous" && derivative != "central_difference" && derivative != "FWBW_difference"){ 
					errors += "derivative must be either 'continuous', 'central_difference' or 'FWBW_difference'\n";
				} else {
					num_grid.derivative = derivative;
				};
		} else if (key == "fftw_plan_mode") {
				std::string fftw_plan_mode = key_.second.as<std::string>();
				if (fftw_plan_mode != "fftw_estimate" && fftw_plan_mode != "fftw_measure" && 
						fftw_plan_mode != "fftw_patient"  && fftw_plan_mode != "fftw_exhaustive"){ 
					errors += "using default 'FFTW_MEASURE' flag in 'fftw_plan_mode' instead of specified '" + fftw_plan_mode + "'";
					num_grid.fftw_plan_mode = "fftw_measure";
				} else {
					num_grid.fftw_plan_mode = fftw_plan_mode;
				};
		} else {
				errors+= std::string("Unknown key:") + key;
		}
  }
	if (errors != ""){
		throw std::runtime_error("errors when parsing numerics yaml: \n" + errors);
	}
}

	// auto assignYamlValue = [&](const std::string& key, auto val) {
	// 		try {
	// 				num_grid[key] = gridNode[key].as<decltype(val)>();
	// 		} catch (YAML::BadConversion) {
	// 			throw std::runtime_error("invalid format for numerics key '" + key +
	// 											"', requires type '"+getTypeName(typeid(decltype(val)))+"'");
	// 		}
	// };

	// for (const auto& [key, val] : num_grid) {
	// 		std::cout << "Key: " << key << std::endl;
	// 		if (gridNode[key].IsDefined()) {
	// 				if (std::holds_alternative<int>(val)) {
	// 					// if (key == "itmin" &&)
	// 					int val = assignYamlValue(key, int{});
	// 					num_grid[key] = val;
	// 				} else if (std::holds_alternative<bool>(val)) {
	// 					assignYamlValue(key, bool{});
	// 				} else if (std::holds_alternative<double>(val)) {
	// 					assignYamlValue(key, double{});
	// 				}
	// 		}
	// }
	// if (gridNode["itmin"].IsDefined()) {
	// 	params.itmin = gridNode["itmin"].as<int>();
	// 	if params.itmin
	// }
	// // params.itmin = gridNode["itmin"].as<int>();
	// params.itmax = gridNode["itmax"].as<int>();

	// std::cout << "itmin: " << gridNode["itmin"] << ", itmax: " << gridNode["itmax"] << std::endl;


