#include <yaml.h>
#include <iostream>
#include <string>
#include <stdexcept>

void parse_grid_mechanical(yaml_parser_t *parser, Config::NumGridMechanical& grid_mechanical, std::string& errors) {
    yaml_token_t token;
    bool inMechanical = false;

    while (1) {
        yaml_parser_scan(parser, &token);

        if (token.type == YAML_KEY_TOKEN && token.data.scalar.value != NULL) {
            std::string key(reinterpret_cast<char*>(token.data.scalar.value));
            inMechanical = true;
            yaml_token_delete(&token);
            yaml_parser_scan(parser, &token); // Move to the next token for value

            if (token.type != YAML_SCALAR_TOKEN) {
                continue;
            }

            if (inMechanical) {
                std::string value(reinterpret_cast<char*>(token.data.scalar.value));

                if (key == "N_iter_min") {
                    int itmin = std::stoi(value);
                    if (itmin < 1) {
                        errors += "N_iter_min must be >= 1\n";
                    } else {
                        grid_mechanical.itmin = itmin;
                    }
                } else if (key == "N_iter_max") {
                    int itmax = std::stoi(value);
                    if (itmax <= 1) {
                        errors += "N_iter_max must be > 1\n";
                    } else {
                        grid_mechanical.itmax = itmax;
                    }
                } else if (key == "update_gamma") {
                    grid_mechanical.update_gamma = (value == "true");
                } else if (key == "eps_abs_div(P)") {
                    double eps_abs_div = std::stod(value);
                    if (eps_abs_div <= 0) {
                        errors += "eps_abs_div(P) must be > 0\n";
                    } else {
                        grid_mechanical.eps_abs_div = eps_abs_div;
                    }
                } else if (key == "eps_rel_div(P)") {
                    double eps_rel_div = std::stod(value);
                    if (eps_rel_div <= 0) {
                        errors += "eps_rel_div(P) must be >= 0\n";
                    } else {
                        grid_mechanical.eps_rel_div = eps_rel_div;
                    }
                } else if (key == "eps_abs_P") {
                    double eps_abs_P = std::stod(value);
                    if (eps_abs_P <= 0) {
                        errors += "eps_abs_P must be > 0\n";
                    } else {
                        grid_mechanical.eps_abs_P = eps_abs_P;
                    }
                } else if (key == "eps_rel_P") {
                    double eps_rel_P = std::stod(value);
                    if (eps_rel_P <= 0) {
                        errors += "eps_rel_P must be >= 0\n";
                    } else {
                        grid_mechanical.eps_rel_P = eps_rel_P;
                    }
                } else {
                    errors += "Unknown key: " + key + "\n";
                }
            }
        } else if (token.type == YAML_MAPPING_END_TOKEN) {
            if (inMechanical) {
                break; // Exit loop when end of the mechanical mapping is reached
            }
        }

        yaml_token_delete(&token);

        if (token.type == YAML_STREAM_END_TOKEN) {
            break;
        }
    }
}





Config::Numerics Config::parse_numerics_yaml(std::string yamlFilePath) {
  Numerics numerics;
	std::string yamlContent = read_file(yamlFilePath);

	YAML::Node rootNode = YAML::Load(yamlContent);

  // TODO: Implement remaining numerics parameters (examples/numerics.yaml)
  YAML::Node solver_node = root_node["solver"];
  YAML::Node grid_node = solver_node["grid"];
  YAML::Node grid_mechanical_node = grid_node["mechanical"];
  YAML::Node grid_damage_node = grid_node["mechanical"];
  YAML::Node grid_thermal_node = grid_node["mechanical"];
  YAML::Node grid_FFT_node = grid_node["FFT"];

  YAML::Node homogenization_node = root_node["homogenization"];
  YAML::Node homogenization_mechanical_node = homogenization_node["homogenization"];
  YAML::Node homogenization_mechanical_RGC_node = homogenization_mechanical_node["RGC"];  

	std::string errors = "";

  for (const auto& key_ : grid_node) {
    std::string key = key_.first.as<std::string>();
		if (key == "maxStaggeredIter") {
			int max_staggered_iter = key_.second.as<int>();
			if (max_staggered_iter < 0) { 
				errors += "maxStaggeredIter must be > 0\n";
			} else {
				numerics.max_staggered_iter = max_staggered_iter;
			};
		} else if (key == "maxCutBack") {
			int max_cut_back = key_.second.as<int>();
			if (max_cut_back < 0) { 
				errors += "maxCutBack must be > 0\n";
			} else {
				numerics.max_cut_back = max_cut_back;
			};
		} else {
			errors+= std::string("Unknown key:") + key;
		}
	}

  for (const auto& key_ : grid_mechanical_node) {
    std::string key = key_.first.as<std::string>();
		if (key == "N_iter_min") {
				int itmin = key_.second.as<int>();
				if (itmin < 1)  {
					errors += "N_iter_min must be >= 1\n";
				} else {
					numerics.grid.itmin = itmin;
				}
		} else if (key == "N_iter_max") {
				int itmax = key_.second.as<int>();
				if (itmax <= 1)  {
					errors += "N_iter_max must be > 1\n";
				} else {
					numerics.grid.itmax = itmax;
				}
		} else if (key == "update_gamma") {
				numerics.grid.update_gamma = key_.second.as<bool>();
		} else if (key == "eps_abs_div(P)") {
				double eps_abs_div = key_.second.as<double>();
				if (eps_abs_div <= 0)  {
					errors += "eps_abs_div(P) must be > 0\n";
				} else {
					numerics.grid.eps_abs_div = eps_abs_div;
				}
		} else if (key == "eps_rel_div(P)") {
				double eps_rel_div = key_.second.as<double>();
				if (eps_rel_div <= 0)  {
					errors += "eps_rel_div(P) must be >= 0\n";
				} else {
					numerics.grid.eps_rel_div = eps_rel_div;
				}
		} else if (key == "eps_abs_P") {
				double eps_abs_P = key_.second.as<double>();
				if (eps_abs_P <= 0)  {
					errors += "eps_abs_P must be > 0\n";
				} else {
					numerics.grid.eps_abs_P = eps_abs_P;
				}
		} else if (key == "eps_rel_P") {
				double eps_rel_P = key_.second.as<double>();
				if (eps_rel_P <= 0)  {
					errors += "eps_rel_P must be >= 0\n";
				} else {
					numerics.grid.eps_rel_P = eps_rel_P;
				}
		} else {
				errors+= std::string("Unknown key:") + key;
		}
	}

  for (const auto& key_ : grid_FFT_node) {
    std::string key = key_.first.as<std::string>();
		if (key == "memory_efficient") {
				numerics.memory_efficient = key_.second.as<int>();
		} else if (key == "divergence_correction") {
				int divergence_correction = key_.second.as<int>();
				if (divergence_correction < 0 || divergence_correction > 2) 
					errors += "divergence_correction must be => 0 and <= 2\n";
				numerics.fft.divergence_correction = divergence_correction;
		} else if (key == "derivative") {
				std::string derivative = key_.second.as<std::string>();
        if (derivative == "continuous") {
            numerics.spectral_derivative_id = DERIVATIVE_CONTINUOUS_ID;
        } else if (derivative == "central_difference") {
            numerics.spectral_derivative_id = DERIVATIVE_CENTRAL_DIFF_ID;
        } else if (derivative == "FWBW_difference") {
            numerics.spectral_derivative_id = DERIVATIVE_FWBW_DIFF_ID;
        } else {
          errors += "derivative must be either 'continuous', 'central_difference' or 'FWBW_difference'\n";
        };

    } else if (key == "FFTW_plan_mode") {
      std::string fftw_plan_mode = key_.second.as<std::string>();
      if (fftw_plan_mode == "fftw_estimate") {
        numerics.fft.fftw_planner_flag = FFTW_ESTIMATE;
      } else if (fftw_plan_mode == "fftw_measure") {
        numerics.fft.fftw_planner_flag = FFTW_MEASURE;
      } else if (fftw_plan_mode == "fftw_patient") {
        numerics.fft.fftw_planner_flag = FFTW_PATIENT;
      } else if (fftw_plan_mode == "fftw_exhaustive") {
        numerics.fft.fftw_planner_flag = FFTW_EXHAUSTIVE;
      } else {
        errors += "using default 'FFTW_MEASURE' flag in 'fftw_plan_mode' instead of unknown specified '" + fftw_plan_mode + "'\n";
        numerics.fft.fftw_planner_flag = FFTW_MEASURE;  // Default value
      }
		} else if (key == "FFTW_timelimit") {
				double fftw_timelimit = key_.second.as<double>();
				if (fftw_timelimit <= 0)  {
					errors += "fftw_timelimit must be > 0\n";
				} else {
					numerics.fft.fftw_timelimit = fftw_timelimit;
				}
		} else if (key == "petsc_options") {
				numerics.petsc_options = key_.second.as<std::string>();
		} else if (key == "alpha") {
				double alpha = key_.second.as<double>();
				if (alpha < 0 || alpha > 2)  {
					errors += "alpha must be > 0 and <= 2\n";
				} else {
					numerics.alpha = alpha;
				}
		} else if (key == "beta") {
				double beta = key_.second.as<double>();
				if (beta < 0 || beta > 2)  {
					errors += "beta must be => 0 and <= 2\n";
				} else {
					numerics.beta = beta;
				}
		} else if (key == "eps_abs_curl_F") {
				double eps_abs_curl_F = key_.second.as<double>();
				if (eps_abs_curl_F <= 0)  {
					errors += "eps_abs_curl_F must be > 0\n";
				} else {
					numerics.fft.eps_abs_curl_F = eps_abs_curl_F;
				}
		} else if (key == "eps_rel_curl_F") {
			double eps_rel_curl_F = key_.second.as<double>();
			if (eps_rel_curl_F <= 0)  {
				errors += "eps_rel_curl_F must be >= 0\n";
			} else {
				numerics.fft.eps_rel_curl_F = eps_rel_curl_F;
			}
		} else {
			errors+= std::string("Unknown key:") + key;
		}
	}


	if (errors != ""){
		throw std::runtime_error("errors when parsing numerics yaml: \n" + errors);
	}
  return numerics;
}


		} else if (key == "eps_thermal_atol") {
				double eps_thermal_atol = key_.second.as<double>();
				if (eps_thermal_atol <= 0)  {
					errors += "eps_thermal_atol must be > 0\n";
				} else {
					numerics.fft.eps_thermal_atol = eps_thermal_atol;
				}
		} else if (key == "eps_thermal_rtol") {
				double eps_thermal_rtol = key_.second.as<double>();
				if (eps_thermal_rtol <= 0)  {
					errors += "eps_thermal_rtol must be > 0\n";
				} else {
					numerics.fft.eps_thermal_rtol = eps_thermal_rtol;
				}
		} else if (key == "eps_damage_atol") {
				double eps_damage_atol = key_.second.as<double>();
				if (eps_damage_atol <= 0)  {
					errors += "eps_damage_atol must be > 0\n";
				} else {
					numerics.fft.eps_damage_atol = eps_damage_atol;
				}
		} else if (key == "eps_damage_rtol") {
				double eps_damage_rtol = key_.second.as<double>();
				if (eps_damage_rtol <= 0)  {
					errors += "eps_damage_rtol must be > 0\n";
				} else {
					numerics.fft.eps_damage_rtol = eps_damage_rtol;
				}
		} else if (key == "phi_min") {
				double phi_min = key_.second.as<double>();
				if (phi_min <= 0)  {
					errors += "phi_min must be => 0\n";
				} else {
					numerics.phi_min = phi_min;
				}


  }

