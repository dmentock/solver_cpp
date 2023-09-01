#include <damask_grid.h>
#include <config.h>
#include <spectral.h>
#include <cstdio>
#include <mech_solver_basic.h>

DamaskGrid::Fields DamaskGrid::init(Config& config, DiscretizationGrid& grid, Spectral& spectral) {

  // write to resultfile
  Fields fields;
  if (config.fields.find("mechanical") != config.fields.end()) {
    if (config.fields["mechanical"] == "spectral_basic") {
      fields.mech = std::make_unique<MechSolverBasic>(config, grid, spectral);
    } else {
      throw std::runtime_error("unkown key for mechanical field specified in loadfile: " + config.fields["mechanical"]);
    }
  } else {
    throw std::runtime_error("mandatory mechanical field key in loadfile is not specified");
  }
  fields.mech->init();
  return fields;
}

void DamaskGrid::forward_fields(Fields& fields, Config::LoadStep& load_step, 
                                bool& cut_back, bool& guess, 
                                double& Delta_t, double& Delta_t_prev, double& t_remaining) {
  if (fields.mech) {
    fields.mech->forward (cut_back, guess, Delta_t, Delta_t_prev, t_remaining, 
                          load_step.deformation, load_step.stress, load_step.rot_bc_q);
  }
  // if (thermal_field_ptr) {
  //   thermal_field_ptr->forward(cut_back);
  // }
  // if (mech_field_ptr) {
  //   thermal_field_ptr->forward(cut_back);
  // }
  if (!cut_back) f_materialpoint_forward();
}

bool DamaskGrid::solve_fields(Fields& fields, SolutionStates& solution_states, double& Delta_t, std::string& inc_info, int& stag_it_max) {  
  int stag_iter = 0;
  bool stag_iterating = true;
  while (stag_iterating) {
    if (fields.mech) {
      solution_states.mech = fields.mech->calculate_solution(inc_info);
      if (!solution_states.mech->converged) break;
    }
    // if (fields.thermal) {
    //   solution_states.thermal = fields.thermal->calculate_solution(Delta_t);
    //   if (!solution_states.thermal.converged) break;
    // }
    // if (fields.damage) {
    //   solution_states.damage = fields.damage->calculate_solution(Delta_t);
    //   if (!solution_states.Tdamage.converged) break;
    // }
    stag_iter++;
    // TODO: it could be confusing to set stag_converged to false when the solution actually converged, maybe rename?
    stag_iterating = stag_iter <= stag_it_max &&
      (solution_states.mech.has_value() ?    solution_states.mech->converged &&    !solution_states.mech->stag_converged: false) &&
      (solution_states.thermal.has_value() ? solution_states.thermal->converged && !solution_states.thermal->stag_converged : true) &&
      (solution_states.damage.has_value() ?  solution_states.damage->converged &&  !solution_states.damage->stag_converged : true);
  }

    // cout << "aa " <<  stag_iter  << " " << stag_it_max << endl;
    // cout << "aa " <<  solution_states.mech.has_value() << endl;
    // cout << "aa " <<  solution_states.mech->converged << endl;
    // cout << "aa " <<  solution_states.mech->stag_converged << endl;
    // cout << "wq " << !solution_states.mech->terminally_ill << endl; 
    // cout << "bb " << !solution_states.mech->stag_converged << endl;
  if ((solution_states.mech.has_value() ?    solution_states.mech->converged &&       solution_states.mech->stag_converged && 
        !solution_states.mech->terminally_ill: false) &&
      (solution_states.thermal.has_value() ? solution_states.thermal->converged &&    solution_states.thermal->stag_converged : true) &&
      (solution_states.damage.has_value() ?  solution_states.damage->converged &&     solution_states.damage->stag_converged : true)) {
    return true;
  } else {
    return false;
  }
}

double DamaskGrid::forward_time(double& Delta_t, double& Delta_t_prev, Config::LoadStep& load_step,
                              int& inc, int& sub_step_factor, int& cut_back_level) {
  Delta_t_prev = Delta_t; // last time interval that brought former inc to an end
  if (std::abs(load_step.r - 1.0) < 1.e-9) { // linear scale
    Delta_t = load_step.t / load_step.N;
  } else {
      Delta_t = load_step.t * (pow(load_step.r, inc) - pow(load_step.r, inc + 1)) /
                (1.0 - pow(load_step.r, load_step.N));
  }
  Delta_t *= pow(static_cast<double>(sub_step_factor), -static_cast<double>(cut_back_level)); // decrease time step depending on cut back level
  return Delta_t;
}

void DamaskGrid::loop_over_loadcase(std::vector<Config::LoadStep> load_steps, Fields& fields, int stag_it_max, int max_cut_back) {
    // Begin loadCaseLooping
  bool cut_back = false;

  int sub_step_factor = 2;
  int cut_back_level = 0;
  int total_inc_counter = 0;

  double t = 0;
  double Delta_t = 1;
  double Delta_t_prev = 0;
  double t_remaining = 0;
  
  SolutionStates solution_states;

  for (int l = 0; l < load_steps.size(); ++l) {
    double t_0 = t; // load case start time
    bool guess = load_steps[l].estimate_rate; // homogeneous guess for the first inc
    for (int inc = 0; inc < load_steps[l].N; ++inc) {
      total_inc_counter++;
// --------------------------------------------------------------------------------------------------
// forwarding time
      Delta_t = forward_time(Delta_t, Delta_t_prev, load_steps[l], inc, sub_step_factor, cut_back_level);

      int CLI_restartInc = 0; // TODO
      if (total_inc_counter <= CLI_restartInc) { // not yet at restart inc?
        t += Delta_t; // just advance time, skip already performed calculation
        guess = true; // forced guessing instead of inheriting loadcase preference
      } else {
        int step_fraction = 0; // fraction scaled by stepFactor**cutLevel
// --------------------------------------------------------------------------------------------------
// Begin subStepLooping
        while (step_fraction < pow(sub_step_factor, cut_back_level)) {
          t_remaining = load_steps[l].t + t_0 - t;
          t += Delta_t; // forward target time
          step_fraction++; // count step
// --------------------------------------------------------------------------------------------------
// report beginning of new step
          std::cout << "\n###########################################################################" << std::endl;
          std::cout << "Time " << t << "s: Increment " << inc+1 << '/' << load_steps[l].N
                    << '-' << step_fraction << '/' << pow(sub_step_factor, cut_back_level)
                    << " of load case " << l << '/' << load_steps.size() << std::endl;

          std::ostringstream oss;
          oss << "Increment " << total_inc_counter << "/" << 
            std::accumulate(load_steps.begin(), load_steps.end(), 0,
            [](int total, const Config::LoadStep& step) {
              return total + step.N;
            }) << 
          "-" << step_fraction << "/" << pow(sub_step_factor, cut_back_level);
          std::string inc_info = oss.str();
// --------------------------------------------------------------------------------------------------
// forward and solve fields
          forward_fields(fields, load_steps[l], cut_back, guess, Delta_t, Delta_t_prev, t_remaining);
          bool solution_found = solve_fields(fields, solution_states, Delta_t, inc_info, stag_it_max);
// -------------------------------------  -------------------------------------------------------------
// check solution and either advance or retry with smaller timestep
          if (solution_found) {
            fields.mech->update_coords();
            Delta_t_prev = Delta_t;
            cut_back = false;
            guess = true;
          } else if (cut_back_level < max_cut_back) {
            cut_back = true;
            step_fraction = (step_fraction - 1) * sub_step_factor;
            cut_back_level++;
            t = t-Delta_t;
            Delta_t = Delta_t / sub_step_factor;
            cout << "cutting back" << endl;
          } else {
	          throw std::runtime_error("no solution found for: \n" + inc_info);
          }
        } // End subStepLooping
// --------------------------------------------------------------------------------------------------
// set up next increment
        cut_back_level = std::max(0, cut_back - 1); // try half number of subincs next inc

        if ((solution_states.mech.has_value() ? solution_states.mech->converged: true) &&
            (solution_states.thermal.has_value() ? solution_states.thermal->converged: true) &&
            (solution_states.damage.has_value() ? solution_states.damage->converged: true)) {
          std::cout << "\n increment " << total_inc_counter << " converged" << std::endl;
        } else {
          std::cout << "\n increment " << total_inc_counter << " NOT converged" << std::endl;
        }

        bool sig = false;
        // int err_MPI = MPI_Allreduce(&signal_SIGUSR1, &sig, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD);
        // if (err_MPI != MPI_SUCCESS) throw std::runtime_error("MPI error");
        if (inc % load_steps[l].f_out == 0 || sig) {
          std::cout << "\n ... saving results ........................................................" << std::endl;
          materialpoint_result(total_inc_counter, t);
        }
        // exit(0);

        // if (sig) signal_setSIGUSR1(false);
        // MPI_Allreduce(&signal_SIGUSR2, &sig, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD, &err_MPI);
        // if (err_MPI != MPI_SUCCESS) throw std::runtime_error("MPI error");
        // if (inc % loadCases[l].f_restart == 0 || sig) {
        //     for (int field = 0; field < nActiveFields; ++field) {
        //         switch (ID[field]) {
        //             case FIELD_MECH_ID:
        //                 mechanical_restartWrite();
        //                 break;
        //             case FIELD_THERMAL_ID:
        //                 grid_thermal_spectral_restartWrite();
        //                 break;
        //             case FIELD_DAMAGE_ID:
        //                 grid_damage_spectral_restartWrite();
        //                 break;
        //         }
        //     }
        //     materialpoint_restartWrite();
        // }
        // if (sig) signal_setSIGUSR2(false);
        // MPI_Allreduce(&signal_SIGINT, &sig, 1, MPI_C_BOOL, MPI_LOR, MPI_COMM_WORLD, &err_MPI);
        // if (err_MPI != MPI_SUCCESS) throw std::runtime_error("MPI error");
        // if (sig) break; // exit loadCaseLooping

      } // End skipping
    } // End incLooping
  } // End loadCaseLooping
}