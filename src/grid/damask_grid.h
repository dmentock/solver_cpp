#ifndef D3915447_4D0E_4170_97A7_199DCA32D4B6
#define D3915447_4D0E_4170_97A7_199DCA32D4B6
#ifndef DAMASK_GRID_H
#define DAMASK_GRID_H

#include <unsupported/Eigen/CXX11/Tensor>

#include "config.h"
#include "spectral.h"
#include "mech_base.h"
#include "mech_solver_basic.h"

using namespace std;
using namespace Eigen;

extern "C" {
  void f_materialpoint_forward();
  void f_materialpoint_result(int* total_inc_counter, double* t);
}

class DamaskGrid {
public:

  struct Fields {
    std::unique_ptr<MechBase> mech = nullptr;
    std::unique_ptr<MechBase> thermal = nullptr;
    std::unique_ptr<MechBase> damage = nullptr;
  };

  struct SolutionStates {
    std::optional<Config::SolutionState> mech;
    std::optional<Config::SolutionState> thermal;
    std::optional<Config::SolutionState> damage;
  };

  Fields fields;

  void init(Config& config, DiscretizationGrid& grid_, Spectral& spectral);
  void loop_over_loadcase(std::vector<Config::LoadStep> load_steps, int stag_it_max, int max_cut_back);
  virtual bool solve_fields(SolutionStates& solution_states, double& delta_t, std::string& inc_info, int& stag_it_max);
  virtual void forward_fields(Config::LoadStep& load_step, 
                              bool& cut_back, bool& guess, 
                              double& delta_t, double& delta_t_prev, double& t_remaining);
  static double forward_time(double& delta_t, double& delta_t_prev, Config::LoadStep& load_step,
                                int& inc, int& sub_step_factor, int& cut_back_level);


  virtual void materialpoint_forward() {
    f_materialpoint_forward();
  }
  virtual void materialpoint_result(int& total_inc_counter, double& t) {
    f_materialpoint_result(&total_inc_counter, &t);
  }

private:
  int world_rank;
  int n_total_load_steps;

};
#endif // DAMASK_GRID_H


#endif /* D3915447_4D0E_4170_97A7_199DCA32D4B6 */
