#ifndef D3915447_4D0E_4170_97A7_199DCA32D4B6
#define D3915447_4D0E_4170_97A7_199DCA32D4B6
#ifndef DAMASK_GRID_H
#define DAMASK_GRID_H

#include <unsupported/Eigen/CXX11/Tensor>

#include <config.h>
#include <spectral.h>
#include <mech_base.h>
#include <mech_solver_basic.h>

using namespace std;
using namespace Eigen;

extern "C" {
  void f_materialpoint_initAll();
}

class DamaskGrid {
public:
  void init(Config& config, DiscretizationGrid& grid, Spectral& spectral);
  void loop_over_loadcase(std::vector<Config::LoadStep> load_steps);
  std::unique_ptr<MechBase> mech_field_ptr;
};
#endif // DAMASK_GRID_H


#endif /* D3915447_4D0E_4170_97A7_199DCA32D4B6 */
