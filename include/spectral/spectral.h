#ifndef SPECTRAL_H
#define SPECTRAL_H

#include <discretization_grid.h>

 // here we will put all the utilities stuff
struct tNumerics {
    bool update_gamma;
    int itmin, itmax;
    double eps_div_atol, eps_div_rtol, eps_stress_atol, eps_stress_rtol;
};

class Spectral {
public:
    Spectral(DiscretizationGrid& grid_) : grid(grid_) {}
protected:
    DiscretizationGrid& grid;
};
#endif // SPECTRAL_H