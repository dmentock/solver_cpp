#ifndef MECH_H
#define MECH_H

#include <spectral/spectral.h>
#include <vector>
#include <array>
#include <string>
#include <memory>

class Mechanical : public Spectral{
public:
    Mechanical(DiscretizationGrid& grid_) : Spectral(grid_) {}

};
#endif // MECH_H