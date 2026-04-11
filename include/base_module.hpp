#pragma once
#include "structs.hpp"

class SimModule {
public:
    virtual ~SimModule() = default;
    virtual void execute(const RunConfig& cfg, SimState& state) = 0;
};
