#pragma once
#include "base_module.hpp"

class SolverModule : public SimModule {
public:
    void execute(const RunConfig& cfg, SimState& state) override;
};
