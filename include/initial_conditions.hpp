#pragma once
#include "base_module.hpp"

class InitialConditionsModule : public SimModule {
public:
    void execute(const RunConfig& cfg, SimState& state) override;
};
