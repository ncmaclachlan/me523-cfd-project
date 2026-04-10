#pragma once
#include "modules/base_module.hpp"

class OutputModule : public SimModule {
public:
    void execute(const RunConfig& cfg, SimState& state) override;
};
