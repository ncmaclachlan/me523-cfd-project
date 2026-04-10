#include "structs.hpp"
#include "modules/grid_module.hpp"
#include "modules/initial_conditions_module.hpp"
#include "modules/solver_module.hpp"
#include "modules/output_module.hpp"

int main() {
    RunConfig cfg;
    SimState  state;

    GridModule              grid;
    InitialConditionsModule ic;
    SolverModule            solver;
    OutputModule            output;

    grid.execute(cfg, state);
    ic.execute(cfg, state);
    solver.execute(cfg, state);
    output.execute(cfg, state);

    return 0;
}
