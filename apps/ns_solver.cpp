#include "structs.hpp"
#include "grid.hpp"
#include "initial_conditions.hpp"
#include "solver.hpp"
#include "output.hpp"

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
