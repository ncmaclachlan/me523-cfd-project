# me523-cfd-project
CFD course project Navier-Stokes + X.

## Build

Requires CMake 3.16+ and a C++17 compiler.

```bash
cd code
mkdir -p build && cd build
cmake ..
cmake --build .
```

The executable is placed at `build/executables/ns_solver`.
