#pragma once

#include <cstddef>

// Broad abstract grid interface.
// This is intentionally small: it exposes only what should be common
// to many grid types, including higher-dimensional ones.
class Grid {
public:
    virtual ~Grid() = default;

    // Spatial dimension of the grid, e.g. 2 for a 2D grid.
    virtual std::size_t dim() const = 0;

    // Number of cells along a given axis.
    // axis = 0 -> x, axis = 1 -> y, axis = 2 -> z, etc.
    virtual int cells(std::size_t axis) const = 0;

    // Physical domain length along a given axis.
    // axis = 0 -> Lx, axis = 1 -> Ly, axis = 2 -> Lz, etc.
    virtual double domain_length(std::size_t axis) const = 0;

    // Grid spacing along a given axis.
    // axis = 0 -> dx, axis = 1 -> dy, axis = 2 -> dz, etc.
    virtual double spacing(std::size_t axis) const = 0;
};