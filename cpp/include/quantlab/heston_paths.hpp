#pragma once

#include <cstdint>
#include <vector>

namespace quantlab {

struct HestonPaths {
    std::vector<double> S;  // n_paths * (n_steps + 1), row by row
    std::vector<double> v;  // same layout: the variance paths
    int zero_hits = 0;      // how many times the Euler variance went below zero (then floored)
};

// Euler scheme with "full truncation" for the Heston model (book Ch 9.4 discusses better
// schemes; Step 6 compares them). Correlated normals: Z_v = rho Z_x + sqrt(1-rho^2) Z_perp.
HestonPaths simulate_heston(double S0, double r, double kappa, double vbar, double gamma,
                            double rho, double v0, double T, int n_steps, int n_paths,
                            std::uint64_t seed);

} // namespace quantlab
