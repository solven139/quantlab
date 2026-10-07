#pragma once

#include <cstdint>
#include <vector>

namespace quantlab {

// Simulate Geometric Brownian Motion under the risk-neutral measure (book Ch 2.1):
//   dS = r S dt + sigma S dW
// using the exact solution on a grid of n_steps equal time steps.
//
// Returns n_paths * (n_steps + 1) values, row by row:
//   path 0: S(t0), S(t1), ..., S(tN), then path 1, ...
std::vector<double> simulate_gbm(double S0, double r, double sigma, double T,
                                 int n_steps, int n_paths, std::uint64_t seed);

} // namespace quantlab