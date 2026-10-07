#pragma once

#include <cstdint>
#include <vector>

namespace quantlab {

// Simulate Merton jump-diffusion paths (book Ch 5.1) on a grid of n_steps steps.
// In each step the log-price gets the GBM increment plus a Poisson(xi*dt) number
// of Normal(mu_j, sigma_j^2) jumps. Same flat row-by-row layout as simulate_gbm.
std::vector<double> simulate_merton(double S0, double r, double sigma, double xi,
                                    double mu_j, double sigma_j, double T,
                                    int n_steps, int n_paths, std::uint64_t seed);

} // namespace quantlab
