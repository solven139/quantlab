#include "quantlab/gbm.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace quantlab {

std::vector<double> simulate_gbm(double S0, double r, double sigma, double T,
                                 int n_steps, int n_paths, std::uint64_t seed) {
    if (S0 <= 0.0 || sigma < 0.0 || T <= 0.0 || n_steps < 1 || n_paths < 1)
        throw std::invalid_argument("simulate_gbm: invalid input");

    const double dt = T / n_steps;
    const double drift = (r - 0.5 * sigma * sigma) * dt;  // deterministic part of log-step
    const double vol = sigma * std::sqrt(dt);             // random part scale

    std::mt19937_64 rng(seed);                     // random number generator
    std::normal_distribution<double> Z(0.0, 1.0);  // standard normal draws

    const int n_cols = n_steps + 1;
    std::vector<double> paths(static_cast<std::size_t>(n_paths) * n_cols);

    for (int p = 0; p < n_paths; ++p) {
        double* row = &paths[static_cast<std::size_t>(p) * n_cols];
        row[0] = S0;
        for (int i = 1; i < n_cols; ++i) {
            row[i] = row[i - 1] * std::exp(drift + vol * Z(rng));
        }
    }
    return paths;
}

} // namespace quantlab