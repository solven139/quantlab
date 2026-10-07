#include "quantlab/jump_paths.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace quantlab {

std::vector<double> simulate_merton(double S0, double r, double sigma, double xi,
                                    double mu_j, double sigma_j, double T,
                                    int n_steps, int n_paths, std::uint64_t seed) {
    if (S0 <= 0.0 || sigma < 0.0 || xi < 0.0 || T <= 0.0 || n_steps < 1 || n_paths < 1)
        throw std::invalid_argument("simulate_merton: invalid input");

    const double dt = T / n_steps;
    const double omega = xi * (std::exp(mu_j + 0.5 * sigma_j * sigma_j) - 1.0);
    const double drift = (r - omega - 0.5 * sigma * sigma) * dt;
    const double vol = sigma * std::sqrt(dt);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);
    std::poisson_distribution<int> n_jumps(xi * dt);

    const int n_cols = n_steps + 1;
    std::vector<double> paths(static_cast<std::size_t>(n_paths) * n_cols);
    for (int p = 0; p < n_paths; ++p) {
        double* row = &paths[static_cast<std::size_t>(p) * n_cols];
        row[0] = S0;
        double logS = std::log(S0);
        for (int i = 1; i < n_cols; ++i) {
            logS += drift + vol * Z(rng);
            const int n = n_jumps(rng);
            for (int j = 0; j < n; ++j)
                logS += mu_j + sigma_j * Z(rng);
            row[i] = std::exp(logS);
        }
    }
    return paths;
}

} // namespace quantlab
