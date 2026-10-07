#include "quantlab/heston_paths.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace quantlab {

HestonPaths simulate_heston(double S0, double r, double kappa, double vbar, double gamma,
                            double rho, double v0, double T, int n_steps, int n_paths,
                            std::uint64_t seed) {
    if (S0 <= 0.0 || T <= 0.0 || n_steps < 1 || n_paths < 1 || v0 < 0.0)
        throw std::invalid_argument("simulate_heston: invalid input");

    const double dt = T / n_steps;
    const double sq_dt = std::sqrt(dt);
    const double rho_perp = std::sqrt(1.0 - rho * rho);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> N01(0.0, 1.0);

    const int n_cols = n_steps + 1;
    HestonPaths out;
    out.S.resize(static_cast<std::size_t>(n_paths) * n_cols);
    out.v.resize(static_cast<std::size_t>(n_paths) * n_cols);

    for (int p = 0; p < n_paths; ++p) {
        double* S = &out.S[static_cast<std::size_t>(p) * n_cols];
        double* v = &out.v[static_cast<std::size_t>(p) * n_cols];
        S[0] = S0;
        v[0] = v0;
        double logS = std::log(S0);
        for (int i = 1; i < n_cols; ++i) {
            const double zx = N01(rng);
            const double zv = rho * zx + rho_perp * N01(rng);   // correlated with zx
            const double vp = std::max(v[i - 1], 0.0);          // "full truncation"
            logS += (r - 0.5 * vp) * dt + std::sqrt(vp) * sq_dt * zx;
            double vn = v[i - 1] + kappa * (vbar - vp) * dt + gamma * std::sqrt(vp) * sq_dt * zv;
            if (vn < 0.0) {
                ++out.zero_hits;
                vn = 0.0;
            }
            v[i] = vn;
            S[i] = std::exp(logS);
        }
    }
    return out;
}

} // namespace quantlab
