#include "quantlab/hedging.hpp"
#include "quantlab/black_scholes.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace quantlab {

std::vector<double> delta_hedge_pnl(double S0, double K, double T, double r,
                                    double sigma_true, double sigma_hedge,
                                    int n_rebalance, int n_paths, std::uint64_t seed) {
    if (S0 <= 0.0 || K <= 0.0 || T <= 0.0 || sigma_true < 0.0 || sigma_hedge <= 0.0 ||
        n_rebalance < 1 || n_paths < 1)
        throw std::invalid_argument("delta_hedge_pnl: invalid input");

    const int m = n_rebalance;
    const double dt = T / m;
    const double growth = std::exp(r * dt);  // bank account interest factor per step
    const double drift = (r - 0.5 * sigma_true * sigma_true) * dt;
    const double vol = sigma_true * std::sqrt(dt);

    // Price we charge and first hedge: same for every path
    const double V0 = bs_price(OptionType::Call, S0, K, T, r, sigma_hedge);
    const double delta0 = bs_delta(OptionType::Call, S0, K, T, r, sigma_hedge);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);
    std::vector<double> result(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double S = S0;
        double delta = delta0;
        double pnl = V0 - delta0 * S0;  // eq. 3.42, first line

        // Rebalancing dates t1 ... t_{m-1}
        for (int i = 1; i < m; ++i) {
            S *= std::exp(drift + vol * Z(rng));  // the stock moves one step
            const double tau = T - i * dt;        // time left to maturity

            const double new_delta = bs_delta(OptionType::Call, S, K, tau, r, sigma_hedge);
            pnl = pnl * growth - (new_delta - delta) * S;  // eq. 3.42
            delta = new_delta;
        }

        // Maturity t_m = T: the stock moves one last step
        S *= std::exp(drift + vol * Z(rng));
        pnl = pnl * growth - std::max(S - K, 0.0) + delta * S;  // eq. 3.43

        result[p] = pnl;
    }
    return result;
}

} // namespace quantlab