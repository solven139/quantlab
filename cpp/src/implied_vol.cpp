#include "quantlab/implied_vol.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace quantlab {

std::vector<double> implied_vol_iterates(OptionType type, double price, double S0,
                                         double K, double T, double r) {
    if (S0 <= 0.0 || K <= 0.0 || T <= 0.0)
        throw std::invalid_argument("implied_vol: S0, K, T must be positive");

    // No-arbitrage bounds: outside these, no volatility can reproduce the price
    const double df = std::exp(-r * T);
    const double lower = (type == OptionType::Call) ? std::max(S0 - K * df, 0.0)
                                                    : std::max(K * df - S0, 0.0);
    const double upper = (type == OptionType::Call) ? S0 : K * df;
    if (!(price > lower && price < upper))
        throw std::invalid_argument("implied_vol: price outside no-arbitrage bounds");

    // g(sigma) = model price - market price. It is increasing in sigma (vega > 0).
    auto g = [&](double s) { return bs_price(type, S0, K, T, r, s) - price; };

    double lo = 1e-4;  // bracket [lo, hi] that contains the root
    double hi = 5.0;
    if (g(lo) > 0.0 || g(hi) < 0.0)
        throw std::invalid_argument("implied_vol: root not inside [0.0001, 5]");

    const double tol = 1e-12;   // stop when the price is matched this closely
    const int max_iter = 100;

    double sigma = 0.2;         // initial guess
    std::vector<double> iterates;

    for (int k = 0; k < max_iter; ++k) {
        iterates.push_back(sigma);
        const double f = g(sigma);
        if (std::abs(f) < tol || hi - lo < 1e-15)
            break;

        // TODO 1: shrink the bracket
        if (f > 0.0)
            hi = sigma;
        else
            lo = sigma;

        // TODO 2: Newton-Raphson step
        const double v = bs_vega(S0, K, T, r, sigma);
        const double newton = sigma - f / v;

        // TODO 3: take Newton if it stays inside the bracket, otherwise bisect
        if (v > 1e-12 && newton > lo && newton < hi)
            sigma = newton;
        else
            sigma = 0.5 * (lo + hi);
    }
    return iterates;
}

double implied_vol(OptionType type, double price, double S0, double K, double T, double r) {
    return implied_vol_iterates(type, price, S0, K, T, r).back();
}

} // namespace quantlab