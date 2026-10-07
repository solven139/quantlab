#include "quantlab/black_scholes.hpp"

#include <cmath>
#include <stdexcept>

namespace quantlab {

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

double bs_price(OptionType type, double S0, double K, double T, double r, double sigma) {
    if (S0 <= 0.0 || K <= 0.0 || T <= 0.0 || sigma <= 0.0)
        throw std::invalid_argument("bs_price: S0, K, T, sigma must be positive");

    const double sqrtT = std::sqrt(T);
    const double d1 = (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrtT);
    const double d2 = d1 - sigma * sqrtT;
    const double df = std::exp(-r * T);

    if (type == OptionType::Call)
        return S0 * norm_cdf(d1) - K * df * norm_cdf(d2);
    return K * df * norm_cdf(-d2) - S0 * norm_cdf(-d1);
}

} // namespace quantlab
