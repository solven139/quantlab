#include "quantlab/black_scholes.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>


namespace quantlab {

double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

namespace {
// d1 from the Black-Scholes formula; used by the price and every Greek
double compute_d1(double S0, double K, double T, double r, double sigma) {
    return (std::log(S0 / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T));
}
} // namespace

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

// ---------- Step 1: your code goes below ----------

double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * std::numbers::pi);
}

double bs_delta(OptionType type, double S0, double K, double T, double r, double sigma) {
    const double d1 = compute_d1(S0, K, T, r, sigma);

    if (type == OptionType::Call)
        return norm_cdf(d1);
    return norm_cdf(d1) - 1.0;
}

double bs_gamma(double S0, double K, double T, double r, double sigma) {
    const double d1 = compute_d1(S0, K, T, r, sigma);
    return norm_pdf(d1) / (S0 * sigma * std::sqrt(T));
}

double bs_vega(double S0, double K, double T, double r, double sigma) {
    const double d1 = compute_d1(S0, K, T, r, sigma);
    return S0 * norm_pdf(d1) * std::sqrt(T);
}

} // namespace quantlab
