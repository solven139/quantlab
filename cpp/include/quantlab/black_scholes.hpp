#pragma once

namespace quantlab {

enum class OptionType { Call, Put };

// Standard normal CDF, N(x)
double norm_cdf(double x);

// Black-Scholes price of a European option (book eq. 3.21-3.22)
//   S0: spot, K: strike, T: maturity (years), r: risk-free rate, sigma: volatility
double bs_price(OptionType type, double S0, double K, double T, double r, double sigma);

} // namespace quantlab
