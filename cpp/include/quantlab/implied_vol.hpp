#pragma once

#include <vector>

#include "quantlab/black_scholes.hpp"

namespace quantlab {

// Black-Scholes implied volatility (book Ch 4.1): the sigma for which
//   bs_price(type, S0, K, T, r, sigma) == price.
// Uses the book's combined root-finding method: Newton-Raphson (with vega as
// the derivative), falling back to bisection whenever Newton would leave the
// bracket [lo, hi] that is known to contain the root.
// Throws std::invalid_argument if the price violates the no-arbitrage bounds.
double implied_vol(OptionType type, double price, double S0, double K, double T, double r);

// Same algorithm, but returns every iterate sigma_0, sigma_1, ..., sigma_final.
// Used by the app to show how the solver converges.
std::vector<double> implied_vol_iterates(OptionType type, double price, double S0,
                                         double K, double T, double r);

} // namespace quantlab