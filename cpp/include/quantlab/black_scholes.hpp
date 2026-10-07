#pragma once

namespace quantlab {

enum class OptionType { Call, Put };

// Standard normal CDF, N(x)
double norm_cdf(double x);

// Black-Scholes price of a European option (book eq. 3.21-3.22)
//   S0: spot, K: strike, T: maturity (years), r: risk-free rate, sigma: volatility
double bs_price(OptionType type, double S0, double K, double T, double r, double sigma);

// Standard normal PDF, phi(x)
double norm_pdf(double x);

// Greeks: partial derivatives of bs_price
double bs_delta(OptionType type, double S0, double K, double T, double r, double sigma); // dV/dS0
double bs_gamma(double S0, double K, double T, double r, double sigma);                  // d2V/dS0^2 (same for call and put)
double bs_vega (double S0, double K, double T, double r, double sigma);                  // dV/dsigma (same for call and put)

} // namespace quantlab
