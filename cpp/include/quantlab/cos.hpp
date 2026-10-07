#pragma once

#include <vector>

#include "quantlab/black_scholes.hpp"
#include "quantlab/models.hpp"

namespace quantlab {

// European option price by the COS method (book Ch 6, eq. 6.28):
//   V ~= exp(-rT) * sum'_{k=0}^{N-1} Re{ phi(u_k) exp(i u_k (x - a)) } * H_k,
//   u_k = k pi / (b - a),  x = log(S0 / K),  sum' = first term weighted by 1/2.
// N: number of cosine terms. L: width of the integration range in standard deviations.
double cos_price(const Model& model, OptionType type, double S0, double K, double T,
                 int N = 128, double L = 8.0);

// Recover the density of X = log(S(T)/S0) on the points in xs from the
// characteristic function alone (book Ch 6.1, the idea behind the COS method).
std::vector<double> cos_density(const Model& model, double T, const std::vector<double>& xs,
                                int N = 128, double L = 8.0);

} // namespace quantlab
