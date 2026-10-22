#pragma once

#include <vector>

namespace quantlab {

// SABR model (Hagan, Kumar, Lesniewski & Woodward 2002), book Ch 4.3.3:
//   dF = sigma F^beta dW_F,   d sigma = nu sigma dW_sigma,   dW_F dW_sigma = rho dt,
//   sigma(0) = alpha.
//
// SABR is NOT a Model in the quantlab sense: it has no characteristic function for log F(T),
// so the COS pricer cannot use it. What it has instead is Hagan's asymptotic formula for the
// Black (lognormal) implied volatility, book eq. 4.61. It is used as a smile
// parametrisation: fit alpha, rho, nu to quotes for one expiry (beta fixed), then read
// vols, and hence prices, at any strike.
double sabr_implied_vol(double F, double K, double T, double alpha, double beta, double rho,
                        double nu);

std::vector<double> sabr_implied_vols(double F, const std::vector<double>& Ks, double T,
                                      double alpha, double beta, double rho, double nu);

} // namespace quantlab
