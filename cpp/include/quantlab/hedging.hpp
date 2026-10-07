#pragma once

#include <cstdint>
#include <vector>

namespace quantlab {

// Delta-hedging experiment for a sold European call (book Ch 3.3, eqs. 3.42-3.43).
//
// At t0 we SELL one call for its Black-Scholes price at volatility sigma_hedge,
// buy Delta shares, and rebalance n_rebalance times until T, financing the
// shares through a bank account that earns rate r.
// The stock itself moves as GBM with volatility sigma_true.
//
// Returns the final P&L(T) of each of the n_paths simulated paths.
std::vector<double> delta_hedge_pnl(double S0, double K, double T, double r,
                                    double sigma_true, double sigma_hedge,
                                    int n_rebalance, int n_paths, std::uint64_t seed);

} // namespace quantlab