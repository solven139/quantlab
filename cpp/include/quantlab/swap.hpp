#pragma once

#include <cstdint>
#include <vector>

#include "quantlab/hull_white.hpp"

namespace quantlab {

// Plain vanilla interest rate swap (book Ch 12.1.4).
// Payment dates T_1, ..., T_m spaced by tau, starting after T_0 = start.
// A payer swap pays the fixed rate K and receives the floating (Libor) rate.
struct Swap {
    double notional = 1.0;
    double K = 0.03;        // fixed rate
    double start = 0.0;     // T_0: first reset date
    double end = 10.0;      // T_m: last payment date
    double tau = 1.0;       // accrual period between payments
    bool payer = true;      // true: pay fixed, receive float; false: the opposite

    std::vector<double> payment_dates() const;   // T_1, ..., T_m
};

// Today's value from the curve alone (book eq. 12.12):
//   V(0) = N [ P(0,T_0) - P(0,T_m) - K * sum_k tau P(0,T_k) ]   (payer; receiver = minus)
double swap_value_today(const NelsonSiegelCurve& curve, const Swap& swap);

// Annuity A(0) = sum_k tau P(0, T_k)  (book eq. 12.13)
double swap_annuity(const NelsonSiegelCurve& curve, const Swap& swap);

// Par swap rate: the K that makes today's value zero (book eq. 12.14)
double par_swap_rate(const NelsonSiegelCurve& curve, const Swap& swap);

// Monte Carlo exposure simulation (book Example 12.3.1).
// Simulates Hull-White on a grid with steps_per_year steps and revalues every swap
// on every path at every grid time with the closed-form bond prices P(t, T).
struct ExposureSimulation {
    std::vector<double> times;      // n_times
    std::vector<double> values;     // n_trades * n_paths * n_times: V_trade(t) on each path
    std::vector<double> discount;   // n_paths * n_times: exp(-int_0^t r ds) = M(0)/M(t)
    int n_trades = 0, n_paths = 0, n_times = 0;
};

ExposureSimulation simulate_exposure(const HullWhiteModel& model, const std::vector<Swap>& trades,
                                     double horizon, int steps_per_year, int n_paths,
                                     std::uint64_t seed);

// Unilateral CVA, book eq. 12.61, with a constant default intensity (hazard rate) h:
//   CVA = LGD * sum_k EE(t_k) * [ F(t_k) - F(t_{k-1}) ],   F(t) = 1 - exp(-h t)
double cva(const std::vector<double>& times, const std::vector<double>& ee, double lgd,
           double hazard);

} // namespace quantlab
