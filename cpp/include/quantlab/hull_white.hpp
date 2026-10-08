#pragma once

#include <cstdint>
#include <vector>

namespace quantlab {

// Today's yield curve, Nelson-Siegel form: zero rate
//   y(T) = b0 + b1 (1 - e^-x)/x + b2 ((1 - e^-x)/x - e^-x),   x = T / tau
// b0: long-run level, b1: short end minus long end (slope), b2: hump (curvature).
class NelsonSiegelCurve {
public:
    NelsonSiegelCurve(double b0, double b1, double b2, double tau);

    double zero_rate(double T) const;       // y(T)
    double discount(double T) const;        // P(0, T) = exp(-y(T) T)
    double inst_forward(double T) const;    // f(0, T) = -d/dT log P(0, T)
    double forward_slope(double T) const;   // d/dT f(0, T)

private:
    double b0_, b1_, b2_, tau_;
};

// Hull-White one-factor short rate model (book Ch 11.3, eq. 11.32):
//   dr = lambda (theta(t) - r) dt + eta dW,
// with theta(t) chosen so that the model reproduces today's curve exactly.
class HullWhiteModel {
public:
    HullWhiteModel(double lambda, double eta, NelsonSiegelCurve curve);

    double theta(double t) const;   // book eq. 11.37
    double psi(double t) const;     // deterministic part of r(t), book eq. 11.41
    double r0() const;              // r(0) = f(0, 0)
    double mean_r(double t) const;  // E[r(t)] = psi(t)
    double var_r(double t) const;   // Var[r(t)] = eta^2 / (2 lambda) (1 - e^{-2 lambda t})
    // Zero-coupon bond price at time t for maturity T, given the short rate r(t) = r
    double zcb(double t, double T, double r) const;
    const NelsonSiegelCurve& curve() const { return curve_; }
    double lambda() const { return lambda_; }
    double eta() const { return eta_; }

private:
    double lambda_, eta_;
    NelsonSiegelCurve curve_;
};

struct HullWhitePaths {
    std::vector<double> r;         // n_paths * (n_steps + 1): short rate
    std::vector<double> integral;  // n_paths * (n_steps + 1): integral of r from 0 to t
};

// Euler scheme for the short rate, with a trapezoid rule for the integral of r,
// so that the discount factor along each path is exp(-integral).
HullWhitePaths simulate_hull_white(const HullWhiteModel& model, double T, int n_steps,
                                   int n_paths, std::uint64_t seed);

} // namespace quantlab
