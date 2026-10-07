#pragma once

#include <cstdint>
#include <vector>

namespace quantlab {

// ---------------- GBM: Euler vs Milstein vs exact (book Ch 9.2) ----------------
// All three schemes are driven by the SAME Brownian increments, so their terminal values
// can be compared path by path (strong error) as well as on average (weak error).
struct GbmSchemeResult {
    std::vector<double> exact;     // S(T) from the exact solution
    std::vector<double> euler;     // S(T) from the Euler scheme
    std::vector<double> milstein;  // S(T) from the Milstein scheme
};
GbmSchemeResult gbm_schemes(double S0, double r, double sigma, double T,
                            int n_steps, int n_paths, std::uint64_t seed);

// ---------------- CIR variance: three ways to take ONE step of size dt ----------------
// dv = kappa (vbar - v) dt + gamma sqrt(v) dW, starting from v0. Returns n samples of v(dt).
enum class CirScheme { Euler, Exact, QE };
std::vector<double> cir_step_samples(CirScheme scheme, double v0, double kappa, double vbar,
                                     double gamma, double dt, int n, std::uint64_t seed);

// ---------------- Heston: "almost exact simulation" (book Ch 9.4.3, eq. 9.63) ----------------
// The variance is sampled with the chosen CIR scheme (Exact = noncentral chi-squared,
// QE = Andersen's Quadratic Exponential, Euler = truncated Euler), and log S is advanced
// with the AES formula x_{i+1} = x_i + k0 + k1 v_i + k2 v_{i+1} + sqrt(k3 v_i) Z.
// Returns the n_paths terminal values S(T).
std::vector<double> heston_terminal(CirScheme scheme, double S0, double r, double kappa,
                                    double vbar, double gamma, double rho, double v0,
                                    double T, int n_steps, int n_paths, std::uint64_t seed);

} // namespace quantlab
