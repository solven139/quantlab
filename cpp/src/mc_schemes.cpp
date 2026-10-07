#include "quantlab/mc_schemes.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

namespace quantlab {

// ============================ GBM schemes ============================

GbmSchemeResult gbm_schemes(double S0, double r, double sigma, double T,
                            int n_steps, int n_paths, std::uint64_t seed) {
    if (S0 <= 0.0 || T <= 0.0 || n_steps < 1 || n_paths < 1)
        throw std::invalid_argument("gbm_schemes: invalid input");

    const double dt = T / n_steps;
    const double sq_dt = std::sqrt(dt);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    GbmSchemeResult out;
    out.exact.resize(n_paths);
    out.euler.resize(n_paths);
    out.milstein.resize(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double s_eu = S0, s_mi = S0, W = 0.0;
        for (int i = 0; i < n_steps; ++i) {
            const double dW = sq_dt * Z(rng);   // one Brownian increment, shared by all schemes
            W += dW;
            // Euler (book eq. 9.13): S_{i+1} = S_i + r S_i dt + sigma S_i dW
            s_eu += r * s_eu * dt + sigma * s_eu * dW;
            s_mi += r * s_mi * dt + sigma * s_mi * dW + 0.5 * sigma * sigma * s_mi * (dW * dW - dt);
        }
        out.exact[p] = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * W);
        out.euler[p] = s_eu;
        out.milstein[p] = s_mi;
    }
    return out;
}

// ============================ CIR schemes ============================

namespace {

// One step of each CIR scheme, given the current value v and the random engine.
double cir_step_euler(double v, double kappa, double vbar, double gamma, double dt,
                      std::mt19937_64& rng) {
    std::normal_distribution<double> Z(0.0, 1.0);
    const double vp = std::max(v, 0.0);  // full truncation
    return std::max(v + kappa * (vbar - vp) * dt + gamma * std::sqrt(vp * dt) * Z(rng), 0.0);
}

double cir_step_exact(double v, double kappa, double vbar, double gamma, double dt,
                      std::mt19937_64& rng) {
    // v(t+dt) = c * chi^2(delta, lambda)   (book eq. 9.30)
    const double ek = std::exp(-kappa * dt);
    const double c = gamma * gamma * (1.0 - ek) / (4.0 * kappa);
    const double delta = 4.0 * kappa * vbar / (gamma * gamma);
    const double lambda = 4.0 * kappa * ek * v / (gamma * gamma * (1.0 - ek));
    // Noncentral chi^2 as a Poisson mixture of central chi^2:
    //   N ~ Poisson(lambda / 2),  chi^2(delta + 2N) = Gamma(shape (delta + 2N)/2, scale 2)
    std::poisson_distribution<int> pois(0.5 * lambda);
    const int N = pois(rng);
    std::gamma_distribution<double> gam(0.5 * delta + N, 2.0);
    return c * gam(rng);
}

double cir_step_qe(double v, double kappa, double vbar, double gamma, double dt,
                   std::mt19937_64& rng) {
    // Andersen's Quadratic Exponential scheme (book Ch 9.3.4)
    // Exact conditional mean and variance of v(t+dt) given v(t) = v   (book eq. 9.32)
    const double ek = std::exp(-kappa * dt);
    const double m = vbar + (v - vbar) * ek;
    const double s2 = v * gamma * gamma * ek * (1.0 - ek) / kappa
                    + vbar * gamma * gamma * (1.0 - ek) * (1.0 - ek) / (2.0 * kappa);
    const double psi = s2 / (m * m);
    const double psi_switch = 1.5;  // a* in the book: any value in [1, 2] works

    if (psi <= psi_switch) {
        // Quadratic branch: v = a (b + Z)^2
        const double b2 = 2.0 / psi - 1.0 + std::sqrt(2.0 / psi) * std::sqrt(2.0 / psi - 1.0);
        const double a = m / (1.0 + b2);
        std::normal_distribution<double> Z(0.0, 1.0);
        const double z = std::sqrt(b2) + Z(rng);
        return a * z * z;
    }
    // Exponential branch: v = 0 with probability c, otherwise exponential
    const double c = (psi - 1.0) / (psi + 1.0);
    const double d = (1.0 - c) / m;
    std::uniform_real_distribution<double> U(0.0, 1.0);
    const double u = U(rng);
    return (u <= c) ? 0.0 : std::log((1.0 - c) / (1.0 - u)) / d;
}

double cir_step(CirScheme scheme, double v, double kappa, double vbar, double gamma, double dt,
                std::mt19937_64& rng) {
    switch (scheme) {
        case CirScheme::Euler: return cir_step_euler(v, kappa, vbar, gamma, dt, rng);
        case CirScheme::Exact: return cir_step_exact(v, kappa, vbar, gamma, dt, rng);
        case CirScheme::QE:    return cir_step_qe(v, kappa, vbar, gamma, dt, rng);
    }
    throw std::invalid_argument("cir_step: unknown scheme");
}

} // namespace

std::vector<double> cir_step_samples(CirScheme scheme, double v0, double kappa, double vbar,
                                     double gamma, double dt, int n, std::uint64_t seed) {
    if (v0 < 0.0 || kappa <= 0.0 || vbar <= 0.0 || gamma <= 0.0 || dt <= 0.0 || n < 1)
        throw std::invalid_argument("cir_step_samples: invalid input");
    std::mt19937_64 rng(seed);
    std::vector<double> out(n);
    for (int j = 0; j < n; ++j)
        out[j] = cir_step(scheme, v0, kappa, vbar, gamma, dt, rng);
    return out;
}

// ============================ Heston AES ============================

std::vector<double> heston_terminal(CirScheme scheme, double S0, double r, double kappa,
                                    double vbar, double gamma, double rho, double v0,
                                    double T, int n_steps, int n_paths, std::uint64_t seed) {
    if (S0 <= 0.0 || T <= 0.0 || n_steps < 1 || n_paths < 1)
        throw std::invalid_argument("heston_terminal: invalid input");

    const double dt = T / n_steps;
    // Constants of book eq. 9.63
    const double k0 = (r - rho * kappa * vbar / gamma) * dt;
    const double k1 = (rho * kappa / gamma - 0.5) * dt - rho / gamma;
    const double k2 = rho / gamma;
    const double k3 = (1.0 - rho * rho) * dt;

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);
    std::vector<double> out(n_paths);

    for (int p = 0; p < n_paths; ++p) {
        double x = std::log(S0);
        double v = v0;
        for (int i = 0; i < n_steps; ++i) {
            const double v_next = cir_step(scheme, v, kappa, vbar, gamma, dt, rng);
            x += k0 + k1 * v + k2 * v_next + std::sqrt(k3 * v) * Z(rng);
            v = v_next;
        }
        out[p] = std::exp(x);
    }
    return out;
}

} // namespace quantlab
