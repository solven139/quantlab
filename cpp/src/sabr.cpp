#include "quantlab/sabr.hpp"

#include <cmath>
#include <stdexcept>

namespace quantlab {

namespace {

// z / x(z), with x(z) = log( (sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho) ).
// At z = 0 (K = F, or nu = 0) both are 0; the ratio tends to 1. Near 0 use the series
// z / x(z) = 1 - rho z / 2 + O(z^2), so the smile is continuous through the money.
double z_over_x(double z, double rho) {
    if (std::abs(z) < 1e-7) return 1.0 - 0.5 * rho * z;
    // Same x(z), rearranged to avoid cancellation for small z: the argument of the log is
    // 1 + y with  y = (z + (sqrt(1 - 2 rho z + z^2) - 1)) / (1 - rho),  and
    // sqrt(q) - 1 = (q - 1) / (sqrt(q) + 1). Then log(1 + y) = std::log1p(y).
    const double q = 1.0 - 2.0 * rho * z + z * z;
    const double y = (z + (z * z - 2.0 * rho * z) / (std::sqrt(q) + 1.0)) / (1.0 - rho);
    return z / std::log1p(y);
}

} // namespace

double sabr_implied_vol(double F, double K, double T, double alpha, double beta, double rho,
                        double nu) {
    if (F <= 0.0 || K <= 0.0 || T < 0.0 || alpha <= 0.0 || beta < 0.0 || beta > 1.0 ||
        rho <= -1.0 || rho >= 1.0 || nu < 0.0)
        throw std::invalid_argument("sabr_implied_vol: invalid input");

    const double om = 1.0 - beta;                 // 1 - beta
    const double logFK = std::log(F / K);
    const double fk_pow = std::pow(F * K, 0.5 * om);   // (F K)^{(1 - beta)/2}

    // Denominator expansion in log(F/K)
    const double l2 = logFK * logFK;
    const double denom = fk_pow * (1.0 + om * om / 24.0 * l2 + om * om * om * om / 1920.0 * l2 * l2);

    // z and the z/x(z) factor
    const double z = nu / alpha * fk_pow * logFK;
    const double zx = z_over_x(z, rho);

    // Time correction term
    const double corr = 1.0 + (om * om / 24.0 * alpha * alpha / (fk_pow * fk_pow)
                              + 0.25 * rho * beta * nu * alpha / fk_pow
                              + (2.0 - 3.0 * rho * rho) / 24.0 * nu * nu) * T;

    return alpha / denom * zx * corr;
}

std::vector<double> sabr_implied_vols(double F, const std::vector<double>& Ks, double T,
                                      double alpha, double beta, double rho, double nu) {
    std::vector<double> out(Ks.size());
    for (std::size_t j = 0; j < Ks.size(); ++j)
        out[j] = sabr_implied_vol(F, Ks[j], T, alpha, beta, rho, nu);
    return out;
}

} // namespace quantlab
