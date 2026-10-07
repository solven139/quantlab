#include "quantlab/models.hpp"

#include <cmath>
#include <stdexcept>

namespace quantlab {

BlackScholesModel::BlackScholesModel(double r, double sigma) : r_(r), sigma_(sigma) {
    if (sigma <= 0.0)
        throw std::invalid_argument("BlackScholesModel: sigma must be positive");
}

std::complex<double> BlackScholesModel::char_fn(double u, double T) const {
    // phi(u) = exp( i u (r - sigma^2/2) T  -  sigma^2 u^2 T / 2 )
    const std::complex<double> i(0.0, 1.0);
    const double mu = (r_ - 0.5 * sigma_ * sigma_) * T;
    return std::exp(i * u * mu - 0.5 * sigma_ * sigma_ * u * u * T);
}

double BlackScholesModel::cumulant1(double T) const { return (r_ - 0.5 * sigma_ * sigma_) * T; }
double BlackScholesModel::cumulant2(double T) const { return sigma_ * sigma_ * T; }
double BlackScholesModel::cumulant4(double) const { return 0.0; }

// ============================ Merton ============================

MertonModel::MertonModel(double r, double sigma, double xi, double mu_j, double sigma_j)
    : r_(r), sigma_(sigma), xi_(xi), mu_j_(mu_j), sigma_j_(sigma_j) {
    if (sigma <= 0.0 || xi < 0.0 || sigma_j < 0.0)
        throw std::invalid_argument("MertonModel: need sigma > 0, xi >= 0, sigma_j >= 0");
}

double MertonModel::drift_correction() const {
    // omega = xi * (E[e^J] - 1), with E[e^J] = exp(mu_j + sigma_j^2 / 2) for normal J
    return xi_ * (std::exp(mu_j_ + 0.5 * sigma_j_ * sigma_j_) - 1.0);
}

std::complex<double> MertonModel::char_fn(double u, double T) const {
    // phi(u) = exp( i u (r - omega - sigma^2/2) T - sigma^2 u^2 T / 2
    //               + xi T (E[e^{iuJ}] - 1) ),   E[e^{iuJ}] = exp(i u mu_j - sigma_j^2 u^2 / 2)
    const std::complex<double> i(0.0, 1.0);
    const double drift = (r_ - drift_correction() - 0.5 * sigma_ * sigma_) * T;
    const std::complex<double> jump_cf = std::exp(i * u * mu_j_ - 0.5 * sigma_j_ * sigma_j_ * u * u);
    return std::exp(i * u * drift - 0.5 * sigma_ * sigma_ * u * u * T + xi_ * T * (jump_cf - 1.0));
}

// For a compound Poisson part, the n-th cumulant is xi * T * E[J^n]
double MertonModel::cumulant1(double T) const {
    return (r_ - drift_correction() - 0.5 * sigma_ * sigma_) * T + xi_ * T * mu_j_;
}
double MertonModel::cumulant2(double T) const {
    return sigma_ * sigma_ * T + xi_ * T * (mu_j_ * mu_j_ + sigma_j_ * sigma_j_);
}
double MertonModel::cumulant4(double T) const {
    const double m = mu_j_, s2 = sigma_j_ * sigma_j_;
    return xi_ * T * (m * m * m * m + 6.0 * m * m * s2 + 3.0 * s2 * s2);
}

// ============================== Kou ==============================

KouModel::KouModel(double r, double sigma, double xi, double p, double eta1, double eta2)
    : r_(r), sigma_(sigma), xi_(xi), p_(p), eta1_(eta1), eta2_(eta2) {
    if (sigma <= 0.0 || xi < 0.0 || p < 0.0 || p > 1.0 || eta1 <= 1.0 || eta2 <= 0.0)
        throw std::invalid_argument("KouModel: need sigma > 0, xi >= 0, 0 <= p <= 1, eta1 > 1, eta2 > 0");
}

double KouModel::drift_correction() const {
    // E[e^J] = p eta1/(eta1 - 1) + (1-p) eta2/(eta2 + 1)   (finite only if eta1 > 1)
    const double mean_ej = p_ * eta1_ / (eta1_ - 1.0) + (1.0 - p_) * eta2_ / (eta2_ + 1.0);
    return xi_ * (mean_ej - 1.0);
}

std::complex<double> KouModel::char_fn(double u, double T) const {
    const std::complex<double> i(0.0, 1.0);
    const double drift = (r_ - drift_correction() - 0.5 * sigma_ * sigma_) * T;
    // E[e^{iuJ}] for the double-exponential jump size
    const std::complex<double> jump_cf = p_ * eta1_ / (eta1_ - i * u)
                                       + (1.0 - p_) * eta2_ / (eta2_ + i * u);
    return std::exp(i * u * drift - 0.5 * sigma_ * sigma_ * u * u * T + xi_ * T * (jump_cf - 1.0));
}

double KouModel::cumulant1(double T) const {
    const double ej = p_ / eta1_ - (1.0 - p_) / eta2_;
    return (r_ - drift_correction() - 0.5 * sigma_ * sigma_) * T + xi_ * T * ej;
}
double KouModel::cumulant2(double T) const {
    const double ej2 = 2.0 * p_ / (eta1_ * eta1_) + 2.0 * (1.0 - p_) / (eta2_ * eta2_);
    return sigma_ * sigma_ * T + xi_ * T * ej2;
}
double KouModel::cumulant4(double T) const {
    const double ej4 = 24.0 * p_ / std::pow(eta1_, 4) + 24.0 * (1.0 - p_) / std::pow(eta2_, 4);
    return xi_ * T * ej4;
}

// ========================= Variance Gamma =========================

VarianceGammaModel::VarianceGammaModel(double r, double sigma, double theta, double beta)
    : r_(r), sigma_(sigma), theta_(theta), beta_(beta) {
    if (sigma <= 0.0 || beta <= 0.0)
        throw std::invalid_argument("VarianceGammaModel: need sigma > 0, beta > 0");
    if (1.0 - beta * theta - 0.5 * beta * sigma * sigma <= 0.0)
        throw std::invalid_argument("VarianceGammaModel: need theta + sigma^2/2 < 1/beta");
}

double VarianceGammaModel::drift_correction() const {
    // book eq. 5.59: omega = (1/beta) log(1 - beta theta - beta sigma^2 / 2)
    return std::log(1.0 - beta_ * theta_ - 0.5 * beta_ * sigma_ * sigma_) / beta_;
}

std::complex<double> VarianceGammaModel::char_fn(double u, double T) const {
    // phi(u) = exp(i u (r + omega) T) * (1 - i u theta beta + beta sigma^2 u^2 / 2)^(-T/beta)
    const std::complex<double> i(0.0, 1.0);
    const std::complex<double> base = 1.0 - i * u * theta_ * beta_
                                    + 0.5 * beta_ * sigma_ * sigma_ * u * u;
    return std::exp(i * u * (r_ + drift_correction()) * T) * std::pow(base, -T / beta_);
}

double VarianceGammaModel::cumulant1(double T) const {
    return (r_ + drift_correction() + theta_) * T;
}
double VarianceGammaModel::cumulant2(double T) const {
    return (sigma_ * sigma_ + beta_ * theta_ * theta_) * T;
}
double VarianceGammaModel::cumulant4(double T) const {
    const double s2 = sigma_ * sigma_, th2 = theta_ * theta_, b = beta_;
    return 3.0 * (s2 * s2 * b + 2.0 * th2 * th2 * b * b * b + 4.0 * s2 * th2 * b * b) * T;
}

// ============================= Heston =============================

HestonModel::HestonModel(double r, double kappa, double vbar, double gamma, double rho, double v0)
    : r_(r), kappa_(kappa), vbar_(vbar), gamma_(gamma), rho_(rho), v0_(v0) {
    if (kappa <= 0.0 || vbar <= 0.0 || gamma <= 0.0 || v0 < 0.0 || rho <= -1.0 || rho >= 1.0)
        throw std::invalid_argument("HestonModel: need kappa, vbar, gamma > 0, v0 >= 0, -1 < rho < 1");
}

bool HestonModel::feller_satisfied() const {
    return 2.0 * kappa_ * vbar_ >= gamma_ * gamma_;
}

std::complex<double> HestonModel::char_fn(double u, double T) const {
    // TODO 1: Heston characteristic function, book Lemma 8.3.2 (without the discount
    //         factor, because our Model::char_fn is for log(S(T)/S0)). Six intermediate
    //         complex numbers, then the result. Members: r_, kappa_, vbar_, gamma_, rho_, v0_.
    const std::complex<double> i(0.0, 1.0);
    const std::complex<double> a = kappa_ - gamma_ * rho_ * i * u;
    const std::complex<double> D1 = std::sqrt(a * a + (u * u + i * u) * gamma_ * gamma_);
    const std::complex<double> g = (a - D1) / (a + D1);
    const std::complex<double> e = std::exp(-D1 * T);
    const std::complex<double> C = (1.0 - e) / (gamma_ * gamma_ * (1.0 - g * e)) * (a - D1);
    const std::complex<double> A = r_ * i * u * T
        + kappa_ * vbar_ / (gamma_ * gamma_) * ((a - D1) * T - 2.0 * std::log((1.0 - g * e) / (1.0 - g)));
    return std::exp(A + C * v0_);
}

double HestonModel::cumulant1(double T) const {
    // E[log(S(T)/S0)] = rT - (1/2) * integral of E[v(t)] dt
    const double ek = std::exp(-kappa_ * T);
    return r_ * T - 0.5 * (vbar_ * T + (v0_ - vbar_) * (1.0 - ek) / kappa_);
}

double HestonModel::cumulant2(double T) const {
    // Variance of X read straight off the characteristic function:
    //   log phi(u) = i u c1 - u^2 c2 / 2 + ...   =>   c2 = -d^2/du^2 log phi(u) at u = 0.
    // (The closed form often quoted from the literature has a typo: it is ~2% off.)
    const double h = 1e-2;   // not too small: phi is computed with round-off, and we divide by h^2
    const double lp = std::real(std::log(char_fn(h, T)));    // log|phi(h)|, phi(-h) is its conjugate
    return -2.0 * lp / (h * h);                               // central difference, log phi(0) = 0
}

double HestonModel::cumulant4(double) const {
    // Not in closed form here; the COS range then relies on cumulant2 with a wider L (12)
    return 0.0;
}

} // namespace quantlab
