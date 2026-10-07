#include "quantlab/cos.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <stdexcept>

namespace quantlab {

namespace {

constexpr double PI = std::numbers::pi;

// chi_k(c, d): cosine coefficients of e^y on [c, d] within [a, b]  (book eq. 6.32)
double chi(int k, double a, double b, double c, double d) {
    const double w = k * PI / (b - a);
    const double num = std::cos(w * (d - a)) * std::exp(d) - std::cos(w * (c - a)) * std::exp(c)
                     + w * std::sin(w * (d - a)) * std::exp(d)
                     - w * std::sin(w * (c - a)) * std::exp(c);
    return num / (1.0 + w * w);
}

// psi_k(c, d): cosine coefficients of 1 on [c, d] within [a, b]  (book eq. 6.33)
double psi(int k, double a, double b, double c, double d) {
    if (k == 0)
        return d - c;
    const double w = k * PI / (b - a);
    return (std::sin(w * (d - a)) - std::sin(w * (c - a))) / w;
}

// H_k: cosine coefficients of the payoff in y = log(S(T)/K)  (book eqs. 6.34-6.35)
double payoff_coeff(int k, OptionType type, double a, double b, double K) {
    if (type == OptionType::Call) {
        if (b <= 0.0) return 0.0;              // the call can never pay inside [a, b]
        const double c = std::max(a, 0.0);
        return 2.0 / (b - a) * K * (chi(k, a, b, c, b) - psi(k, a, b, c, b));
    }
    if (a >= 0.0) return 0.0;                  // the put can never pay inside [a, b]
    const double d = std::min(b, 0.0);
    return 2.0 / (b - a) * K * (-chi(k, a, b, a, d) + psi(k, a, b, a, d));
}

// Integration range for y = x + X, centred on x + c1  (book eq. 6.44)
void truncation_range(const Model& model, double T, double x, double L, double& a, double& b) {
    const double width = L * std::sqrt(model.cumulant2(T) + std::sqrt(model.cumulant4(T)));
    a = x + model.cumulant1(T) - width;
    b = x + model.cumulant1(T) + width;
}

} // namespace

double cos_price(const Model& model, OptionType type, double S0, double K, double T,
                 int N, double L) {
    if (S0 <= 0.0 || K <= 0.0 || T <= 0.0 || N < 1 || L <= 0.0)
        throw std::invalid_argument("cos_price: invalid input");

    const double x = std::log(S0 / K);
    double a, b;
    truncation_range(model, T, x, L, a, b);

    const std::complex<double> i(0.0, 1.0);
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        const double u = k * PI / (b - a);
        const double term = std::real(model.char_fn(u, T) * std::exp(i * u * (x - a)))
                          * payoff_coeff(k, type, a, b, K);
        if (k == 0) {
            sum += 0.5 * term;
        } else {
            sum += term;
        }
    }
    return std::exp(-model.rate() * T) * sum;
}

std::vector<double> cos_density(const Model& model, double T, const std::vector<double>& xs,
                                int N, double L) {
    double a, b;
    truncation_range(model, T, 0.0, L, a, b);
    const std::complex<double> i(0.0, 1.0);

    std::vector<double> f(xs.size(), 0.0);
    for (int k = 0; k < N; ++k) {
        const double u = k * PI / (b - a);
        const double Fk = 2.0 / (b - a) * std::real(model.char_fn(u, T) * std::exp(-i * u * a));
        const double weight = (k == 0) ? 0.5 : 1.0;
        for (std::size_t j = 0; j < xs.size(); ++j)
            f[j] += weight * Fk * std::cos(u * (xs[j] - a));
    }
    return f;
}

} // namespace quantlab
