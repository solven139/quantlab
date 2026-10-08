#include "quantlab/hull_white.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace quantlab {

// ============================ Nelson-Siegel curve ============================

NelsonSiegelCurve::NelsonSiegelCurve(double b0, double b1, double b2, double tau)
    : b0_(b0), b1_(b1), b2_(b2), tau_(tau) {
    if (tau <= 0.0) throw std::invalid_argument("NelsonSiegelCurve: tau must be positive");
}

double NelsonSiegelCurve::zero_rate(double T) const {
    if (T < 1e-10) return b0_ + b1_;            // limit T -> 0
    const double x = T / tau_, e = std::exp(-x), g = (1.0 - e) / x;
    return b0_ + b1_ * g + b2_ * (g - e);
}

double NelsonSiegelCurve::discount(double T) const {
    return std::exp(-zero_rate(T) * T);
}

double NelsonSiegelCurve::inst_forward(double T) const {
    // f(0,T) = d/dT [ y(T) T ] = b0 + b1 e^-x + b2 x e^-x
    const double x = T / tau_, e = std::exp(-x);
    return b0_ + b1_ * e + b2_ * x * e;
}

double NelsonSiegelCurve::forward_slope(double T) const {
    const double x = T / tau_, e = std::exp(-x);
    return (-b1_ * e + b2_ * (e - x * e)) / tau_;
}

// ============================ Hull-White model ============================

HullWhiteModel::HullWhiteModel(double lambda, double eta, NelsonSiegelCurve curve)
    : lambda_(lambda), eta_(eta), curve_(curve) {
    if (lambda <= 0.0 || eta <= 0.0)
        throw std::invalid_argument("HullWhiteModel: lambda and eta must be positive");
}

double HullWhiteModel::theta(double t) const {
    // TODO 1: book eq. 11.37
    //   theta(t) = f(0,t) + (1/lambda) * df(0,t)/dt + eta^2 / (2 lambda^2) * (1 - e^{-2 lambda t})
    // Ingredients:
    //   f(0,t)      ->  curve_.inst_forward(t)
    //   df(0,t)/dt  ->  curve_.forward_slope(t)
    //   lambda, eta ->  the members lambda_ and eta_   (note the underscore!)
    //   e^{x}       ->  std::exp(x)
    const double l = lambda_;
    return curve_.inst_forward(t) + curve_.forward_slope(t) / l
         + eta_ * eta_ / (2.0 * l * l) * (1.0 - std::exp(-2.0 * l * t));   // placeholder: replace 0.0 with the formula
}

double HullWhiteModel::r0() const { return curve_.inst_forward(0.0); }

double HullWhiteModel::psi(double t) const {
    // book eq. 11.41, written as f(0,t) + eta^2/(2 lambda^2) (1 - e^{-lambda t})^2
    const double l = lambda_, b = 1.0 - std::exp(-l * t);
    return curve_.inst_forward(t) + eta_ * eta_ / (2.0 * l * l) * b * b;
}

double HullWhiteModel::mean_r(double t) const { return psi(t); }

double HullWhiteModel::var_r(double t) const {
    return eta_ * eta_ / (2.0 * lambda_) * (1.0 - std::exp(-2.0 * lambda_ * t));
}

double HullWhiteModel::zcb(double t, double T, double r) const {
    // Book Lemma 11.3.1 with u = 0, combined with the fit to today's curve:
    //   P(t,T) = P(0,T)/P(0,t) * exp( A(tau) - A(T) + A(t) + B(tau) (r - psi(t)) )
    const double l = lambda_, e2 = eta_ * eta_;
    auto A = [&](double s) {
        return e2 / (2.0 * l * l * l)
             * (l * s - 2.0 * (1.0 - std::exp(-l * s)) + 0.5 * (1.0 - std::exp(-2.0 * l * s)));
    };
    const double tau = T - t;
    const double B = -(1.0 - std::exp(-l * tau)) / l;
    return curve_.discount(T) / curve_.discount(t)
         * std::exp(A(tau) - A(T) + A(t) + B * (r - psi(t)));
}

// ============================ Simulation ============================

HullWhitePaths simulate_hull_white(const HullWhiteModel& model, double T, int n_steps,
                                   int n_paths, std::uint64_t seed) {
    if (T <= 0.0 || n_steps < 1 || n_paths < 1)
        throw std::invalid_argument("simulate_hull_white: invalid input");
    const double dt = T / n_steps, sq_dt = std::sqrt(dt);
    const double lambda = model.lambda(), eta = model.eta();

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);
    const int n_cols = n_steps + 1;
    HullWhitePaths out;
    out.r.resize(static_cast<std::size_t>(n_paths) * n_cols);
    out.integral.resize(static_cast<std::size_t>(n_paths) * n_cols);

    for (int p = 0; p < n_paths; ++p) {
        double* r = &out.r[static_cast<std::size_t>(p) * n_cols];
        double* I = &out.integral[static_cast<std::size_t>(p) * n_cols];
        r[0] = model.r0();
        I[0] = 0.0;
        for (int i = 1; i < n_cols; ++i) {
            const double t = (i - 1) * dt;
            // TODO 2a: Euler step of book eq. 11.32,  dr = lambda (theta(t) - r) dt + eta dW
            //   new r = old r + lambda * (theta(t) - old r) * dt + eta * sqrt(dt) * Z
            //   old r -> r[i - 1],  theta(t) -> model.theta(t),  sqrt(dt) -> sq_dt,  Z -> Z(rng)
            r[i] = r[i - 1] + lambda * (model.theta(t) - r[i - 1]) * dt + eta * sq_dt * Z(rng);   // placeholder: add the drift and the random term

            // TODO 2b: trapezoid rule for the running integral of r
            //   new I = old I + (average of old r and new r) * dt
            //   old I -> I[i - 1],  average -> 0.5 * (r[i - 1] + r[i])
            I[i] = I[i - 1] + 0.5 * (r[i - 1] + r[i]) * dt;       // placeholder: replace 0.0
        }
    }
    return out;
}

} // namespace quantlab
