#include "quantlab/local_vol.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

#include "quantlab/cos.hpp"

namespace quantlab {

double dupire_local_vol(const Model& model, double S0, double K, double T,
                        double dK, double dT) {
    if (T <= dT || K <= dK || dK <= 0.0 || dT <= 0.0)
        throw std::invalid_argument("dupire_local_vol: need T > dT > 0 and K > dK > 0");

    // Call prices from the model: these play the role of "market" prices
    auto C = [&](double k, double t) {
        return cos_price(model, OptionType::Call, S0, k, t, 256, 12.0);
    };
    const double r = model.rate();

    // Central finite differences (book Ch 4.3: four option values per local vol).
    // C(k, t) gives the call price at strike k and maturity t.
    //
    // TODO 1a: slope in maturity   dC/dT   ~ [ C(K, T+dT) - C(K, T-dT) ] / (2 dT)
    const double dC_dT = (C(K, T + dT) - C(K, T - dT)) / (2.0 * dT);     // placeholder: replace 0.0
    // TODO 1b: slope in strike     dC/dK   ~ [ C(K+dK, T) - C(K-dK, T) ] / (2 dK)
    const double dC_dK = (C(K + dK, T) - C(K - dK, T)) / (2.0 * dK);    // placeholder: replace 0.0
    // TODO 1c: curvature in strike d2C/dK2 ~ [ C(K+dK, T) - 2 C(K, T) + C(K-dK, T) ] / dK^2
    const double d2C_dK2 = (C(K + dK, T) - 2.0 * C(K, T) + C(K - dK, T)) / (dK * dK);   // placeholder: replace 0.0

    // TODO 2: Dupire's formula (book eq. 4.49)
    //   numerator   = dC/dT + r K dC/dK
    //   denominator = 1/2 K^2 d2C/dK2
    const double numerator = dC_dT + r * K * dC_dK;     // placeholder: replace 0.0
    const double denominator = 0.5 * K * K * d2C_dK2;   // placeholder: replace 0.0

    // Far in the wings the density d2C/dK2 is ~0 and the ratio is rounding noise:
    // the book warns about exactly this (Ch 4.3). Below this (scale-free) threshold
    // we return 0, which tells the caller "unreliable, fill it from a neighbour".
    if (d2C_dK2 * S0 < 1e-4 || numerator <= 0.0)
        return 0.0;
    return std::sqrt(numerator / denominator);
}

LocalVolSurface::LocalVolSurface(const Model& model, double S0, std::vector<double> strikes,
                                 std::vector<double> maturities)
    : K_(std::move(strikes)), T_(std::move(maturities)) {
    if (K_.size() < 2 || T_.size() < 2)
        throw std::invalid_argument("LocalVolSurface: need at least 2 strikes and 2 maturities");
    vol_.resize(K_.size() * T_.size());
    for (std::size_t i = 0; i < T_.size(); ++i) {
        // Finite-difference steps shrink with maturity: for short T the distribution of
        // S(T) is narrow, so a coarse strike step would blur its curvature
        const double dT = std::min(0.01, 0.02 * T_[i]);
        const double dK = 0.01 * S0 * std::sqrt(std::min(T_[i], 1.0));
        for (std::size_t j = 0; j < K_.size(); ++j) {
            vol_[i * K_.size() + j] = dupire_local_vol(model, S0, K_[j], T_[i], dK, dT);
        }
        // Unreliable wing values (returned as 0): copy the nearest reliable neighbour,
        // working outwards from the strike closest to S0, where the density is largest.
        double* row = &vol_[i * K_.size()];
        const std::size_t n = K_.size();
        std::size_t atm = 0;
        for (std::size_t j = 1; j < n; ++j)
            if (std::abs(K_[j] - S0) < std::abs(K_[atm] - S0)) atm = j;
        for (std::size_t j = atm; j-- > 0;)
            if (row[j] <= 0.0 || row[j] > 3.0) row[j] = row[j + 1];
        for (std::size_t j = atm + 1; j < n; ++j)
            if (row[j] <= 0.0 || row[j] > 3.0) row[j] = row[j - 1];
    }
}

double LocalVolSurface::sigma(double S, double t) const {
    // Locate the cell, with flat extrapolation outside the grid
    auto locate = [](const std::vector<double>& g, double x, std::size_t& i, double& w) {
        if (x <= g.front()) { i = 0; w = 0.0; return; }
        if (x >= g.back())  { i = g.size() - 2; w = 1.0; return; }
        i = static_cast<std::size_t>(std::upper_bound(g.begin(), g.end(), x) - g.begin()) - 1;
        w = (x - g[i]) / (g[i + 1] - g[i]);
    };
    std::size_t iT, iK;
    double wT, wK;
    locate(T_, t, iT, wT);
    locate(K_, S, iK, wK);
    const std::size_t n = K_.size();
    const double v00 = vol_[iT * n + iK],       v01 = vol_[iT * n + iK + 1];
    const double v10 = vol_[(iT + 1) * n + iK], v11 = vol_[(iT + 1) * n + iK + 1];
    // Bilinear interpolation: first along K, then along T
    const double lower = (1.0 - wK) * v00 + wK * v01;
    const double upper = (1.0 - wK) * v10 + wK * v11;
    return (1.0 - wT) * lower + wT * upper;
}

namespace {
void check_times(const std::vector<double>& times) {
    if (times.empty() || times.front() <= 0.0 || !std::is_sorted(times.begin(), times.end()))
        throw std::invalid_argument("times must be positive and increasing");
}
}  // namespace

std::vector<double> local_vol_at_times(const LocalVolSurface& surface, double S0, double r,
                                       const std::vector<double>& times, int steps_per_year,
                                       int n_paths, std::uint64_t seed) {
    check_times(times);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);
    const std::size_t m = times.size();
    std::vector<double> out(static_cast<std::size_t>(n_paths) * m);

    for (int p = 0; p < n_paths; ++p) {
        double x = std::log(S0), t = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
            const int n = std::max(1, static_cast<int>(std::ceil((times[k] - t) * steps_per_year)));
            const double dt = (times[k] - t) / n;
            for (int i = 0; i < n; ++i) {
                const double s = surface.sigma(std::exp(x), t);
                x += (r - 0.5 * s * s) * dt + s * std::sqrt(dt) * Z(rng);
                t += dt;
            }
            t = times[k];
            out[static_cast<std::size_t>(p) * m + k] = std::exp(x);
        }
    }
    return out;
}

std::vector<double> heston_at_times(CirScheme scheme, double S0, double r, double kappa,
                                    double vbar, double gamma, double rho, double v0,
                                    const std::vector<double>& times, int steps_per_year,
                                    int n_paths, std::uint64_t seed) {
    check_times(times);
    const std::size_t m = times.size();
    std::vector<double> out(static_cast<std::size_t>(n_paths) * m);
    // Simulate each interval with heston_terminal-style AES steps, carrying (S, v) forward
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);
    for (int p = 0; p < n_paths; ++p) {
        double x = std::log(S0), v = v0, t = 0.0;
        for (std::size_t k = 0; k < m; ++k) {
            const int n = std::max(1, static_cast<int>(std::ceil((times[k] - t) * steps_per_year)));
            const double dt = (times[k] - t) / n;
            const double k0 = (r - rho * kappa * vbar / gamma) * dt;
            const double k1 = (rho * kappa / gamma - 0.5) * dt - rho / gamma;
            const double k2 = rho / gamma;
            const double k3 = (1.0 - rho * rho) * dt;
            for (int i = 0; i < n; ++i) {
                const double v_next = cir_step_samples(scheme, v, kappa, vbar, gamma, dt, 1,
                                                       rng())[0];
                x += k0 + k1 * v + k2 * v_next + std::sqrt(k3 * v) * Z(rng);
                v = v_next;
            }
            t = times[k];
            out[static_cast<std::size_t>(p) * m + k] = std::exp(x);
        }
    }
    return out;
}

} // namespace quantlab
