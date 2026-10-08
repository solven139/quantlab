#include "quantlab/swap.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace quantlab {

std::vector<double> Swap::payment_dates() const {
    if (tau <= 0.0 || end <= start) throw std::invalid_argument("Swap: need tau > 0, end > start");
    const int m = static_cast<int>(std::lround((end - start) / tau));
    if (m < 1 || std::abs(start + m * tau - end) > 1e-9)
        throw std::invalid_argument("Swap: (end - start) must be a whole number of periods tau");
    std::vector<double> dates(m);
    for (int k = 0; k < m; ++k) dates[k] = start + (k + 1) * tau;
    return dates;
}

double swap_annuity(const NelsonSiegelCurve& curve, const Swap& swap) {
    double A = 0.0;
    for (double Tk : swap.payment_dates()) A += swap.tau * curve.discount(Tk);
    return A;
}

double swap_value_today(const NelsonSiegelCurve& curve, const Swap& swap) {
    // TODO 1: book eq. 12.12, for the payer (receive float, pay fixed)
    //   float leg = P(0, T_0) - P(0, T_m)            (telescoping sum of the Libor coupons)
    //   fixed leg = K * A(0),   A(0) = sum_k tau P(0, T_k)  (the annuity)
    // Ingredients:
    //   P(0, T)       ->  curve.discount(T)
    //   T_0 and T_m   ->  swap.start and swap.end
    //   K             ->  swap.K
    //   A(0)          ->  swap_annuity(curve, swap)   (already written above)
    const double float_leg = curve.discount(swap.start) - curve.discount(swap.end);   // placeholder: replace 0.0
    const double fixed_leg = swap.K * swap_annuity(curve, swap);   // placeholder: replace 0.0
    const double payer_value = swap.notional * (float_leg - fixed_leg);
    return swap.payer ? payer_value : -payer_value;
}

double par_swap_rate(const NelsonSiegelCurve& curve, const Swap& swap) {
    return (curve.discount(swap.start) - curve.discount(swap.end)) / swap_annuity(curve, swap);
}

namespace {

// Value at grid time t on one path, given r(t) and the bond price fixed at the last reset.
//   Fixed leg:  K * sum over payments T_k > t of tau P(t, T_k)
//   Float leg:  before the start:  P(t, T_0) - P(t, T_m)
//               inside period j (T_{j-1} <= t < T_j), the coupon was fixed at T_{j-1}:
//                 P(t, T_j) / P(T_{j-1}, T_j) - P(t, T_m)
double swap_value_on_path(const HullWhiteModel& m, const Swap& s, const std::vector<double>& dates,
                          double t, double r, double p_fix) {
    const double eps = 1e-9;
    if (t >= s.end - eps) return 0.0;      // every payment made: nothing left
    double fixed = 0.0;
    double first_after = -1.0;            // next payment date T_j > t
    for (double Tk : dates) {
        if (Tk > t + eps) {
            if (first_after < 0.0) first_after = Tk;
            fixed += s.tau * m.zcb(t, Tk, r);
        }
    }
    const double P_tm = m.zcb(t, s.end, r);
    double flt;
    if (t < s.start - eps) flt = m.zcb(t, s.start, r) - P_tm;
    else flt = m.zcb(t, first_after, r) / p_fix - P_tm;
    const double v = s.notional * (flt - s.K * fixed);
    return s.payer ? v : -v;
}

} // namespace

ExposureSimulation simulate_exposure(const HullWhiteModel& model, const std::vector<Swap>& trades,
                                     double horizon, int steps_per_year, int n_paths,
                                     std::uint64_t seed) {
    if (horizon <= 0.0 || steps_per_year < 1 || n_paths < 1 || trades.empty())
        throw std::invalid_argument("simulate_exposure: invalid input");
    const int n_steps = static_cast<int>(std::lround(horizon * steps_per_year));
    const double dt = horizon / n_steps, sq_dt = std::sqrt(dt);
    const double lambda = model.lambda(), eta = model.eta();
    const int n_times = n_steps + 1, n_trades = static_cast<int>(trades.size());

    // Every reset and payment date must sit on the grid
    std::vector<std::vector<double>> dates(n_trades);
    for (int a = 0; a < n_trades; ++a) {
        dates[a] = trades[a].payment_dates();
        auto on_grid = [&](double T) {
            return std::abs(T / dt - std::lround(T / dt)) < 1e-6;
        };
        if (!on_grid(trades[a].start) || !on_grid(trades[a].tau))
            throw std::invalid_argument("simulate_exposure: swap dates must lie on the time grid");
    }

    ExposureSimulation out;
    out.n_trades = n_trades; out.n_paths = n_paths; out.n_times = n_times;
    out.times.resize(n_times);
    for (int i = 0; i < n_times; ++i) out.times[i] = i * dt;
    out.values.assign(static_cast<std::size_t>(n_trades) * n_paths * n_times, 0.0);
    out.discount.resize(static_cast<std::size_t>(n_paths) * n_times);

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);
    std::vector<double> p_fix(n_trades);

    for (int p = 0; p < n_paths; ++p) {
        double r = model.r0(), I = 0.0;
        for (int a = 0; a < n_trades; ++a) p_fix[a] = 1.0;
        for (int i = 0; i < n_times; ++i) {
            const double t = out.times[i];
            if (i > 0) {   // Hull-White Euler step + trapezoid integral (as in Step 8)
                const double r_new = r + lambda * (model.theta(t - dt) - r) * dt + eta * sq_dt * Z(rng);
                I += 0.5 * (r + r_new) * dt;
                r = r_new;
            }
            out.discount[static_cast<std::size_t>(p) * n_times + i] = std::exp(-I);
            for (int a = 0; a < n_trades; ++a) {
                const Swap& s = trades[a];
                // At a reset date T_{j-1} (start, or a payment date before the end) fix the coupon
                const double k = (t - s.start) / s.tau;
                if (t >= s.start - 1e-9 && t < s.end - 1e-9 && std::abs(k - std::lround(k)) < 1e-6)
                    p_fix[a] = model.zcb(t, t + s.tau, r);
                out.values[(static_cast<std::size_t>(a) * n_paths + p) * n_times + i] =
                    swap_value_on_path(model, s, dates[a], t, r, p_fix[a]);
            }
        }
    }
    return out;
}

double cva(const std::vector<double>& times, const std::vector<double>& ee, double lgd,
           double hazard) {
    if (times.size() != ee.size()) throw std::invalid_argument("cva: times and ee differ in size");
    double total = 0.0;
    for (std::size_t k = 1; k < times.size(); ++k) {
        // TODO 2a: probability of default in (t_{k-1}, t_k]
        //   survival to time t is  e^{-h t},  so  PD = e^{-h t_{k-1}} - e^{-h t_k}
        //   h -> hazard,   t_{k-1} -> times[k - 1],   t_k -> times[k]
        const double pd = std::exp(-hazard * times[k - 1]) - std::exp(-hazard * times[k]);   // placeholder: replace 0.0

        // TODO 2b: add this period's contribution EE(t_k) * PD to the running total
        //   EE(t_k) -> ee[k]
        total += ee[k] * pd;            // placeholder: replace 0.0
    }
    return lgd * total;
}

} // namespace quantlab
