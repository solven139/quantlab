#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <vector>
#include "quantlab/black_scholes.hpp"
#include "quantlab/gbm.hpp"
#include "quantlab/hedging.hpp"
#include "quantlab/implied_vol.hpp"
#include "quantlab/models.hpp"
#include "quantlab/cos.hpp"
#include "quantlab/jump_paths.hpp"
#include "quantlab/heston_paths.hpp"
#include "quantlab/mc_schemes.hpp"
#include "quantlab/local_vol.hpp"
#include "quantlab/hull_white.hpp"
#include "quantlab/swap.hpp"

namespace py = pybind11;
using namespace quantlab;

PYBIND11_MODULE(_core, m) {
    m.doc() = "quantlab C++ pricing engine";

    py::enum_<OptionType>(m, "OptionType")
        .value("Call", OptionType::Call)
        .value("Put",  OptionType::Put);

    m.def("norm_cdf", &norm_cdf, py::arg("x"));
    m.def("bs_price", &bs_price,
          py::arg("type"), py::arg("S0"), py::arg("K"),
          py::arg("T"), py::arg("r"), py::arg("sigma"),
          "Black-Scholes European option price");

    m.def("norm_pdf", &norm_pdf, py::arg("x"));
    m.def("bs_delta", &bs_delta,
          py::arg("type"), py::arg("S0"), py::arg("K"),
          py::arg("T"), py::arg("r"), py::arg("sigma"));
    m.def("bs_gamma", &bs_gamma,
          py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"));
    m.def("bs_vega", &bs_vega,
          py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r"), py::arg("sigma"));

    m.def("simulate_gbm",
        [](double S0, double r, double sigma, double T, int n_steps, int n_paths, std::uint64_t seed) {
            std::vector<double> flat = simulate_gbm(S0, r, sigma, T, n_steps, n_paths, seed);
            // Copy the flat C++ vector into a 2-D NumPy array of shape (n_paths, n_steps + 1)
            py::array_t<double> out({n_paths, n_steps + 1});
            std::copy(flat.begin(), flat.end(), out.mutable_data());
            return out;
        },
        py::arg("S0"), py::arg("r"), py::arg("sigma"), py::arg("T"),
        py::arg("n_steps"), py::arg("n_paths"), py::arg("seed") = 42,
        "Simulate GBM paths; returns array of shape (n_paths, n_steps + 1)");

    m.def("delta_hedge_pnl",
        [](double S0, double K, double T, double r, double sigma_true, double sigma_hedge,
           int n_rebalance, int n_paths, std::uint64_t seed) {
            std::vector<double> v = delta_hedge_pnl(S0, K, T, r, sigma_true, sigma_hedge,
                                                    n_rebalance, n_paths, seed);
            py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
            std::copy(v.begin(), v.end(), out.mutable_data());
            return out;
        },
        py::arg("S0"), py::arg("K"), py::arg("T"), py::arg("r"),
        py::arg("sigma_true"), py::arg("sigma_hedge"),
        py::arg("n_rebalance"), py::arg("n_paths"), py::arg("seed") = 42,
        "Final P&L of delta-hedging a sold call, one value per path");

    m.def("implied_vol", &implied_vol,
          py::arg("type"), py::arg("price"), py::arg("S0"), py::arg("K"),
          py::arg("T"), py::arg("r"),
          "Black-Scholes implied volatility (Newton-Raphson with bisection fallback)");
    m.def("implied_vol_iterates",
        [](OptionType type, double price, double S0, double K, double T, double r) {
            std::vector<double> v = implied_vol_iterates(type, price, S0, K, T, r);
            py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
            std::copy(v.begin(), v.end(), out.mutable_data());
            return out;
        },
        py::arg("type"), py::arg("price"), py::arg("S0"), py::arg("K"),
        py::arg("T"), py::arg("r"),
        "Every iterate of the implied-vol solver, first guess to final answer");

    // ---- Models (Step 3+) ----
    py::class_<Model>(m, "Model")
        .def("rate", &Model::rate)
        .def("char_fn", &Model::char_fn, py::arg("u"), py::arg("T"))
        .def("cumulant1", &Model::cumulant1, py::arg("T"))
        .def("cumulant2", &Model::cumulant2, py::arg("T"))
        .def("cumulant4", &Model::cumulant4, py::arg("T"));

    py::class_<BlackScholesModel, Model>(m, "BlackScholesModel")
        .def(py::init<double, double>(), py::arg("r"), py::arg("sigma"))
        .def("sigma", &BlackScholesModel::sigma);

    py::class_<MertonModel, Model>(m, "MertonModel")
        .def(py::init<double, double, double, double, double>(),
             py::arg("r"), py::arg("sigma"), py::arg("xi"), py::arg("mu_j"), py::arg("sigma_j"))
        .def("drift_correction", &MertonModel::drift_correction);

    py::class_<KouModel, Model>(m, "KouModel")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("r"), py::arg("sigma"), py::arg("xi"), py::arg("p"),
             py::arg("eta1"), py::arg("eta2"))
        .def("drift_correction", &KouModel::drift_correction);

    py::class_<VarianceGammaModel, Model>(m, "VarianceGammaModel")
        .def(py::init<double, double, double, double>(),
             py::arg("r"), py::arg("sigma"), py::arg("theta"), py::arg("beta"))
        .def("drift_correction", &VarianceGammaModel::drift_correction);

    py::class_<HestonModel, Model>(m, "HestonModel")
        .def(py::init<double, double, double, double, double, double>(),
             py::arg("r"), py::arg("kappa"), py::arg("vbar"), py::arg("gamma"),
             py::arg("rho"), py::arg("v0"))
        .def("feller_satisfied", &HestonModel::feller_satisfied)
        .def("kappa", &HestonModel::kappa)
        .def("vbar", &HestonModel::vbar)
        .def("gamma", &HestonModel::gamma)
        .def("rho", &HestonModel::rho)
        .def("v0", &HestonModel::v0);

    // ---- COS method ----
    m.def("cos_price", &cos_price,
          py::arg("model"), py::arg("type"), py::arg("S0"), py::arg("K"), py::arg("T"),
          py::arg("N") = 128, py::arg("L") = 8.0,
          "European option price by the COS method for any Model");
    m.def("cos_density", &cos_density,
          py::arg("model"), py::arg("T"), py::arg("xs"), py::arg("N") = 128, py::arg("L") = 8.0,
          "Density of log(S(T)/S0) recovered from the characteristic function");

    m.def("simulate_merton",
        [](double S0, double r, double sigma, double xi, double mu_j, double sigma_j,
           double T, int n_steps, int n_paths, std::uint64_t seed) {
            std::vector<double> flat = simulate_merton(S0, r, sigma, xi, mu_j, sigma_j,
                                                       T, n_steps, n_paths, seed);
            py::array_t<double> out({n_paths, n_steps + 1});
            std::copy(flat.begin(), flat.end(), out.mutable_data());
            return out;
        },
        py::arg("S0"), py::arg("r"), py::arg("sigma"), py::arg("xi"), py::arg("mu_j"),
        py::arg("sigma_j"), py::arg("T"), py::arg("n_steps"), py::arg("n_paths"),
        py::arg("seed") = 42,
        "Simulate Merton jump-diffusion paths; array of shape (n_paths, n_steps + 1)");

    m.def("simulate_heston",
        [](double S0, double r, double kappa, double vbar, double gamma, double rho, double v0,
           double T, int n_steps, int n_paths, std::uint64_t seed) {
            HestonPaths res = simulate_heston(S0, r, kappa, vbar, gamma, rho, v0,
                                              T, n_steps, n_paths, seed);
            py::array_t<double> S({n_paths, n_steps + 1});
            py::array_t<double> v({n_paths, n_steps + 1});
            std::copy(res.S.begin(), res.S.end(), S.mutable_data());
            std::copy(res.v.begin(), res.v.end(), v.mutable_data());
            return py::make_tuple(S, v, res.zero_hits);
        },
        py::arg("S0"), py::arg("r"), py::arg("kappa"), py::arg("vbar"), py::arg("gamma"),
        py::arg("rho"), py::arg("v0"), py::arg("T"), py::arg("n_steps"), py::arg("n_paths"),
        py::arg("seed") = 42,
        "Euler (full truncation) Heston paths; returns (S, v, zero_hits)");

    // ---- Monte Carlo lab (Step 6) ----
    py::enum_<CirScheme>(m, "CirScheme")
        .value("Euler", CirScheme::Euler)
        .value("Exact", CirScheme::Exact)
        .value("QE", CirScheme::QE);

    auto to_numpy = [](const std::vector<double>& v) {
        py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
        std::copy(v.begin(), v.end(), out.mutable_data());
        return out;
    };

    m.def("gbm_schemes",
        [to_numpy](double S0, double r, double sigma, double T, int n_steps, int n_paths,
                   std::uint64_t seed) {
            GbmSchemeResult res = gbm_schemes(S0, r, sigma, T, n_steps, n_paths, seed);
            return py::make_tuple(to_numpy(res.exact), to_numpy(res.euler),
                                  to_numpy(res.milstein));
        },
        py::arg("S0"), py::arg("r"), py::arg("sigma"), py::arg("T"), py::arg("n_steps"),
        py::arg("n_paths"), py::arg("seed") = 42,
        "Terminal S(T) under exact, Euler and Milstein with the same Brownian paths");

    m.def("cir_step_samples",
        [to_numpy](CirScheme scheme, double v0, double kappa, double vbar, double gamma,
                   double dt, int n, std::uint64_t seed) {
            return to_numpy(cir_step_samples(scheme, v0, kappa, vbar, gamma, dt, n, seed));
        },
        py::arg("scheme"), py::arg("v0"), py::arg("kappa"), py::arg("vbar"), py::arg("gamma"),
        py::arg("dt"), py::arg("n"), py::arg("seed") = 42,
        "n samples of v(dt) for a CIR process started at v0, using the chosen scheme");

    m.def("heston_terminal",
        [to_numpy](CirScheme scheme, double S0, double r, double kappa, double vbar,
                   double gamma, double rho, double v0, double T, int n_steps, int n_paths,
                   std::uint64_t seed) {
            return to_numpy(heston_terminal(scheme, S0, r, kappa, vbar, gamma, rho, v0, T,
                                            n_steps, n_paths, seed));
        },
        py::arg("scheme"), py::arg("S0"), py::arg("r"), py::arg("kappa"), py::arg("vbar"),
        py::arg("gamma"), py::arg("rho"), py::arg("v0"), py::arg("T"), py::arg("n_steps"),
        py::arg("n_paths"), py::arg("seed") = 42,
        "Heston S(T) by almost-exact simulation with the chosen variance scheme");

    // ---- Local volatility (Step 7) ----
    m.def("dupire_local_vol", &dupire_local_vol,
          py::arg("model"), py::arg("S0"), py::arg("K"), py::arg("T"),
          py::arg("dK"), py::arg("dT"),
          "Dupire local volatility from a model's call prices (finite differences)");

    py::class_<LocalVolSurface>(m, "LocalVolSurface")
        .def(py::init<const Model&, double, std::vector<double>, std::vector<double>>(),
             py::arg("model"), py::arg("S0"), py::arg("strikes"), py::arg("maturities"))
        .def("sigma", &LocalVolSurface::sigma, py::arg("S"), py::arg("t"))
        .def("strikes", &LocalVolSurface::strikes)
        .def("maturities", &LocalVolSurface::maturities)
        .def("values", [](const LocalVolSurface& s) {
            py::array_t<double> out({s.maturities().size(), s.strikes().size()});
            std::copy(s.values().begin(), s.values().end(), out.mutable_data());
            return out;
        });

    m.def("local_vol_at_times",
        [](const LocalVolSurface& surface, double S0, double r, std::vector<double> times,
           int steps_per_year, int n_paths, std::uint64_t seed) {
            std::vector<double> v = local_vol_at_times(surface, S0, r, times, steps_per_year,
                                                       n_paths, seed);
            py::array_t<double> out({static_cast<py::ssize_t>(n_paths),
                                     static_cast<py::ssize_t>(times.size())});
            std::copy(v.begin(), v.end(), out.mutable_data());
            return out;
        },
        py::arg("surface"), py::arg("S0"), py::arg("r"), py::arg("times"),
        py::arg("steps_per_year"), py::arg("n_paths"), py::arg("seed") = 42,
        "Local vol paths sampled at the given times; array (n_paths, len(times))");

    m.def("heston_at_times",
        [](CirScheme scheme, double S0, double r, double kappa, double vbar, double gamma,
           double rho, double v0, std::vector<double> times, int steps_per_year, int n_paths,
           std::uint64_t seed) {
            std::vector<double> v = heston_at_times(scheme, S0, r, kappa, vbar, gamma, rho, v0,
                                                    times, steps_per_year, n_paths, seed);
            py::array_t<double> out({static_cast<py::ssize_t>(n_paths),
                                     static_cast<py::ssize_t>(times.size())});
            std::copy(v.begin(), v.end(), out.mutable_data());
            return out;
        },
        py::arg("scheme"), py::arg("S0"), py::arg("r"), py::arg("kappa"), py::arg("vbar"),
        py::arg("gamma"), py::arg("rho"), py::arg("v0"), py::arg("times"),
        py::arg("steps_per_year"), py::arg("n_paths"), py::arg("seed") = 42,
        "Heston AES paths sampled at the given times; array (n_paths, len(times))");

    // ---- Interest rates: Hull-White (Step 8) ----
    py::class_<NelsonSiegelCurve>(m, "NelsonSiegelCurve")
        .def(py::init<double, double, double, double>(),
             py::arg("b0"), py::arg("b1"), py::arg("b2"), py::arg("tau"))
        .def("zero_rate", &NelsonSiegelCurve::zero_rate, py::arg("T"))
        .def("discount", &NelsonSiegelCurve::discount, py::arg("T"))
        .def("inst_forward", &NelsonSiegelCurve::inst_forward, py::arg("T"))
        .def("forward_slope", &NelsonSiegelCurve::forward_slope, py::arg("T"));

    py::class_<HullWhiteModel>(m, "HullWhiteModel")
        .def(py::init<double, double, NelsonSiegelCurve>(),
             py::arg("lambda_"), py::arg("eta"), py::arg("curve"))
        .def("theta", &HullWhiteModel::theta, py::arg("t"))
        .def("psi", &HullWhiteModel::psi, py::arg("t"))
        .def("r0", &HullWhiteModel::r0)
        .def("mean_r", &HullWhiteModel::mean_r, py::arg("t"))
        .def("var_r", &HullWhiteModel::var_r, py::arg("t"))
        .def("zcb", &HullWhiteModel::zcb, py::arg("t"), py::arg("T"), py::arg("r"))
        .def("curve", &HullWhiteModel::curve)
        .def("lambda_", &HullWhiteModel::lambda)
        .def("eta", &HullWhiteModel::eta);

    m.def("simulate_hull_white",
        [](const HullWhiteModel& model, double T, int n_steps, int n_paths, std::uint64_t seed) {
            HullWhitePaths res = simulate_hull_white(model, T, n_steps, n_paths, seed);
            py::array_t<double> r({n_paths, n_steps + 1});
            py::array_t<double> I({n_paths, n_steps + 1});
            std::copy(res.r.begin(), res.r.end(), r.mutable_data());
            std::copy(res.integral.begin(), res.integral.end(), I.mutable_data());
            return py::make_tuple(r, I);
        },
        py::arg("model"), py::arg("T"), py::arg("n_steps"), py::arg("n_paths"),
        py::arg("seed") = 42,
        "Hull-White short-rate paths and the running integral of r; returns (r, integral)");

    // ---- Swaps, exposure and CVA (Step 9) ----
    py::class_<Swap>(m, "Swap")
        .def(py::init([](double notional, double K, double start, double end, double tau,
                         bool payer) {
                 Swap s; s.notional = notional; s.K = K; s.start = start; s.end = end;
                 s.tau = tau; s.payer = payer;
                 s.payment_dates();   // validates the schedule
                 return s;
             }),
             py::arg("notional") = 1.0, py::arg("K") = 0.03, py::arg("start") = 0.0,
             py::arg("end") = 10.0, py::arg("tau") = 1.0, py::arg("payer") = true)
        .def_readwrite("notional", &Swap::notional)
        .def_readwrite("K", &Swap::K)
        .def_readwrite("start", &Swap::start)
        .def_readwrite("end", &Swap::end)
        .def_readwrite("tau", &Swap::tau)
        .def_readwrite("payer", &Swap::payer)
        .def("payment_dates", &Swap::payment_dates);

    m.def("swap_value_today", &swap_value_today, py::arg("curve"), py::arg("swap"),
          "Today's swap value from the curve (book eq. 12.12)");
    m.def("swap_annuity", &swap_annuity, py::arg("curve"), py::arg("swap"));
    m.def("par_swap_rate", &par_swap_rate, py::arg("curve"), py::arg("swap"),
          "Fixed rate that makes the swap worth zero today (book eq. 12.14)");

    m.def("simulate_exposure",
        [](const HullWhiteModel& model, const std::vector<Swap>& trades, double horizon,
           int steps_per_year, int n_paths, std::uint64_t seed) {
            ExposureSimulation res =
                simulate_exposure(model, trades, horizon, steps_per_year, n_paths, seed);
            py::array_t<double> times(res.n_times);
            py::array_t<double> values({res.n_trades, res.n_paths, res.n_times});
            py::array_t<double> disc({res.n_paths, res.n_times});
            std::copy(res.times.begin(), res.times.end(), times.mutable_data());
            std::copy(res.values.begin(), res.values.end(), values.mutable_data());
            std::copy(res.discount.begin(), res.discount.end(), disc.mutable_data());
            return py::make_tuple(times, values, disc);
        },
        py::arg("model"), py::arg("trades"), py::arg("horizon"), py::arg("steps_per_year") = 24,
        py::arg("n_paths") = 5000, py::arg("seed") = 42,
        "Hull-White paths with every swap revalued on them; returns (times, values, discount) "
        "with values of shape (n_trades, n_paths, n_times)");

    m.def("cva", &cva, py::arg("times"), py::arg("ee"), py::arg("lgd"), py::arg("hazard"),
          "Unilateral CVA = LGD * sum EE(t_k) * PD(t_{k-1}, t_k) (book eq. 12.61)");
}
