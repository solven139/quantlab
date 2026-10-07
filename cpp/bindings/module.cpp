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
}
