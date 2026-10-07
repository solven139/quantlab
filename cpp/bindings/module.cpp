#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <vector>
#include "quantlab/black_scholes.hpp"
#include "quantlab/gbm.hpp"

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
}