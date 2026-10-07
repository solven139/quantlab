#include <pybind11/pybind11.h>
#include "quantlab/black_scholes.hpp"

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
}