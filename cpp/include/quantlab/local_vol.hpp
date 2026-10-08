#pragma once

#include <cstdint>
#include <vector>

#include "quantlab/mc_schemes.hpp"
#include "quantlab/models.hpp"

namespace quantlab {

// Dupire's local volatility (book eq. 4.49), from the call prices of ANY Model:
//   sigma_LV^2(T, K) = ( dC/dT + r K dC/dK ) / ( 1/2 K^2 d^2C/dK^2 )
// The derivatives are central finite differences of COS prices with steps dK and dT.
double dupire_local_vol(const Model& model, double S0, double K, double T,
                        double dK, double dT);

// sigma_LV on a (T, K) grid, with bilinear interpolation in between and flat
// extrapolation outside the grid.
class LocalVolSurface {
public:
    LocalVolSurface(const Model& model, double S0, std::vector<double> strikes,
                    std::vector<double> maturities);

    double sigma(double S, double t) const;  // local vol at stock level S and time t
    const std::vector<double>& strikes() const { return K_; }
    const std::vector<double>& maturities() const { return T_; }
    const std::vector<double>& values() const { return vol_; }  // row-major [T][K]

private:
    std::vector<double> K_, T_, vol_;
};

// Simulate dS/S = r dt + sigma_LV(S, t) dW (Euler on log S) and record S at each of
// the requested times. Returns n_paths * times.size() values, row by row.
std::vector<double> local_vol_at_times(const LocalVolSurface& surface, double S0, double r,
                                       const std::vector<double>& times, int steps_per_year,
                                       int n_paths, std::uint64_t seed);

// Heston AES simulation (Step 6) recording S at each of the requested times.
std::vector<double> heston_at_times(CirScheme scheme, double S0, double r, double kappa,
                                    double vbar, double gamma, double rho, double v0,
                                    const std::vector<double>& times, int steps_per_year,
                                    int n_paths, std::uint64_t seed);

} // namespace quantlab
