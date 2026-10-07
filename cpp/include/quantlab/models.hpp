#pragma once

#include <complex>

namespace quantlab {

// Interface for any asset model that the COS method (book Ch 6) can price.
//
// A model only has to describe X = log(S(T) / S(0)) under the risk-neutral measure:
//   - its characteristic function  phi(u) = E[ exp(i u X) ]
//   - its first, second and fourth cumulants (used to choose the integration range)
//
// Pricers depend on this interface, never on a concrete model. Adding Merton, Kou,
// Variance Gamma or Heston later means adding a new class, not changing the pricer.
class Model {
public:
    virtual ~Model() = default;

    virtual double rate() const = 0;  // risk-free rate r used for discounting
    virtual std::complex<double> char_fn(double u, double T) const = 0;
    virtual double cumulant1(double T) const = 0;  // mean of X
    virtual double cumulant2(double T) const = 0;  // variance of X
    virtual double cumulant4(double T) const = 0;  // 4th cumulant of X (0 for normal)
};

// Geometric Brownian Motion: X ~ Normal((r - sigma^2/2) T, sigma^2 T)
class BlackScholesModel : public Model {
public:
    BlackScholesModel(double r, double sigma);

    double rate() const override { return r_; }
    double sigma() const { return sigma_; }
    std::complex<double> char_fn(double u, double T) const override;
    double cumulant1(double T) const override;
    double cumulant2(double T) const override;
    double cumulant4(double T) const override;

private:
    double r_;
    double sigma_;
};

} // namespace quantlab
