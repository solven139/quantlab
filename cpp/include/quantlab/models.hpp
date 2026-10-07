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

// Merton jump diffusion (book Ch 5.1): GBM plus jumps that arrive as a Poisson process
// with intensity xi (jumps per year). Each jump multiplies S by e^J, J ~ Normal(mu_j, sigma_j^2).
//   dS/S = (r - omega) dt + sigma dW + (e^J - 1) dN,   omega = xi * (E[e^J] - 1)
class MertonModel : public Model {
public:
    MertonModel(double r, double sigma, double xi, double mu_j, double sigma_j);

    double rate() const override { return r_; }
    std::complex<double> char_fn(double u, double T) const override;
    double cumulant1(double T) const override;
    double cumulant2(double T) const override;
    double cumulant4(double T) const override;
    double drift_correction() const;  // omega: keeps the discounted stock a martingale

private:
    double r_, sigma_, xi_, mu_j_, sigma_j_;
};

// Kou double-exponential jump diffusion (book Ch 5.1.3): like Merton, but a jump is
// upward with probability p (size ~ Exponential(eta1)) or downward with probability 1-p
// (size ~ -Exponential(eta2)). Asymmetric, fat-tailed jumps. Needs eta1 > 1.
class KouModel : public Model {
public:
    KouModel(double r, double sigma, double xi, double p, double eta1, double eta2);

    double rate() const override { return r_; }
    std::complex<double> char_fn(double u, double T) const override;
    double cumulant1(double T) const override;
    double cumulant2(double T) const override;
    double cumulant4(double T) const override;
    double drift_correction() const;

private:
    double r_, sigma_, xi_, p_, eta1_, eta2_;
};

// Variance Gamma (book Ch 5.4.1): no Brownian part at all. Brownian motion with drift
// theta and volatility sigma, run on a random Gamma "business clock" with variance rate beta.
// Infinitely many small jumps. theta < 0 gives a negative skew, beta controls the kurtosis.
class VarianceGammaModel : public Model {
public:
    VarianceGammaModel(double r, double sigma, double theta, double beta);

    double rate() const override { return r_; }
    std::complex<double> char_fn(double u, double T) const override;
    double cumulant1(double T) const override;
    double cumulant2(double T) const override;
    double cumulant4(double T) const override;
    double drift_correction() const;  // omega-bar of book eq. 5.59

private:
    double r_, sigma_, theta_, beta_;
};

// Heston stochastic volatility (book Ch 8.2-8.3):
//   dS/S = r dt + sqrt(v) dW_x
//   dv   = kappa (vbar - v) dt + gamma sqrt(v) dW_v,    dW_x dW_v = rho dt
// v is the variance (volatility squared). kappa: speed of mean reversion, vbar: long-run
// variance, gamma: volatility of variance ("vol of vol"), rho: correlation, v0: today's variance.
class HestonModel : public Model {
public:
    HestonModel(double r, double kappa, double vbar, double gamma, double rho, double v0);

    double rate() const override { return r_; }
    std::complex<double> char_fn(double u, double T) const override;
    double cumulant1(double T) const override;
    double cumulant2(double T) const override;
    double cumulant4(double T) const override;
    // Feller condition 2 kappa vbar >= gamma^2: if it holds, v(t) never reaches zero
    bool feller_satisfied() const;

    double kappa() const { return kappa_; }
    double vbar() const { return vbar_; }
    double gamma() const { return gamma_; }
    double rho() const { return rho_; }
    double v0() const { return v0_; }

private:
    double r_, kappa_, vbar_, gamma_, rho_, v0_;
};

} // namespace quantlab
