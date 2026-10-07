#include "quantlab/models.hpp"

#include <cmath>
#include <stdexcept>

namespace quantlab {

BlackScholesModel::BlackScholesModel(double r, double sigma) : r_(r), sigma_(sigma) {
    if (sigma <= 0.0)
        throw std::invalid_argument("BlackScholesModel: sigma must be positive");
}

std::complex<double> BlackScholesModel::char_fn(double u, double T) const {
    const std::complex<double> i(0.0, 1.0);
    const double mu = (r_ - 0.5 * sigma_ * sigma_) * T;
    return std::exp(i * u * mu - 0.5 * sigma_ * sigma_ * u * u * T);
    //         The member variables are r_ and sigma_ (the trailing _ marks a member).
    return std::complex<double>(1.0, 0.0);  // placeholder: replace this line
}

double BlackScholesModel::cumulant1(double T) const { return (r_ - 0.5 * sigma_ * sigma_) * T; }
double BlackScholesModel::cumulant2(double T) const { return sigma_ * sigma_ * T; }
double BlackScholesModel::cumulant4(double) const { return 0.0; }

} // namespace quantlab
