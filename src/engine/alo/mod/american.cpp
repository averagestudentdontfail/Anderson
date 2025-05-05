#include "american.h"
#include "../num/chebyshev.h"
#include "../num/integrate.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace engine {
namespace alo {
namespace mod {

AmericanOption::AmericanOption(std::shared_ptr<num::Integrator> integrator)
    : integrator_(integrator) {
  if (!integrator_) {
    throw std::invalid_argument("AmericanOption: Integrator cannot be null");
  }
}

AmericanPut::AmericanPut(std::shared_ptr<num::Integrator> integrator)
    : AmericanOption(integrator) {}

double AmericanPut::xMax(double K, double r, double q) const {
  // Table 2 from the ALO paper for puts
  if (r > 0.0 && q > 0.0)
    return K * std::min(1.0, r / q);
  else if (r > 0.0 && q <= 0.0)
    return K;
  else if (r == 0.0 && q < 0.0)
    return K;
  else if (r == 0.0 && q >= 0.0)
    return 0.0; // European case
  else if (r < 0.0 && q >= 0.0)
    return 0.0; // European case
  else if (r < 0.0 && q < r)
    return K; // double boundary case
  else if (r < 0.0 && r <= q && q < 0.0)
    return 0.0; // European case
  else
    throw std::runtime_error("Internal error in xMax calculation for put");
}

double AmericanPut::calculateEarlyExercisePremium(
    double S, double K, double r, double q, double vol, double T,
    const std::shared_ptr<num::ChebyshevInterpolation> &boundary) const {

  if (!boundary) {
    throw std::invalid_argument(
        "AmericanPut: Boundary interpolation cannot be null");
  }

  const double xmax = xMax(K, r, q);

  // Function to map tau to z in [-1, 1]
  auto tauToZ = [T](double tau) -> double {
    return 2.0 * std::sqrt(tau / T) - 1.0;
  };

  // Function to get boundary value at tau
  auto B = [xmax, boundary, tauToZ](double tau) -> double {
    if (tau <= 0.0)
      return xmax;

    const double z = tauToZ(tau);
    return xmax * std::exp(-std::sqrt(std::max(0.0, (*boundary)(z, true))));
  };

  // Integrand for put early exercise premium
  auto integrand = [S, K, r, q, vol, B](double z) -> double {
    const double t = z * z; // tau = z^2
    const double b_t = B(t);

    if (b_t <= 0.0)
      return 0.0;

    const double dr = std::exp(-r * t);
    const double dq = std::exp(-q * t);
    const double v = vol * std::sqrt(t);

    if (v < 1e-10) {
      if (std::abs(S * dq - b_t * dr) < 1e-10)
        return z * (r * K * dr - q * S * dq);
      else if (b_t * dr > S * dq)
        return 2.0 * z * (r * K * dr - q * S * dq);
      else
        return 0.0;
    }

    const double dp = (std::log(S * dq / (b_t * dr)) / v) + 0.5 * v;

    return 2.0 * z *
           (r * K * dr * (0.5 * (1.0 + std::erf((-dp + v) / std::sqrt(2.0)))) -
            q * S * dq * (0.5 * (1.0 + std::erf(-dp / std::sqrt(2.0)))));
  };

  // Integrate to get early exercise premium
  return integrator_->integrate(integrand, 0.0, std::sqrt(T));
}

std::shared_ptr<num::ChebyshevInterpolation>
AmericanPut::calculateExerciseBoundary(
    double S, double K, double r, double q, double vol, double T,
    size_t num_nodes, size_t num_iterations,
    std::shared_ptr<num::Integrator> fpIntegrator) const {

  if (!fpIntegrator) {
    throw std::invalid_argument(
        "AmericanPut: Fixed point integrator cannot be null");
  }

  const double xmax = xMax(K, r, q);

  // Initialize interpolation nodes
  std::vector<double> nodes(num_nodes);
  std::vector<double> y(num_nodes, 0.0); // Boundary function values at nodes

  // Chebyshev nodes of the second kind in [-1, 1]
  for (size_t i = 0; i < num_nodes; ++i) {
    nodes[i] = std::cos(M_PI * i / (num_nodes - 1)); // x_i in [-1, 1]
  }

  // Create initial interpolation
  auto interp =
      num::createChebyshevInterpolation(nodes, y, num::SECOND_KIND, -1.0, 1.0);

  // Function to map tau to z in [-1, 1]
  auto tauToZ = [T](double tau) -> double {
    return 2.0 * std::sqrt(tau / T) - 1.0;
  };

  // Function to get boundary value at tau
  auto B = [xmax, &interp, tauToZ](double tau) -> double {
    if (tau <= 0.0)
      return xmax;

    const double z = tauToZ(tau);
    return xmax * std::exp(-std::sqrt(std::max(0.0, (*interp)(z, true))));
  };

  // Function to map boundary value to interpolation y-value
  auto h = [xmax](double fv) -> double {
    return std::pow(std::log(fv / xmax), 2);
  };

  // Create fixed point evaluator
  auto evaluator =
      createFixedPointEvaluator('A', K, r, q, vol, B, fpIntegrator);

  // Perform fixed point iterations
  // First is a Jacobi-Newton step
  for (size_t k = 0; k < 1; ++k) {
    for (size_t i = 1; i < num_nodes;
         ++i) { // Skip first node (corresponds to tau=0)
      const double z = nodes[i];
      const double tau = T * std::pow(0.5 * (1.0 + z), 2);
      const double b = B(tau);

      const auto [N, D, fv] = evaluator->evaluate(tau, b);

      if (tau < 1e-10) {
        y[i] = h(fv);
      } else {
        const auto [Nd, Dd] = evaluator->derivatives(tau, b);

        // Newton step
        const double fd =
            K * std::exp(-(r - q) * tau) * (Nd / D - Dd * N / (D * D));
        const double b_new = b - (fv - b) / (fd - 1.0);

        y[i] = h(b_new);
      }
    }

    // Update interpolation
    interp->updateValues(y);
  }

  // Remaining iterations are standard fixed point
  for (size_t k = 1; k < num_iterations; ++k) {
    for (size_t i = 1; i < num_nodes;
         ++i) { // Skip first node (corresponds to tau=0)
      const double z = nodes[i];
      const double tau = T * std::pow(0.5 * (1.0 + z), 2);
      const double b = B(tau);

      const auto [N, D, fv] = evaluator->evaluate(tau, b);

      y[i] = h(fv);
    }

    // Update interpolation
    interp->updateValues(y);
  }

  return interp;
}

AmericanCall::AmericanCall(std::shared_ptr<num::Integrator> integrator)
    : AmericanOption(integrator) {}

double AmericanCall::xMax(double K, double r, double q) const {
  // For call options, the early exercise boundary is different
  if (q > 0.0 && r >= 0.0)
    return K * std::max(1.0, r / q);
  else if (q <= 0.0 && r >= 0.0)
    return std::numeric_limits<double>::
        infinity(); // Effectively infinite, early exercise never optimal
  else if (q >= r && r < 0.0)
    return std::numeric_limits<double>::infinity(); // European case
  else if (q < r && r < 0.0)
    return K; // Double boundary case
  else
    throw std::runtime_error("Internal error in xMax calculation for call");
}

double AmericanCall::calculateEarlyExercisePremium(
    double S, double K, double r, double q, double vol, double T,
    const std::shared_ptr<num::ChebyshevInterpolation> &boundary) const {

  if (!boundary) {
    throw std::invalid_argument(
        "AmericanCall: Boundary interpolation cannot be null");
  }

  // If zero dividend, calls should never be exercised early
  if (q <= 0.0) {
    return 0.0;
  }

  const double xmax = xMax(K, r, q);

  // For infinite boundary (no early exercise), return zero premium
  if (std::isinf(xmax) || xmax > 1e12) {
    return 0.0;
  }

  // Function to map tau to z in [-1, 1]
  auto tauToZ = [T](double tau) -> double {
    return 2.0 * std::sqrt(tau / T) - 1.0;
  };

  // Function to get boundary value at tau
  auto B = [xmax, boundary, tauToZ](double tau) -> double {
    if (tau <= 0.0)
      return xmax;

    const double z = tauToZ(tau);
    return xmax * std::exp(-std::sqrt(std::max(0.0, (*boundary)(z, true))));
  };

  // Integrand for call early exercise premium
  auto integrand = [S, K, r, q, vol, B](double z) -> double {
    const double t = z * z; // tau = z^2
    const double b_t = B(t);

    if (b_t <= 0.0 || std::isinf(b_t))
      return 0.0;

    const double dr = std::exp(-r * t);
    const double dq = std::exp(-q * t);
    const double v = vol * std::sqrt(t);

    if (v < 1e-10) {
      if (std::abs(b_t * dq - K * dr) < 1e-10)
        return z * (q * b_t * dq - r * K * dr);
      else if (b_t * dq > K * dr)
        return 2.0 * z * (q * b_t * dq - r * K * dr);
      else
        return 0.0;
    }

    const double dp = (std::log(b_t * dq / (K * dr)) / v) + 0.5 * v;

    return 2.0 * z *
           (q * b_t * dq * (0.5 * (1.0 + std::erf(dp / std::sqrt(2.0)))) -
            r * K * dr * (0.5 * (1.0 + std::erf((dp - v) / std::sqrt(2.0)))));
  };

  // Integrate to get early exercise premium
  return integrator_->integrate(integrand, 0.0, std::sqrt(T));
}

std::shared_ptr<num::ChebyshevInterpolation>
AmericanCall::calculateExerciseBoundary(
    double S, double K, double r, double q, double vol, double T,
    size_t num_nodes, size_t num_iterations,
    std::shared_ptr<num::Integrator> fpIntegrator) const {

  if (!fpIntegrator) {
    throw std::invalid_argument(
        "AmericanCall: Fixed point integrator cannot be null");
  }

  const double xmax = xMax(K, r, q);

  // For infinite boundary (no early exercise), return an empty boundary
  if (std::isinf(xmax) || xmax > 1e12) {
    std::vector<double> nodes(num_nodes);
    std::vector<double> y(num_nodes, 0.0);

    for (size_t i = 0; i < num_nodes; ++i) {
      nodes[i] = std::cos(M_PI * i / (num_nodes - 1));
    }

    return num::createChebyshevInterpolation(nodes, y, num::SECOND_KIND, -1.0,
                                             1.0);
  }

  // Initialize interpolation nodes
  std::vector<double> nodes(num_nodes);
  std::vector<double> y(num_nodes, 0.0); // Boundary function values at nodes

  // Chebyshev nodes of the second kind in [-1, 1]
  for (size_t i = 0; i < num_nodes; ++i) {
    nodes[i] = std::cos(M_PI * i / (num_nodes - 1)); // x_i in [-1, 1]
  }

  // Create initial interpolation
  auto interp =
      num::createChebyshevInterpolation(nodes, y, num::SECOND_KIND, -1.0, 1.0);

  // Function to map tau to z in [-1, 1]
  auto tauToZ = [T](double tau) -> double {
    return 2.0 * std::sqrt(tau / T) - 1.0;
  };

  // Function to get boundary value at tau
  auto B = [xmax, &interp, tauToZ](double tau) -> double {
    if (tau <= 0.0)
      return xmax;

    const double z = tauToZ(tau);
    return xmax * std::exp(-std::sqrt(std::max(0.0, (*interp)(z, true))));
  };

  // Function to map boundary value to interpolation y-value
  auto h = [xmax](double fv) -> double {
    return std::pow(std::log(fv / xmax), 2);
  };

  // For calls, we swap r and q in the fixed point equation (put-call symmetry)
  auto evaluator =
      createFixedPointEvaluator('A', K, q, r, vol, B, fpIntegrator);

  // Perform fixed point iterations
  // First is a Jacobi-Newton step
  for (size_t k = 0; k < 1; ++k) {
    for (size_t i = 1; i < num_nodes;
         ++i) { // Skip first node (corresponds to tau=0)
      const double z = nodes[i];
      const double tau = T * std::pow(0.5 * (1.0 + z), 2);
      const double b = B(tau);

      const auto [N, D, fv] = evaluator->evaluate(tau, b);

      if (tau < 1e-10) {
        y[i] = h(fv);
      } else {
        const auto [Nd, Dd] = evaluator->derivatives(tau, b);

        // Newton step
        const double fd =
            K * std::exp(-(q - r) * tau) * (Nd / D - Dd * N / (D * D));
        const double b_new = b - (fv - b) / (fd - 1.0);

        y[i] = h(b_new);
      }
    }

    // Update interpolation
    interp->updateValues(y);
  }

  // Remaining iterations are standard fixed point
  for (size_t k = 1; k < num_iterations; ++k) {
    for (size_t i = 1; i < num_nodes;
         ++i) { // Skip first node (corresponds to tau=0)
      const double z = nodes[i];
      const double tau = T * std::pow(0.5 * (1.0 + z), 2);
      const double b = B(tau);

      const auto [N, D, fv] = evaluator->evaluate(tau, b);

      y[i] = h(fv);
    }

    // Update interpolation
    interp->updateValues(y);
  }

  return interp;
}

FixedPointEvaluator::FixedPointEvaluator(
    double K, double r, double q, double vol,
    const std::function<double(double)> &B,
    std::shared_ptr<num::Integrator> integrator)
    : K_(K), r_(r), q_(q), vol_(vol), vol2_(vol * vol), B_(B),
      integrator_(integrator) {}

std::pair<double, double> FixedPointEvaluator::d(double t, double z) const {
  if (t <= 0.0 || z <= 0.0) {
    return {-std::numeric_limits<double>::infinity(),
            -std::numeric_limits<double>::infinity()};
  }

  const double v = vol_ * std::sqrt(t);
  const double m = (std::log(z) + (r_ - q_) * t) / v + 0.5 * v;

  return {m, m - v};
}

double FixedPointEvaluator::normalCDF(double x) const {
  // Using standard error function implementation
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double FixedPointEvaluator::normalPDF(double x) const {
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

EquationA::EquationA(double K, double r, double q, double vol,
                     const std::function<double(double)> &B,
                     std::shared_ptr<num::Integrator> integrator)
    : FixedPointEvaluator(K, r, q, vol, B, integrator) {}

std::tuple<double, double, double> EquationA::evaluate(double tau,
                                                       double b) const {
  double N, D;
  const double eps = 1e-10;

  if (tau < eps * eps) {
    // For very small tau, use limit values
    if (std::abs(b - K_) < eps) {
      N = D = 0.5;
    } else {
      N = D = (b > K_) ? 1.0 : 0.0;
    }
  } else {
    const double stv = std::sqrt(tau) / vol_;

    // Integrate to get K1 + K2
    auto K12_integrand = [&](double y) -> double {
      const double m = 0.25 * tau * (1.0 + y) * (1.0 + y);
      const double df = std::exp(q_ * tau - q_ * m);

      if (y <= 5.0 * eps - 1.0) {
        if (std::abs(b - B_(tau - m)) < eps) {
          return df * stv / (std::sqrt(2.0 * M_PI));
        } else {
          return 0.0;
        }
      } else {
        const auto dp = d(m, b / B_(tau - m)).first;
        return df *
               (0.5 * tau * (y + 1.0) * normalCDF(dp) + stv * normalPDF(dp));
      }
    };

    // Integrate to get K3
    auto K3_integrand = [&](double y) -> double {
      const double m = 0.25 * tau * (1.0 + y) * (1.0 + y);
      const double df = std::exp(r_ * tau - r_ * m);

      if (y <= 5.0 * eps - 1.0) {
        if (std::abs(b - B_(tau - m)) < eps) {
          return df * stv / (std::sqrt(2.0 * M_PI));
        } else {
          return 0.0;
        }
      } else {
        return df * stv * normalPDF(d(m, b / B_(tau - m)).second);
      }
    };

    double K12 = integrator_->integrate(K12_integrand, -1.0, 1.0);
    double K3 = integrator_->integrate(K3_integrand, -1.0, 1.0);

    const auto dpm = d(tau, b / K_);
    N = normalPDF(dpm.second) / vol_ / std::sqrt(tau) + r_ * K3;
    D = normalPDF(dpm.first) / vol_ / std::sqrt(tau) + normalCDF(dpm.first) +
        q_ * K12;
  }

  // Calculate function value
  const double alpha = K_ * std::exp(-(r_ - q_) * tau);
  double fv;

  if (tau < eps * eps) {
    if (std::abs(b - K_) < eps || b > K_) {
      fv = alpha;
    } else {
      if (std::abs(q_) < eps) {
        fv = alpha * r_ * ((q_ < 0.0) ? -1.0 : 1.0) / eps;
      } else {
        fv = alpha * r_ / q_;
      }
    }
  } else {
    fv = alpha * N / D;
  }

  return {N, D, fv};
}

std::pair<double, double> EquationA::derivatives(double tau, double b) const {
  double Nd, Dd;
  const double eps = 1e-10;

  if (tau < eps * eps) {
    if (std::abs(b - K_) < eps) {
      const double sqTau = std::sqrt(tau);
      Dd = M_2_SQRTPI * M_SQRT1_2 *
           (-(0.5 * vol2_ + r_ - q_) / (b * vol_ * vol2_ * sqTau) +
            1.0 / (b * vol_ * sqTau));
      Nd = M_2_SQRTPI * M_SQRT1_2 * (-0.5 * vol2_ + r_ - q_) /
           (b * vol_ * vol2_ * sqTau);
    } else {
      Dd = Nd = 0.0;
    }
  } else {
    const auto dpm = d(tau, b / K_);

    Dd = -normalPDF(dpm.first) * dpm.first / (b * vol2_ * tau) +
         normalPDF(dpm.first) / (b * vol_ * std::sqrt(tau));
    Nd = -normalPDF(dpm.second) * dpm.second / (b * vol2_ * tau);
  }

  return {Nd, Dd};
}

EquationB::EquationB(double K, double r, double q, double vol,
                     const std::function<double(double)> &B,
                     std::shared_ptr<num::Integrator> integrator)
    : FixedPointEvaluator(K, r, q, vol, B, integrator) {}

std::tuple<double, double, double> EquationB::evaluate(double tau,
                                                       double b) const {
  double N, D;
  const double eps = 1e-10;

  if (tau < eps * eps) {
    // For very small tau, use limit values
    if (std::abs(b - K_) < eps) {
      N = D = 0.5;
    } else if (b < K_) {
      N = D = 0.0;
    } else {
      N = D = 1.0;
    }
  } else {
    // Integrate for N and D
    auto N_integrand = [&](double u) -> double {
      const double df = std::exp(r_ * u);
      if (u >= tau * (1.0 - 5.0 * eps)) {
        if (std::abs(b - B_(u)) < eps) {
          return 0.5 * df;
        } else {
          return df * ((b < B_(u)) ? 0.0 : 1.0);
        }
      } else {
        return df * normalCDF(d(tau - u, b / B_(u)).second);
      }
    };

    auto D_integrand = [&](double u) -> double {
      const double df = std::exp(q_ * u);
      if (u >= tau * (1.0 - 5.0 * eps)) {
        if (std::abs(b - B_(u)) < eps) {
          return 0.5 * df;
        } else {
          return df * ((b < B_(u)) ? 0.0 : 1.0);
        }
      } else {
        return df * normalCDF(d(tau - u, b / B_(u)).first);
      }
    };

    double ni = integrator_->integrate(N_integrand, 0.0, tau);
    double di = integrator_->integrate(D_integrand, 0.0, tau);

    const auto dpm = d(tau, b / K_);

    N = normalCDF(dpm.second) + r_ * ni;
    D = normalCDF(dpm.first) + q_ * di;
  }

  // Calculate function value
  const double alpha = K_ * std::exp(-(r_ - q_) * tau);
  double fv;

  if (tau < eps * eps) {
    if (std::abs(b - K_) < eps || b > K_) {
      fv = alpha;
    } else {
      if (std::abs(q_) < eps) {
        fv = alpha * r_ * ((q_ < 0.0) ? -1.0 : 1.0) / eps;
      } else {
        fv = alpha * r_ / q_;
      }
    }
  } else {
    fv = alpha * N / D;
  }

  return {N, D, fv};
}

std::pair<double, double> EquationB::derivatives(double tau, double b) const {
  const double eps = 1e-10;

  if (tau < eps * eps) {
    return {0.0, 0.0};
  }

  const auto dpm = d(tau, b / K_);

  return {normalPDF(dpm.second) / (b * vol_ * std::sqrt(tau)),
          normalPDF(dpm.first) / (b * vol_ * std::sqrt(tau))};
}

std::shared_ptr<FixedPointEvaluator>
createFixedPointEvaluator(char equation, double K, double r, double q,
                          double vol, const std::function<double(double)> &B,
                          std::shared_ptr<num::Integrator> integrator) {

  if (equation == 'A') {
    return std::make_shared<EquationA>(K, r, q, vol, B, integrator);
  } else if (equation == 'B') {
    return std::make_shared<EquationB>(K, r, q, vol, B, integrator);
  } else {
    // Auto-select equation based on r and q
    if (std::abs(r - q) < 0.001) {
      return std::make_shared<EquationA>(K, r, q, vol, B, integrator);
    } else {
      return std::make_shared<EquationB>(K, r, q, vol, B, integrator);
    }
  }
}

AmericanOptionFloat::AmericanOptionFloat(
    std::shared_ptr<num::IntegratorFloat> integrator)
    : integrator_(integrator) {
  if (!integrator_) {
    throw std::invalid_argument(
        "AmericanOptionFloat: Integrator cannot be null");
  }
}

// AmericanPutFloat implementation
AmericanPutFloat::AmericanPutFloat(
    std::shared_ptr<num::IntegratorFloat> integrator)
    : AmericanOptionFloat(integrator) {}

float AmericanPutFloat::xMax(float K, float r, float q) const {
  // Table 2 from the ALO paper for puts
  if (r > 0.0f && q > 0.0f)
    return K * std::min(1.0f, r / q);
  else if (r > 0.0f && q <= 0.0f)
    return K;
  else if (r == 0.0f && q < 0.0f)
    return K;
  else if (r == 0.0f && q >= 0.0f)
    return 0.0f; // European case
  else if (r < 0.0f && q >= 0.0f)
    return 0.0f; // European case
  else if (r < 0.0f && q < r)
    return K; // double boundary case
  else if (r < 0.0f && r <= q && q < 0.0f)
    return 0.0f; // European case
  else
    throw std::runtime_error("Internal error in xMax calculation for put");
}

float AmericanPutFloat::calculateEarlyExercisePremium(
    float S, float K, float r, float q, float vol, float T,
    const std::shared_ptr<num::ChebyshevInterpolationFloat> &boundary) const {

  if (!boundary) {
    throw std::invalid_argument(
        "AmericanPutFloat: Boundary interpolation cannot be null");
  }

  const float xmax = xMax(K, r, q);

  // Function to map tau to z in [-1, 1]
  auto tauToZ = [T](float tau) -> float {
    return 2.0f * std::sqrt(tau / T) - 1.0f;
  };

  // Function to get boundary value at tau
  auto B = [xmax, boundary, tauToZ](float tau) -> float {
    if (tau <= 0.0f)
      return xmax;

    const float z = tauToZ(tau);
    return xmax * std::exp(-std::sqrt(std::max(0.0f, (*boundary)(z, true))));
  };

  // Integrand for put early exercise premium
  auto integrand = [S, K, r, q, vol, B](float z) -> float {
    const float t = z * z; // tau = z^2
    const float b_t = B(t);

    if (b_t <= 0.0f)
      return 0.0f;

    const float dr = std::exp(-r * t);
    const float dq = std::exp(-q * t);
    const float v = vol * std::sqrt(t);

    if (v < 1e-5f) {
      if (std::abs(S * dq - b_t * dr) < 1e-5f)
        return z * (r * K * dr - q * S * dq);
      else if (b_t * dr > S * dq)
        return 2.0f * z * (r * K * dr - q * S * dq);
      else
        return 0.0f;
    }

    const float dp = (std::log(S * dq / (b_t * dr)) / v) + 0.5f * v;

    return 2.0f * z *
           (r * K * dr *
                (0.5f * (1.0f + num::fast_erf((-dp + v) / 1.414213562f))) -
            q * S * dq * (0.5f * (1.0f + num::fast_erf(-dp / 1.414213562f))));
  };

  // Integrate to get early exercise premium
  return integrator_->integrate(integrand, 0.0f, std::sqrt(T));
}

std::shared_ptr<num::ChebyshevInterpolationFloat>
AmericanPutFloat::calculateExerciseBoundary(
    float S, float K, float r, float q, float vol, float T, size_t num_nodes,
    size_t num_iterations,
    std::shared_ptr<num::IntegratorFloat> fpIntegrator) const {

  if (!fpIntegrator) {
    throw std::invalid_argument(
        "AmericanPutFloat: Fixed point integrator cannot be null");
  }

  const float xmax = xMax(K, r, q);

  // Initialize interpolation nodes
  std::vector<float> nodes(num_nodes);
  std::vector<float> y(num_nodes, 0.0f); // Boundary function values at nodes

  // Chebyshev nodes of the second kind in [-1, 1]
  for (size_t i = 0; i < num_nodes; ++i) {
    nodes[i] = std::cos(3.14159265358979323846f * i /
                        (num_nodes - 1)); // x_i in [-1, 1]
  }

  // Create initial interpolation
  auto interp = num::createChebyshevInterpolationFloat(
      nodes, y, num::SECOND_KIND, -1.0f, 1.0f);

  // Function to map tau to z in [-1, 1]
  auto tauToZ = [T](float tau) -> float {
    return 2.0f * std::sqrt(tau / T) - 1.0f;
  };

  // Function to get boundary value at tau
  auto B = [xmax, &interp, tauToZ](float tau) -> float {
    if (tau <= 0.0f)
      return xmax;

    const float z = tauToZ(tau);
    return xmax * std::exp(-std::sqrt(std::max(0.0f, (*interp)(z, true))));
  };

  // Function to map boundary value to interpolation y-value
  auto h = [xmax](float fv) -> float {
    return std::pow(std::log(fv / xmax), 2);
  };

  // Create fixed point evaluator
  auto evaluator =
      createFixedPointEvaluatorFloat('A', K, r, q, vol, B, fpIntegrator);

  // Perform fixed point iterations
  // First is a Jacobi-Newton step
  for (size_t k = 0; k < 1; ++k) {
    for (size_t i = 1; i < num_nodes;
         ++i) { // Skip first node (corresponds to tau=0)
      const float z = nodes[i];
      const float tau = T * std::pow(0.5f * (1.0f + z), 2);
      const float b = B(tau);

      const auto [N, D, fv] = evaluator->evaluate(tau, b);

      if (tau < 1e-5f) {
        y[i] = h(fv);
      } else {
        const auto [Nd, Dd] = evaluator->derivatives(tau, b);

        // Newton step
        const float fd =
            K * std::exp(-(r - q) * tau) * (Nd / D - Dd * N / (D * D));
        const float b_new = b - (fv - b) / (fd - 1.0f);

        y[i] = h(b_new);
      }
    }

    // Update interpolation
    interp->updateValues(y);
  }

  // Remaining iterations are standard fixed point
  for (size_t k = 1; k < num_iterations; ++k) {
    for (size_t i = 1; i < num_nodes;
         ++i) { // Skip first node (corresponds to tau=0)
      const float z = nodes[i];
      const float tau = T * std::pow(0.5f * (1.0f + z), 2);
      const float b = B(tau);

      const auto [N, D, fv] = evaluator->evaluate(tau, b);

      y[i] = h(fv);
    }

    // Update interpolation
    interp->updateValues(y);
  }

  return interp;
}

float AmericanPutFloat::approximatePrice(float S, float K, float r, float q,
                                         float vol, float T) const {
  // Barone-Adesi-Whaley approximation for American put options

  // Calculate European put price
  EuropeanPutFloat euro_put;
  float euro_price = euro_put.calculatePrice(S, K, r, q, vol, T);

  // If deep out-of-the-money or very short time to expiry, use European price
  if (S > 1.5f * K || T < 1.0f / 365.0f) {
    return euro_price;
  }

  // Compute critical parameters for the approximation
  const float b = r - q;
  const float M = 2.0f * r / (vol * vol);
  const float N = 2.0f * b / (vol * vol);
  const float K0 = 2.0f / (1.0f + std::sqrt(1.0f + 4.0f * M));
  const float h = -(b * T + 2.0f * vol * std::sqrt(T)) * K / (K - S);

  const float q2 =
      (-(N - 1.0f) + std::sqrt((N - 1.0f) * (N - 1.0f) + 4.0f * M)) / 2.0f;
  const float alpha =
      (K / q2) *
      (1.0f - std::exp((b - r) * T) *
                  num::fast_normal_cdf(-euro_put.d1(S, K, r, q, vol, T)));

  // Compute early exercise boundary
  const float S_star = K / (1.0f + K0);

  // If spot is below the early exercise boundary, then exercise
  if (S <= S_star) {
    return std::max(K - S, 0.0f);
  }

  // Otherwise, add the early exercise premium to the European price
  return euro_price + alpha * std::pow(S / S_star, -q2);
}

std::vector<float> AmericanPutFloat::batchApproximatePrice(
    float S, const std::vector<float> &strikes, float r, float q, float vol,
    float T) const {

  std::vector<float> results(strikes.size());

  // Process in groups of 8 using AVX2
  size_t i = 0;
  for (; i + 7 < strikes.size(); i += 8) {
    // Load 8 strikes
    __m256 K_vec = _mm256_loadu_ps(&strikes[i]);
    __m256 S_vec = _mm256_set1_ps(S);
    __m256 r_vec = _mm256_set1_ps(r);
    __m256 q_vec = _mm256_set1_ps(q);
    __m256 vol_vec = _mm256_set1_ps(vol);
    __m256 T_vec = _mm256_set1_ps(T);

    // Calculate Barone-Adesi-Whaley prices using SIMD
    __m256 prices = opt::SimdOpsFloat::americanPut(S_vec, K_vec, r_vec, q_vec,
                                                   vol_vec, T_vec);

    // Store results
    _mm256_storeu_ps(&results[i], prices);
  }

  // Handle remaining options
  for (; i < strikes.size(); ++i) {
    results[i] = approximatePrice(S, strikes[i], r, q, vol, T);
  }

  return results;
}

// AmericanCallFloat implementation
AmericanCallFloat::AmericanCallFloat(
    std::shared_ptr<num::IntegratorFloat> integrator)
    : AmericanOptionFloat(integrator) {}

float AmericanCallFloat::xMax(float K, float r, float q) const {
  // For call options, the early exercise boundary is different
  if (q > 0.0f && r >= 0.0f)
    return K * std::max(1.0f, r / q);
  else if (q <= 0.0f && r >= 0.0f)
    return std::numeric_limits<float>::
        infinity(); // Effectively infinite, early exercise never optimal
  else if (q >= r && r < 0.0f)
    return std::numeric_limits<float>::infinity(); // European case
  else if (q < r && r < 0.0f)
    return K; // Double boundary case
  else
    throw std::runtime_error("Internal error in xMax calculation for call");
}

float AmericanCallFloat::calculateEarlyExercisePremium(
    float S, float K, float r, float q, float vol, float T,
    const std::shared_ptr<num::ChebyshevInterpolationFloat> &boundary) const {

  if (!boundary) {
    throw std::invalid_argument(
        "AmericanCallFloat: Boundary interpolation cannot be null");
  }

  // If zero dividend, calls should never be exercised early
  if (q <= 0.0f) {
    return 0.0f;
  }

  const float xmax = xMax(K, r, q);

  // For infinite boundary (no early exercise), return zero premium
  if (std::isinf(xmax) || xmax > 1e6f) {
    return 0.0f;
  }

  // Function to map tau to z in [-1, 1]
  auto tauToZ = [T](float tau) -> float {
    return 2.0f * std::sqrt(tau / T) - 1.0f;
  };

  // Function to get boundary value at tau
  auto B = [xmax, boundary, tauToZ](float tau) -> float {
    if (tau <= 0.0f)
      return xmax;

    const float z = tauToZ(tau);
    return xmax * std::exp(-std::sqrt(std::max(0.0f, (*boundary)(z, true))));
  };

  // Integrand for call early exercise premium
  auto integrand = [S, K, r, q, vol, B](float z) -> float {
    const float t = z * z; // tau = z^2
    const float b_t = B(t);

    if (b_t <= 0.0f || std::isinf(b_t))
      return 0.0f;

    const float dr = std::exp(-r * t);
    const float dq = std::exp(-q * t);
    const float v = vol * std::sqrt(t);

    if (v < 1e-5f) {
      if (std::abs(b_t * dq - K * dr) < 1e-5f)
        return z * (q * b_t * dq - r * K * dr);
      else if (b_t * dq > K * dr)
        return 2.0f * z * (q * b_t * dq - r * K * dr);
      else
        return 0.0f;
    }

    const float dp = (std::log(b_t * dq / (K * dr)) / v) + 0.5f * v;

    return 2.0f * z *
           (q * b_t * dq * (0.5f * (1.0f + num::fast_erf(dp / 1.414213562f))) -
            r * K * dr *
                (0.5f * (1.0f + num::fast_erf((dp - v) / 1.414213562f))));
  };

  // Integrate to get early exercise premium
  return integrator_->integrate(integrand, 0.0f, std::sqrt(T));
}

std::shared_ptr<num::ChebyshevInterpolationFloat>
AmericanCallFloat::calculateExerciseBoundary(
    float S, float K, float r, float q, float vol, float T, size_t num_nodes,
    size_t num_iterations,
    std::shared_ptr<num::IntegratorFloat> fpIntegrator) const {

  if (!fpIntegrator) {
    throw std::invalid_argument(
        "AmericanCallFloat: Fixed point integrator cannot be null");
  }

  const float xmax = xMax(K, r, q);

  // For infinite boundary (no early exercise), return an empty boundary
  if (std::isinf(xmax) || xmax > 1e6f) {
    std::vector<float> nodes(num_nodes);
    std::vector<float> y(num_nodes, 0.0f);

    for (size_t i = 0; i < num_nodes; ++i) {
      nodes[i] = std::cos(3.14159265358979323846f * i / (num_nodes - 1));
    }

    return num::createChebyshevInterpolationFloat(nodes, y, num::SECOND_KIND,
                                                  -1.0f, 1.0f);
  }

  // Initialize interpolation nodes
  std::vector<float> nodes(num_nodes);
  std::vector<float> y(num_nodes, 0.0f); // Boundary function values at nodes

  // Chebyshev nodes of the second kind in [-1, 1]
  for (size_t i = 0; i < num_nodes; ++i) {
    nodes[i] = std::cos(3.14159265358979323846f * i /
                        (num_nodes - 1)); // x_i in [-1, 1]
  }

  // Create initial interpolation
  auto interp = num::createChebyshevInterpolationFloat(
      nodes, y, num::SECOND_KIND, -1.0f, 1.0f);

  // Function to map tau to z in [-1, 1]
  auto tauToZ = [T](float tau) -> float {
    return 2.0f * std::sqrt(tau / T) - 1.0f;
  };

  // Function to get boundary value at tau
  auto B = [xmax, &interp, tauToZ](float tau) -> float {
    if (tau <= 0.0f)
      return xmax;

    const float z = tauToZ(tau);
    return xmax * std::exp(-std::sqrt(std::max(0.0f, (*interp)(z, true))));
  };

  // Function to map boundary value to interpolation y-value
  auto h = [xmax](float fv) -> float {
    return std::pow(std::log(fv / xmax), 2);
  };

  // For calls, we swap r and q in the fixed point equation (put-call symmetry)
  auto evaluator =
      createFixedPointEvaluatorFloat('A', K, q, r, vol, B, fpIntegrator);

  // Perform fixed point iterations
  // First is a Jacobi-Newton step
  for (size_t k = 0; k < 1; ++k) {
    for (size_t i = 1; i < num_nodes;
         ++i) { // Skip first node (corresponds to tau=0)
      const float z = nodes[i];
      const float tau = T * std::pow(0.5f * (1.0f + z), 2);
      const float b = B(tau);

      const auto [N, D, fv] = evaluator->evaluate(tau, b);

      if (tau < 1e-5f) {
        y[i] = h(fv);
      } else {
        const auto [Nd, Dd] = evaluator->derivatives(tau, b);

        // Newton step
        const float fd =
            K * std::exp(-(q - r) * tau) * (Nd / D - Dd * N / (D * D));
        const float b_new = b - (fv - b) / (fd - 1.0f);

        y[i] = h(b_new);
      }
    }

    // Update interpolation
    interp->updateValues(y);
  }

  // Remaining iterations are standard fixed point
  for (size_t k = 1; k < num_iterations; ++k) {
    for (size_t i = 1; i < num_nodes;
         ++i) { // Skip first node (corresponds to tau=0)
      const float z = nodes[i];
      const float tau = T * std::pow(0.5f * (1.0f + z), 2);
      const float b = B(tau);

      const auto [N, D, fv] = evaluator->evaluate(tau, b);

      y[i] = h(fv);
    }

    // Update interpolation
    interp->updateValues(y);
  }

  return interp;
}

float AmericanCallFloat::approximatePrice(float S, float K, float r, float q,
                                          float vol, float T) const {
  // For zero dividend, use European price
  if (q <= 0.0f) {
    EuropeanCallFloat euro_call;
    return euro_call.calculatePrice(S, K, r, q, vol, T);
  }

  // Barone-Adesi-Whaley approximation for American call options

  // Calculate European call price
  EuropeanCallFloat euro_call;
  float euro_price = euro_call.calculatePrice(S, K, r, q, vol, T);

  // If deep out-of-the-money or very short time to expiry, use European price
  if (S < 0.7f * K || T < 1.0f / 365.0f) {
    return euro_price;
  }

  // Compute critical parameters for the approximation
  const float b = r - q;
  const float M = 2.0f * r / (vol * vol);
  const float N = 2.0f * b / (vol * vol);
  const float K0 = 2.0f / (1.0f + std::sqrt(1.0f + 4.0f * M));
  const float h = (b * T + 2.0f * vol * std::sqrt(T)) * K / (S - K);

  const float q1 =
      ((N - 1.0f) + std::sqrt((N - 1.0f) * (N - 1.0f) + 4.0f * M)) / 2.0f;
  const float alpha =
      -(S / q1) *
      (1.0f - std::exp((b - r) * T) *
                  num::fast_normal_cdf(euro_call.d1(S, K, r, q, vol, T)));

  // Compute early exercise boundary
  const float S_star = K * (1.0f + 1.0f / K0);

  // If spot is above the early exercise boundary, then exercise
  if (S >= S_star) {
    return std::max(S - K, 0.0f);
  }

  // Otherwise, add the early exercise premium to the European price
  return euro_price + alpha * std::pow(S / S_star, q1);
}

std::vector<float> AmericanCallFloat::batchApproximatePrice(
    float S, const std::vector<float> &strikes, float r, float q, float vol,
    float T) const {

  std::vector<float> results(strikes.size());

  // Process in groups of 8 using AVX2
  size_t i = 0;
  for (; i + 7 < strikes.size(); i += 8) {
    // Load 8 strikes
    __m256 K_vec = _mm256_loadu_ps(&strikes[i]);
    __m256 S_vec = _mm256_set1_ps(S);
    __m256 r_vec = _mm256_set1_ps(r);
    __m256 q_vec = _mm256_set1_ps(q);
    __m256 vol_vec = _mm256_set1_ps(vol);
    __m256 T_vec = _mm256_set1_ps(T);

    // Calculate Barone-Adesi-Whaley prices using SIMD
    __m256 prices = opt::SimdOpsFloat::americanCall(S_vec, K_vec, r_vec, q_vec,
                                                    vol_vec, T_vec);

    // Store results
    _mm256_storeu_ps(&results[i], prices);
  }

  // Handle remaining options
  for (; i < strikes.size(); ++i) {
    results[i] = approximatePrice(S, strikes[i], r, q, vol, T);
  }

  return results;
}

// FixedPointEvaluatorFloat implementation
FixedPointEvaluatorFloat::FixedPointEvaluatorFloat(
    float K, float r, float q, float vol, const std::function<float(float)> &B,
    std::shared_ptr<num::IntegratorFloat> integrator)
    : K_(K), r_(r), q_(q), vol_(vol), vol2_(vol * vol), B_(B),
      integrator_(integrator) {}

std::pair<float, float> FixedPointEvaluatorFloat::d(float t, float z) const {
  if (t <= 0.0f || z <= 0.0f) {
    return {-std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};
  }

  const float v = vol_ * std::sqrt(t);
  const float m = (std::log(z) + (r_ - q_) * t) / v + 0.5f * v;

  return {m, m - v};
}

float FixedPointEvaluatorFloat::normalCDF(float x) const {
  // Using optimized error function implementation
  return 0.5f * (1.0f + num::fast_erf(x / 1.414213562f));
}

float FixedPointEvaluatorFloat::normalPDF(float x) const {
  return num::fast_normal_pdf(x);
}

// EquationAFloat implementation
EquationAFloat::EquationAFloat(float K, float r, float q, float vol,
                               const std::function<float(float)> &B,
                               std::shared_ptr<num::IntegratorFloat> integrator)
    : FixedPointEvaluatorFloat(K, r, q, vol, B, integrator) {}

std::tuple<float, float, float> EquationAFloat::evaluate(float tau,
                                                         float b) const {
  float N, D;
  const float eps = 1e-5f;

  if (tau < eps * eps) {
    // For very small tau, use limit values
    if (std::abs(b - K_) < eps) {
      N = D = 0.5f;
    } else {
      N = D = (b > K_) ? 1.0f : 0.0f;
    }
  } else {
    const float stv = std::sqrt(tau) / vol_;

    // Integrate to get K1 + K2
    auto K12_integrand = [&](float y) -> float {
      const float m = 0.25f * tau * (1.0f + y) * (1.0f + y);
      const float df = std::exp(q_ * tau - q_ * m);

      if (y <= 5.0f * eps - 1.0f) {
        if (std::abs(b - B_(tau - m)) < eps) {
          return df * stv / (std::sqrt(2.0f * 3.14159265358979323846f));
        } else {
          return 0.0f;
        }
      } else {
        const auto dp = d(m, b / B_(tau - m)).first;
        return df *
               (0.5f * tau * (y + 1.0f) * normalCDF(dp) + stv * normalPDF(dp));
      }
    };

    // Integrate to get K3
    auto K3_integrand = [&](float y) -> float {
      const float m = 0.25f * tau * (1.0f + y) * (1.0f + y);
      const float df = std::exp(r_ * tau - r_ * m);

      if (y <= 5.0f * eps - 1.0f) {
        if (std::abs(b - B_(tau - m)) < eps) {
          return df * stv / (std::sqrt(2.0f * 3.14159265358979323846f));
        } else {
          return 0.0f;
        }
      } else {
        return df * stv * normalPDF(d(m, b / B_(tau - m)).second);
      }
    };

    float K12 = integrator_->integrate(K12_integrand, -1.0f, 1.0f);
    float K3 = integrator_->integrate(K3_integrand, -1.0f, 1.0f);

    const auto dpm = d(tau, b / K_);
    N = normalPDF(dpm.second) / vol_ / std::sqrt(tau) + r_ * K3;
    D = normalPDF(dpm.first) / vol_ / std::sqrt(tau) + normalCDF(dpm.first) +
        q_ * K12;
  }

  // Calculate function value
  const float alpha = K_ * std::exp(-(r_ - q_) * tau);
  float fv;

  if (tau < eps * eps) {
    if (std::abs(b - K_) < eps || b > K_) {
      fv = alpha;
    } else {
      if (std::abs(q_) < eps) {
        fv = alpha * r_ * ((q_ < 0.0f) ? -1.0f : 1.0f) / eps;
      } else {
        fv = alpha * r_ / q_;
      }
    }
  } else {
    fv = alpha * N / D;
  }

  return {N, D, fv};
}

std::pair<float, float> EquationAFloat::derivatives(float tau, float b) const {
  float Nd, Dd;
  const float eps = 1e-5f;

  if (tau < eps * eps) {
    if (std::abs(b - K_) < eps) {
      const float sqTau = std::sqrt(tau);
      Dd = 2.0f / 1.772453851f *
           (-(0.5f * vol2_ + r_ - q_) / (b * vol_ * vol2_ * sqTau) +
            1.0f / (b * vol_ * sqTau));
      Nd = 2.0f / 1.772453851f * (-0.5f * vol2_ + r_ - q_) /
           (b * vol_ * vol2_ * sqTau);
    } else {
      Dd = Nd = 0.0f;
    }
  } else {
    const auto dpm = d(tau, b / K_);

    Dd = -normalPDF(dpm.first) * dpm.first / (b * vol2_ * tau) +
         normalPDF(dpm.first) / (b * vol_ * std::sqrt(tau));
    Nd = -normalPDF(dpm.second) * dpm.second / (b * vol2_ * tau);
  }

  return {Nd, Dd};
}

// EquationBFloat implementation
EquationBFloat::EquationBFloat(float K, float r, float q, float vol,
                               const std::function<float(float)> &B,
                               std::shared_ptr<num::IntegratorFloat> integrator)
    : FixedPointEvaluatorFloat(K, r, q, vol, B, integrator) {}

std::tuple<float, float, float> EquationBFloat::evaluate(float tau,
                                                         float b) const {
  float N, D;
  const float eps = 1e-5f;

  if (tau < eps * eps) {
    // For very small tau, use limit values
    if (std::abs(b - K_) < eps) {
      N = D = 0.5f;
    } else if (b < K_) {
      N = D = 0.0f;
    } else {
      N = D = 1.0f;
    }
  } else {
    // Integrate for N and D
    auto N_integrand = [&](float u) -> float {
      const float df = std::exp(r_ * u);
      if (u >= tau * (1.0f - 5.0f * eps)) {
        if (std::abs(b - B_(u)) < eps) {
          return 0.5f * df;
        } else {
          return df * ((b < B_(u)) ? 0.0f : 1.0f);
        }
      } else {
        return df * normalCDF(d(tau - u, b / B_(u)).second);
      }
    };

    auto D_integrand = [&](float u) -> float {
      const float df = std::exp(q_ * u);
      if (u >= tau * (1.0f - 5.0f * eps)) {
        if (std::abs(b - B_(u)) < eps) {
          return 0.5f * df;
        } else {
          return df * ((b < B_(u)) ? 0.0f : 1.0f);
        }
      } else {
        return df * normalCDF(d(tau - u, b / B_(u)).first);
      }
    };

    float ni = integrator_->integrate(N_integrand, 0.0f, tau);
    float di = integrator_->integrate(D_integrand, 0.0f, tau);

    const auto dpm = d(tau, b / K_);

    N = normalCDF(dpm.second) + r_ * ni;
    D = normalCDF(dpm.first) + q_ * di;
  }

  // Calculate function value
  const float alpha = K_ * std::exp(-(r_ - q_) * tau);
  float fv;

  if (tau < eps * eps) {
    if (std::abs(b - K_) < eps || b > K_) {
      fv = alpha;
    } else {
      if (std::abs(q_) < eps) {
        fv = alpha * r_ * ((q_ < 0.0f) ? -1.0f : 1.0f) / eps;
      } else {
        fv = alpha * r_ / q_;
      }
    }
  } else {
    fv = alpha * N / D;
  }

  return {N, D, fv};
}

std::pair<float, float> EquationBFloat::derivatives(float tau, float b) const {
  const float eps = 1e-5f;

  if (tau < eps * eps) {
    return {0.0f, 0.0f};
  }

  const auto dpm = d(tau, b / K_);

  return {normalPDF(dpm.second) / (b * vol_ * std::sqrt(tau)),
          normalPDF(dpm.first) / (b * vol_ * std::sqrt(tau))};
}

std::shared_ptr<FixedPointEvaluatorFloat> createFixedPointEvaluatorFloat(
    char equation, float K, float r, float q, float vol,
    const std::function<float(float)> &B,
    std::shared_ptr<num::IntegratorFloat> integrator) {

  if (equation == 'A') {
    return std::make_shared<EquationAFloat>(K, r, q, vol, B, integrator);
  } else if (equation == 'B') {
    return std::make_shared<EquationBFloat>(K, r, q, vol, B, integrator);
  } else {
    // Auto-select equation based on r and q
    if (std::abs(r - q) < 0.001f) {
      return std::make_shared<EquationAFloat>(K, r, q, vol, B, integrator);
    } else {
      return std::make_shared<EquationBFloat>(K, r, q, vol, B, integrator);
    }
  }
}

} // namespace mod
} // namespace alo
} // namespace engine