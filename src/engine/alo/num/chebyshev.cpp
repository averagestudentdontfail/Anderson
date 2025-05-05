#include "chebyshev.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine {
namespace alo {
namespace num {

ChebyshevInterpolationDouble::ChebyshevInterpolationDouble(
    size_t num_points, const std::function<double(double)> &func,
    ChebyshevKind kind, double domain_start, double domain_end)
    : num_points_(num_points), kind_(kind), domain_start_(domain_start),
      domain_end_(domain_end) {

  if (num_points_ < 2) {
    throw std::invalid_argument(
        "ChebyshevInterpolationDouble requires at least 2 points");
  }

  if (domain_end_ <= domain_start_) {
    throw std::invalid_argument("Domain end must be greater than domain start");
  }

  // Initialize nodes
  initializeNodes();

  // Evaluate function at nodes
  values_.resize(num_points_);
  for (size_t i = 0; i < num_points_; ++i) {
    double x = mapFromStandardDomain(nodes_[i]);
    values_[i] = func(x);
  }

  // Compute Chebyshev coefficients
  computeCoefficients();
}

ChebyshevInterpolationDouble::ChebyshevInterpolationDouble(
    const std::vector<double> &nodes, const std::vector<double> &values,
    ChebyshevKind kind, double domain_start, double domain_end)
    : num_points_(nodes.size()), kind_(kind), domain_start_(domain_start),
      domain_end_(domain_end), nodes_(nodes), values_(values) {

  if (nodes_.size() != values_.size()) {
    throw std::invalid_argument("Nodes and values must have the same size");
  }

  if (nodes_.size() < 2) {
    throw std::invalid_argument(
        "ChebyshevInterpolationDouble requires at least 2 points");
  }

  if (domain_end_ <= domain_start_) {
    throw std::invalid_argument("Domain end must be greater than domain start");
  }

  // Compute Chebyshev coefficients
  computeCoefficients();
}

void ChebyshevInterpolationDouble::initializeNodes() {
  nodes_.resize(num_points_);

  if (kind_ == FIRST_KIND) {
    // Chebyshev nodes of the first kind: x_j = cos((2j+1)π/(2n)), j=0,...,n-1
    for (size_t j = 0; j < num_points_; ++j) {
      nodes_[j] = std::cos(M_PI * (2.0 * j + 1.0) / (2.0 * num_points_));
    }
  } else { // SECOND_KIND
    // Chebyshev nodes of the second kind: x_j = cos(jπ/(n-1)), j=0,...,n-1
    for (size_t j = 0; j < num_points_; ++j) {
      nodes_[j] = std::cos(M_PI * j / (num_points_ - 1));
    }
  }
}

void ChebyshevInterpolationDouble::computeCoefficients() {
  coefficients_.resize(num_points_);

  if (kind_ == FIRST_KIND) {
    // For first kind, use DCT-II (Discrete Cosine Transform Type II)
    for (size_t k = 0; k < num_points_; ++k) {
      double sum = 0.0;
      for (size_t j = 0; j < num_points_; ++j) {
        sum += values_[j] *
               std::cos(M_PI * k * (2.0 * j + 1.0) / (2.0 * num_points_));
      }
      coefficients_[k] = (2.0 / num_points_) * sum;
    }
    // Adjust the first coefficient
    coefficients_[0] *= 0.5;
  } else { // SECOND_KIND
    // For second kind nodes, use a simplified DCT approach
    for (size_t k = 0; k < num_points_; ++k) {
      double sum = 0.0;
      for (size_t j = 0; j < num_points_; ++j) {
        // For second kind nodes, we can use a simplified DCT
        sum += values_[j] * std::cos(M_PI * k * j / (num_points_ - 1));
      }
      if (k == 0 || k == num_points_ - 1) {
        coefficients_[k] = sum / (num_points_ - 1);
      } else {
        coefficients_[k] = 2.0 * sum / (num_points_ - 1);
      }
    }
  }
}

double ChebyshevInterpolationDouble::operator()(double x,
                                                bool extrapolate) const {
  // Map x to standard domain [-1, 1]
  double t = mapToStandardDomain(x);

  // Check if x is within the domain
  if (!extrapolate && (t < -1.0 || t > 1.0)) {
    throw std::domain_error("Evaluation point outside interpolation domain");
  }

  // Apply Clenshaw's algorithm to evaluate the Chebyshev series
  if (kind_ == FIRST_KIND) {
    double b_kp2 = 0.0;
    double b_kp1 = 0.0;
    double b_k;

    for (int k = static_cast<int>(num_points_) - 1; k >= 0; --k) {
      b_k = coefficients_[k] + 2.0 * t * b_kp1 - b_kp2;
      b_kp2 = b_kp1;
      b_kp1 = b_k;
    }

    return b_kp1 - t * b_kp2;
  } else { // SECOND_KIND
    // Direct evaluation using Horner's method
    double result = 0.0;
    for (int k = num_points_ - 1; k >= 0; --k) {
      result = result * t + coefficients_[k];
    }
    return result;
  }
}

void ChebyshevInterpolationDouble::updateValues(
    const std::vector<double> &values) {
  if (values.size() != num_points_) {
    throw std::invalid_argument(
        "Number of values must match the number of nodes");
  }

  values_ = values;
  computeCoefficients();
}

double ChebyshevInterpolationDouble::mapToStandardDomain(double x) const {
  return 2.0 * (x - domain_start_) / (domain_end_ - domain_start_) - 1.0;
}

double ChebyshevInterpolationDouble::mapFromStandardDomain(double t) const {
  return 0.5 *
         ((domain_end_ - domain_start_) * t + domain_end_ + domain_start_);
}

double
ChebyshevInterpolationDouble::evaluateChebyshev(int n, double x,
                                                ChebyshevKind kind) const {
  if (n == 0)
    return 1.0;
  if (n == 1)
    return x;

  // Use recurrence relation to compute higher-order polynomials
  double T_n_minus_2 = 1.0; // T_0(x)
  double T_n_minus_1 = x;   // T_1(x)
  double T_n = 0.0;

  for (int i = 2; i <= n; ++i) {
    T_n = 2.0 * x * T_n_minus_1 - T_n_minus_2;
    T_n_minus_2 = T_n_minus_1;
    T_n_minus_1 = T_n;
  }

  if (kind == FIRST_KIND) {
    return T_n;
  } else { // SECOND_KIND
    // Convert to second kind Chebyshev polynomial
    if (n == 0)
      return 1.0;
    return (1.0 - x * x) * T_n;
  }
}

SimdChebyshevInterpolationDouble::SimdChebyshevInterpolationDouble(
    const ChebyshevInterpolationDouble &interp)
    : interp_(interp) {}

void SimdChebyshevInterpolationDouble::evaluate(const std::vector<double> &x,
                                                std::vector<double> &y,
                                                bool extrapolate) const {

  if (y.size() < x.size()) {
    y.resize(x.size());
  }

  // Check if SIMD is available and beneficial
  if (x.size() >= 4) {
    // Process in groups of 4 using SIMD
    size_t i = 0;
    for (; i + 3 < x.size(); i += 4) {
      std::array<double, 4> x_block = {x[i], x[i + 1], x[i + 2], x[i + 3]};
      std::array<double, 4> y_block = evaluate4(x_block, extrapolate);

      for (size_t j = 0; j < 4; ++j) {
        y[i + j] = y_block[j];
      }
    }

    // Handle remaining points
    for (; i < x.size(); ++i) {
      y[i] = interp_(x[i], extrapolate);
    }
  } else {
    // For small input, just use scalar computation
    for (size_t i = 0; i < x.size(); ++i) {
      y[i] = interp_(x[i], extrapolate);
    }
  }
}

std::array<double, 4>
SimdChebyshevInterpolationDouble::evaluate4(const std::array<double, 4> &x,
                                            bool extrapolate) const {

  std::array<double, 4> result;

  // Map input points to standard domain
  std::array<double, 4> t;
  const double domain_start = interp_.getDomainStart();
  const double domain_end = interp_.getDomainEnd();

  for (size_t i = 0; i < 4; ++i) {
    t[i] = 2.0 * (x[i] - domain_start) / (domain_end - domain_start) - 1.0;

    // Check domain bounds if not extrapolating
    if (!extrapolate && (t[i] < -1.0 || t[i] > 1.0)) {
      throw std::domain_error("Evaluation point outside interpolation domain");
    }
  }

  // Get interpolation details
  const auto &coeffs = interp_.getCoefficients();
  ChebyshevKind kind = interp_.getKind();

  // Load coefficients into SIMD registers (if enough coefficients)
  if (coeffs.size() >= 4 && kind == SECOND_KIND) {
    // Use SIMD for Horner's method (second kind)
    __m256d result_vec = _mm256_setzero_pd();

    for (int k = coeffs.size() - 1; k >= 0; --k) {
      __m256d coeff_vec = _mm256_set1_pd(coeffs[k]);
      __m256d t_vec = _mm256_loadu_pd(t.data());

      // result = result * t + coeff
      result_vec = _mm256_fmadd_pd(result_vec, t_vec, coeff_vec);
    }

    _mm256_storeu_pd(result.data(), result_vec);
  } else {
    // Fall back to scalar computation for small coefficient sets or first kind
    for (int i = 0; i < 4; ++i) {
      if (kind == FIRST_KIND) {
        // Apply Clenshaw's algorithm
        double b_kp2 = 0.0;
        double b_kp1 = 0.0;
        double b_k;

        for (int k = static_cast<int>(coeffs.size()) - 1; k >= 0; --k) {
          b_k = coeffs[k] + 2.0 * t[i] * b_kp1 - b_kp2;
          b_kp2 = b_kp1;
          b_kp1 = b_k;
        }

        result[i] = b_kp1 - t[i] * b_kp2;
      } else { // SECOND_KIND
        // Apply Horner's method
        double r = 0.0;
        for (int k = coeffs.size() - 1; k >= 0; --k) {
          r = r * t[i] + coeffs[k];
        }
        result[i] = r;
      }
    }
  }

  return result;
}

std::shared_ptr<ChebyshevInterpolationDouble>
createChebyshevInterpolationDouble(size_t num_points,
                                   const std::function<double(double)> &func,
                                   ChebyshevKind kind, double domain_start,
                                   double domain_end) {

  return std::make_shared<ChebyshevInterpolationDouble>(
      num_points, func, kind, domain_start, domain_end);
}

std::shared_ptr<ChebyshevInterpolationDouble>
createChebyshevInterpolationDouble(const std::vector<double> &nodes,
                                   const std::vector<double> &values,
                                   ChebyshevKind kind, double domain_start,
                                   double domain_end) {

  return std::make_shared<ChebyshevInterpolationDouble>(
      nodes, values, kind, domain_start, domain_end);
}

ChebyshevInterpolationSingle::ChebyshevInterpolationSingle(
    size_t num_points, const std::function<float(float)> &func,
    ChebyshevKind kind, float domain_start, float domain_end)
    : num_points_(num_points), kind_(kind), domain_start_(domain_start),
      domain_end_(domain_end) {

  if (num_points_ < 2) {
    throw std::invalid_argument(
        "ChebyshevInterpolationSingle requires at least 2 points");
  }

  if (domain_end_ <= domain_start_) {
    throw std::invalid_argument("Domain end must be greater than domain start");
  }

  // Initialize nodes
  initializeNodes();

  // Evaluate function at nodes
  values_.resize(num_points_);
  for (size_t i = 0; i < num_points_; ++i) {
    float x = mapFromStandardDomain(nodes_[i]);
    values_[i] = func(x);
  }

  // Compute Chebyshev coefficients
  computeCoefficients();
}

ChebyshevInterpolationSingle::ChebyshevInterpolationSingle(
    const std::vector<float> &nodes, const std::vector<float> &values,
    ChebyshevKind kind, float domain_start, float domain_end)
    : num_points_(nodes.size()), kind_(kind), domain_start_(domain_start),
      domain_end_(domain_end), nodes_(nodes), values_(values) {

  if (nodes_.size() != values_.size()) {
    throw std::invalid_argument("Nodes and values must have the same size");
  }

  if (nodes_.size() < 2) {
    throw std::invalid_argument(
        "ChebyshevInterpolationSingle requires at least 2 points");
  }

  if (domain_end_ <= domain_start_) {
    throw std::invalid_argument("Domain end must be greater than domain start");
  }

  // Compute Chebyshev coefficients
  computeCoefficients();
}

void ChebyshevInterpolationSingle::initializeNodes() {
  nodes_.resize(num_points_);

  if (kind_ == FIRST_KIND) {
    // Chebyshev nodes of the first kind: x_j = cos((2j+1)π/(2n)), j=0,...,n-1
    for (size_t j = 0; j < num_points_; ++j) {
      nodes_[j] = std::cos(M_PI * (2.0f * j + 1.0f) / (2.0f * num_points_));
    }
  } else { // SECOND_KIND
    // Chebyshev nodes of the second kind: x_j = cos(jπ/(n-1)), j=0,...,n-1
    for (size_t j = 0; j < num_points_; ++j) {
      nodes_[j] = std::cos(M_PI * j / (num_points_ - 1));
    }
  }
}

void ChebyshevInterpolationSingle::computeCoefficients() {
  coefficients_.resize(num_points_);

  if (kind_ == FIRST_KIND) {
    // For first kind, use DCT-II (Discrete Cosine Transform Type II)
    for (size_t k = 0; k < num_points_; ++k) {
      float sum = 0.0f;
      for (size_t j = 0; j < num_points_; ++j) {
        sum += values_[j] *
               std::cos(M_PI * k * (2.0f * j + 1.0f) / (2.0f * num_points_));
      }
      coefficients_[k] = (2.0f / num_points_) * sum;
    }
    // Adjust the first coefficient
    coefficients_[0] *= 0.5f;
  } else { // SECOND_KIND
    // For second kind nodes, use a simplified DCT approach
    for (size_t k = 0; k < num_points_; ++k) {
      float sum = 0.0f;
      for (size_t j = 0; j < num_points_; ++j) {
        // For second kind nodes, we can use a simplified DCT
        sum += values_[j] * std::cos(M_PI * k * j / (num_points_ - 1));
      }
      if (k == 0 || k == num_points_ - 1) {
        coefficients_[k] = sum / (num_points_ - 1);
      } else {
        coefficients_[k] = 2.0f * sum / (num_points_ - 1);
      }
    }
  }
}

float ChebyshevInterpolationSingle::operator()(float x,
                                               bool extrapolate) const {
  // Map x to standard domain [-1, 1]
  float t = mapToStandardDomain(x);

  // Check if x is within the domain
  if (!extrapolate && (t < -1.0f || t > 1.0f)) {
    throw std::domain_error("Evaluation point outside interpolation domain");
  }

  // Apply Clenshaw's algorithm to evaluate the Chebyshev series
  if (kind_ == FIRST_KIND) {
    float b_kp2 = 0.0f;
    float b_kp1 = 0.0f;
    float b_k;

    for (int k = static_cast<int>(num_points_) - 1; k >= 0; --k) {
      b_k = coefficients_[k] + 2.0f * t * b_kp1 - b_kp2;
      b_kp2 = b_kp1;
      b_kp1 = b_k;
    }

    return b_kp1 - t * b_kp2;
  } else { // SECOND_KIND
    // Direct evaluation using Horner's method
    float result = 0.0f;
    for (int k = num_points_ - 1; k >= 0; --k) {
      result = result * t + coefficients_[k];
    }
    return result;
  }
}

void ChebyshevInterpolationSingle::updateValues(
    const std::vector<float> &values) {
  if (values.size() != num_points_) {
    throw std::invalid_argument(
        "Number of values must match the number of nodes");
  }

  values_ = values;
  computeCoefficients();
}

float ChebyshevInterpolationSingle::mapToStandardDomain(float x) const {
  return 2.0f * (x - domain_start_) / (domain_end_ - domain_start_) - 1.0f;
}

float ChebyshevInterpolationSingle::mapFromStandardDomain(float t) const {
  return 0.5f *
         ((domain_end_ - domain_start_) * t + domain_end_ + domain_start_);
}

float ChebyshevInterpolationSingle::evaluateChebyshev(
    int n, float x, ChebyshevKind kind) const {
  if (n == 0)
    return 1.0f;
  if (n == 1)
    return x;

  // Use recurrence relation to compute higher-order polynomials
  float T_n_minus_2 = 1.0f; // T_0(x)
  float T_n_minus_1 = x;    // T_1(x)
  float T_n = 0.0f;

  for (int i = 2; i <= n; ++i) {
    T_n = 2.0f * x * T_n_minus_1 - T_n_minus_2;
    T_n_minus_2 = T_n_minus_1;
    T_n_minus_1 = T_n;
  }

  if (kind == FIRST_KIND) {
    return T_n;
  } else { // SECOND_KIND
    // Convert to second kind Chebyshev polynomial
    if (n == 0)
      return 1.0f;
    return (1.0f - x * x) * T_n;
  }
}

SimdChebyshevInterpolationSingle::SimdChebyshevInterpolationSingle(
    const ChebyshevInterpolationSingle &interp)
    : interp_(interp) {}

void SimdChebyshevInterpolationSingle::evaluate(const std::vector<float> &x,
                                                std::vector<float> &y,
                                                bool extrapolate) const {

  if (y.size() < x.size()) {
    y.resize(x.size());
  }

  // Process in groups of 8 using AVX2 SIMD (instead of 4 as in the double
  // version)
  size_t i = 0;
  for (; i + 7 < x.size(); i += 8) {
    std::array<float, 8> x_block = {x[i],     x[i + 1], x[i + 2], x[i + 3],
                                    x[i + 4], x[i + 5], x[i + 6], x[i + 7]};
    std::array<float, 8> y_block = evaluate8(x_block, extrapolate);

    for (size_t j = 0; j < 8; ++j) {
      y[i + j] = y_block[j];
    }
  }

  // Handle remaining points
  for (; i < x.size(); ++i) {
    y[i] = interp_(x[i], extrapolate);
  }
}

std::array<float, 8>
SimdChebyshevInterpolationSingle::evaluate8(const std::array<float, 8> &x,
                                            bool extrapolate) const {

  std::array<float, 8> result;

  // Map input points to standard domain
  std::array<float, 8> t;
  const float domain_start = interp_.getDomainStart();
  const float domain_end = interp_.getDomainEnd();

  for (size_t i = 0; i < 8; ++i) {
    t[i] = 2.0f * (x[i] - domain_start) / (domain_end - domain_start) - 1.0f;

    // Check domain bounds if not extrapolating
    if (!extrapolate && (t[i] < -1.0f || t[i] > 1.0f)) {
      throw std::domain_error("Evaluation point outside interpolation domain");
    }
  }

  // Get interpolation details
  const auto &coeffs = interp_.getCoefficients();
  ChebyshevKind kind = interp_.getKind();

  // Load coefficients and t values into SIMD registers
  if (coeffs.size() >= 8 && kind == SECOND_KIND) {
    // Use AVX2 SIMD for Horner's method (second kind)
    __m256 result_vec = _mm256_setzero_ps();
    __m256 t_vec = _mm256_loadu_ps(t.data());

    // Horner's method with SIMD
    for (int k = coeffs.size() - 1; k >= 0; --k) {
      __m256 coeff_vec = _mm256_set1_ps(coeffs[k]);
      // result = result * t + coeff
      result_vec = _mm256_fmadd_ps(result_vec, t_vec, coeff_vec);
    }

    // Store the result
    _mm256_storeu_ps(result.data(), result_vec);
  } else {
    // Fall back to scalar computation for small coefficient sets or first kind
    for (int i = 0; i < 8; ++i) {
      if (kind == FIRST_KIND) {
        // Apply Clenshaw's algorithm
        float b_kp2 = 0.0f;
        float b_kp1 = 0.0f;
        float b_k;

        for (int k = static_cast<int>(coeffs.size()) - 1; k >= 0; --k) {
          b_k = coeffs[k] + 2.0f * t[i] * b_kp1 - b_kp2;
          b_kp2 = b_kp1;
          b_kp1 = b_k;
        }

        result[i] = b_kp1 - t[i] * b_kp2;
      } else { // SECOND_KIND
        // Apply Horner's method
        float r = 0.0f;
        for (int k = coeffs.size() - 1; k >= 0; --k) {
          r = r * t[i] + coeffs[k];
        }
        result[i] = r;
      }
    }
  }

  return result;
}

std::shared_ptr<ChebyshevInterpolationSingle>
createChebyshevInterpolationSingle(size_t num_points,
                                   const std::function<float(float)> &func,
                                   ChebyshevKind kind, float domain_start,
                                   float domain_end) {

  return std::make_shared<ChebyshevInterpolationSingle>(
      num_points, func, kind, domain_start, domain_end);
}

std::shared_ptr<ChebyshevInterpolationSingle>
createChebyshevInterpolationSingle(const std::vector<float> &nodes,
                                   const std::vector<float> &values,
                                   ChebyshevKind kind, float domain_start,
                                   float domain_end) {

  return std::make_shared<ChebyshevInterpolationSingle>(
      nodes, values, kind, domain_start, domain_end);
}

} // namespace num
} // namespace alo
} // namespace engine