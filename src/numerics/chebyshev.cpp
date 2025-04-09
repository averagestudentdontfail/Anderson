#include "chebyshev.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <eigen3/Eigen/Dense>

namespace numerics {

ChebyshevInterpolation::ChebyshevInterpolation(size_t num_points, 
                                             const std::function<double(double)>& func, 
                                             ChebyshevKind kind,
                                             double domain_start,
                                             double domain_end)
    : num_points_(num_points), 
      kind_(kind),
      domain_start_(domain_start),
      domain_end_(domain_end) {
    
    if (num_points_ < 2) {
        throw std::invalid_argument("ChebyshevInterpolation requires at least 2 points");
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

ChebyshevInterpolation::ChebyshevInterpolation(const std::vector<double>& nodes,
                                             const std::vector<double>& values,
                                             ChebyshevKind kind,
                                             double domain_start,
                                             double domain_end)
    : num_points_(nodes.size()),
      kind_(kind),
      domain_start_(domain_start),
      domain_end_(domain_end),
      nodes_(nodes),
      values_(values) {
    
    if (nodes_.size() != values_.size()) {
        throw std::invalid_argument("Nodes and values must have the same size");
    }
    
    if (nodes_.size() < 2) {
        throw std::invalid_argument("ChebyshevInterpolation requires at least 2 points");
    }
    
    if (domain_end_ <= domain_start_) {
        throw std::invalid_argument("Domain end must be greater than domain start");
    }
    
    // Compute Chebyshev coefficients
    computeCoefficients();
}

void ChebyshevInterpolation::initializeNodes() {
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

void ChebyshevInterpolation::computeCoefficients() {
    coefficients_.resize(num_points_);
    
    if (kind_ == FIRST_KIND) {
        // For first kind, use DCT-II (Discrete Cosine Transform Type II)
        for (size_t k = 0; k < num_points_; ++k) {
            double sum = 0.0;
            for (size_t j = 0; j < num_points_; ++j) {
                sum += values_[j] * std::cos(M_PI * k * (2.0 * j + 1.0) / (2.0 * num_points_));
            }
            coefficients_[k] = (2.0 / num_points_) * sum;
        }
        // Adjust the first coefficient
        coefficients_[0] *= 0.5;
    } else { // SECOND_KIND
#ifndef NO_EIGEN
        // Use Eigen if available for more accurate calculations
        Eigen::MatrixXd T(num_points_, num_points_);
        
        for (size_t i = 0; i < num_points_; ++i) {
            for (size_t j = 0; j < num_points_; ++j) {
                if (j == 0) {
                    T(i, j) = 1.0;
                } else if (j == 1) {
                    T(i, j) = nodes_[i];
                } else {
                    // Use recurrence relation: T_n(x) = 2x*T_{n-1}(x) - T_{n-2}(x)
                    T(i, j) = 2.0 * nodes_[i] * T(i, j-1) - T(i, j-2);
                }
            }
        }
        
        // Compute least squares solution for the coefficients
        Eigen::VectorXd y(num_points_);
        for (size_t i = 0; i < num_points_; ++i) {
            y(i) = values_[i];
        }
        
        // Solve the least squares problem: T * coeffs = y
        Eigen::VectorXd c = T.colPivHouseholderQr().solve(y);
        
        for (size_t i = 0; i < num_points_; ++i) {
            coefficients_[i] = c(i);
        }
#else
        // Fallback implementation without Eigen - use Discrete Cosine Transform approach
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
#endif
    }
}

double ChebyshevInterpolation::operator()(double x, bool extrapolate) const {
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

void ChebyshevInterpolation::updateValues(const std::vector<double>& values) {
    if (values.size() != num_points_) {
        throw std::invalid_argument("Number of values must match the number of nodes");
    }
    
    values_ = values;
    computeCoefficients();
}

double ChebyshevInterpolation::mapToStandardDomain(double x) const {
    return 2.0 * (x - domain_start_) / (domain_end_ - domain_start_) - 1.0;
}

double ChebyshevInterpolation::mapFromStandardDomain(double t) const {
    return 0.5 * ((domain_end_ - domain_start_) * t + domain_end_ + domain_start_);
}

double ChebyshevInterpolation::evaluateChebyshev(int n, double x, ChebyshevKind kind) const {
    if (n == 0) return 1.0;
    if (n == 1) return x;
    
    // Use recurrence relation to compute higher-order polynomials
    double T_n_minus_2 = 1.0;  // T_0(x)
    double T_n_minus_1 = x;    // T_1(x)
    double T_n = 0.0;
    
    for (int i = 2; i <= n; ++i) {
        T_n = 2.0 * x * T_n_minus_1 - T_n_minus_2;
        T_n_minus_2 = T_n_minus_1;
        T_n_minus_1 = T_n;
    }
    
    if (kind == FIRST_KIND) {
        return T_n;
    } else { // SECOND_KIND
        // Convert to second kind Chebyshev polynomial if needed
        // U_n(x) = (T_{n+1}'(x) - T_{n-1}'(x)) / 2
        if (n == 0) return 1.0;
        return (1.0 - x * x) * T_n;
    }
}

} // namespace numerics