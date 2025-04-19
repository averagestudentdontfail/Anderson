#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

#include <vector>
#include <functional>
#include <stdexcept>
#include <cmath>
#include <immintrin.h>
#include "../common/simd/simdops.h"
#include "../common/simd/vectmth.h"

namespace numerics {

/**
 * @brief Chebyshev polynomial kind
 */
enum ChebyshevKind {
    FIRST_KIND,   // Chebyshev polynomials of the first kind T_n(x)
    SECOND_KIND   // Chebyshev polynomials of the second kind U_n(x)
};

/**
 * @brief Base class for Chebyshev interpolation
 */
class ChebyshevInterpolation {
public:
    /**
     * @brief Constructor from function
     * 
     * @param num_points Number of interpolation points
     * @param func Function to interpolate
     * @param kind Chebyshev polynomial kind
     * @param domain_start Start of domain
     * @param domain_end End of domain
     */
    ChebyshevInterpolation(
        size_t num_points, 
        const std::function<double(double)>& func, 
        ChebyshevKind kind = SECOND_KIND,
        double domain_start = -1.0,
        double domain_end = 1.0)
        : kind_(kind), domain_start_(domain_start), domain_end_(domain_end) {
        
        if (num_points < 2) {
            throw std::invalid_argument("ChebyshevInterpolation: Number of points must be at least 2");
        }
        
        if (domain_end <= domain_start) {
            throw std::invalid_argument("ChebyshevInterpolation: Domain end must be greater than domain start");
        }
        
        // Generate Chebyshev nodes
        std::vector<double> nodes(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            // Chebyshev nodes of the first kind (extrema of T_n)
            double t = std::cos(M_PI * i / (num_points - 1));
            
            // Map from [-1, 1] to [domain_start, domain_end]
            double x = 0.5 * ((domain_end - domain_start) * t + (domain_end + domain_start));
            nodes[i] = x;
        }
        
        // Evaluate function at nodes
        std::vector<double> values(num_points);
        for (size_t i = 0; i < num_points; ++i) {
            values[i] = func(nodes[i]);
        }
        
        // Compute coefficients
        computeCoefficients(nodes, values);
    }
    
    /**
     * @brief Constructor from nodes and values
     * 
     * @param nodes Interpolation nodes
     * @param values Function values at nodes
     * @param kind Chebyshev polynomial kind
     * @param domain_start Start of domain
     * @param domain_end End of domain
     */
    ChebyshevInterpolation(
        const std::vector<double>& nodes,
        const std::vector<double>& values,
        ChebyshevKind kind = SECOND_KIND,
        double domain_start = -1.0,
        double domain_end = 1.0)
        : kind_(kind), domain_start_(domain_start), domain_end_(domain_end) {
        
        if (nodes.size() < 2) {
            throw std::invalid_argument("ChebyshevInterpolation: Number of nodes must be at least 2");
        }
        
        if (nodes.size() != values.size()) {
            throw std::invalid_argument("ChebyshevInterpolation: Number of nodes and values must match");
        }
        
        if (domain_end <= domain_start) {
            throw std::invalid_argument("ChebyshevInterpolation: Domain end must be greater than domain start");
        }
        
        // Compute coefficients
        computeCoefficients(nodes, values);
    }
    
    /**
     * @brief Virtual destructor
     */
    virtual ~ChebyshevInterpolation() = default;
    
    /**
     * @brief Evaluate interpolation at a point
     * 
     * @param x Evaluation point
     * @param extrapolate Whether to allow extrapolation
     * @return Interpolated value
     */
    virtual double operator()(double x, bool extrapolate = false) const {
        // Map from [domain_start, domain_end] to [-1, 1]
        double t = 2.0 * (x - domain_start_) / (domain_end_ - domain_start_) - 1.0;
        
        // Check domain bounds if not extrapolating
        if (!extrapolate && (t < -1.0 || t > 1.0)) {
            throw std::domain_error("Evaluation point outside interpolation domain");
        }
        
        if (kind_ == FIRST_KIND) {
            // Use Clenshaw algorithm for first kind
            return evaluateClenshawFirstKind(t);
        } else {
            // Use Horner's method for second kind
            return evaluateHornerSecondKind(t);
        }
    }
    
    /**
     * @brief Get Chebyshev polynomial kind
     * 
     * @return Chebyshev polynomial kind
     */
    ChebyshevKind kind() const {
        return kind_;
    }
    
    /**
     * @brief Get domain start
     * 
     * @return Domain start
     */
    double domainStart() const {
        return domain_start_;
    }
    
    /**
     * @brief Get domain end
     * 
     * @return Domain end
     */
    double domainEnd() const {
        return domain_end_;
    }
    
    /**
     * @brief Get interpolation coefficients
     * 
     * @return Interpolation coefficients
     */
    const std::vector<double>& coefficients() const {
        return coefficients_;
    }
    
protected:
    /**
     * @brief Compute Chebyshev coefficients from nodes and values
     * 
     * @param nodes Interpolation nodes
     * @param values Function values at nodes
     */
    void computeCoefficients(
        const std::vector<double>& nodes,
        const std::vector<double>& values) {
        
        size_t N = nodes.size();
        coefficients_.resize(N);
        
        // Map nodes from [domain_start, domain_end] to [-1, 1]
        std::vector<double> t(N);
        for (size_t i = 0; i < N; ++i) {
            t[i] = 2.0 * (nodes[i] - domain_start_) / (domain_end_ - domain_start_) - 1.0;
        }
        
        // Compute coefficients using discrete cosine transform
        for (size_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < N; ++k) {
                sum += values[k] * std::cos(j * M_PI * k / (N - 1));
            }
            
            // Apply normalization
            if (j == 0 || j == N - 1) {
                coefficients_[j] = sum / (N - 1);
            } else {
                coefficients_[j] = 2.0 * sum / (N - 1);
            }
        }
    }
    
    /**
     * @brief Evaluate using Clenshaw algorithm for first kind
     * 
     * @param t Point in [-1, 1]
     * @return Interpolated value
     */
    double evaluateClenshawFirstKind(double t) const {
        double b_kp2 = 0.0;
        double b_kp1 = 0.0;
        double b_k;
        
        // Apply Clenshaw algorithm in reverse
        for (int k = coefficients_.size() - 1; k >= 0; --k) {
            b_k = coefficients_[k] + 2.0 * t * b_kp1 - b_kp2;
            b_kp2 = b_kp1;
            b_kp1 = b_k;
        }
        
        return b_kp1 - t * b_kp2;
    }
    
    /**
     * @brief Evaluate using Horner's method for second kind
     * 
     * @param t Point in [-1, 1]
     * @return Interpolated value
     */
    double evaluateHornerSecondKind(double t) const {
        double result = 0.0;
        
        // Apply Horner's method in reverse
        for (int k = coefficients_.size() - 1; k >= 0; --k) {
            result = result * t + coefficients_[k];
        }
        
        return result;
    }
    
private:
    ChebyshevKind kind_;
    double domain_start_;
    double domain_end_;
    std::vector<double> coefficients_;
};

/**
 * @brief SIMD-accelerated Chebyshev interpolation
 * 
 * This class implements Chebyshev interpolation with SIMD acceleration
 * for improved performance when evaluating at multiple points.
 */
class SimdChebyshevInterpolation : public ChebyshevInterpolation {
public:
    /**
     * @brief Constructor from function
     * 
     * @param num_points Number of interpolation points
     * @param func Function to interpolate
     * @param kind Chebyshev polynomial kind
     * @param domain_start Start of domain
     * @param domain_end End of domain
     * @param use_simd Whether to use SIMD acceleration (default: true)
     */
    SimdChebyshevInterpolation(
        size_t num_points, 
        const std::function<double(double)>& func, 
        ChebyshevKind kind = SECOND_KIND,
        double domain_start = -1.0,
        double domain_end = 1.0,
        bool use_simd = true)
        : ChebyshevInterpolation(num_points, func, kind, domain_start, domain_end),
          use_simd_(use_simd) {}
    
    /**
     * @brief Constructor from nodes and values
     * 
     * @param nodes Interpolation nodes
     * @param values Function values at nodes
     * @param kind Chebyshev polynomial kind
     * @param domain_start Start of domain
     * @param domain_end End of domain
     * @param use_simd Whether to use SIMD acceleration (default: true)
     */
    SimdChebyshevInterpolation(
        const std::vector<double>& nodes,
        const std::vector<double>& values,
        ChebyshevKind kind = SECOND_KIND,
        double domain_start = -1.0,
        double domain_end = 1.0,
        bool use_simd = true)
        : ChebyshevInterpolation(nodes, values, kind, domain_start, domain_end),
          use_simd_(use_simd) {}
    
    /**
     * @brief Evaluate interpolation at a point
     * 
     * Overrides ChebyshevInterpolation::operator()
     * 
     * @param x Evaluation point
     * @param extrapolate Whether to allow extrapolation
     * @return Interpolated value
     */
    double operator()(double x, bool extrapolate = false) const override {
        // For a single point, just use the base class implementation
        return ChebyshevInterpolation::operator()(x, extrapolate);
    }
    
    /**
     * @brief Batch evaluation at multiple points
     * 
     * @param x Vector of evaluation points
     * @param extrapolate Whether to allow extrapolation
     * @return Vector of interpolated values
     */
    std::vector<double> batchEvaluate(
        const std::vector<double>& x, 
        bool extrapolate = false) const {
        
        std::vector<double> result(x.size());
        
        if (use_simd_ && x.size() >= 4) {
            batchEvaluateWithSIMD(x.data(), result.data(), x.size(), extrapolate);
        } else {
            // Regular evaluation for each point
            for (size_t i = 0; i < x.size(); ++i) {
                result[i] = (*this)(x[i], extrapolate);
            }
        }
        
        return result;
    }
    
    /**
     * @brief Enable or disable SIMD acceleration
     * 
     * @param use_simd Whether to use SIMD
     */
    void setUseSIMD(bool use_simd) {
        use_simd_ = use_simd;
    }
    
    /**
     * @brief Check if SIMD acceleration is enabled
     * 
     * @return True if SIMD is enabled, false otherwise
     */
    bool getUseSIMD() const {
        return use_simd_;
    }
    
private:
    /**
     * @brief SIMD-accelerated batch evaluation
     * 
     * @param x Input points array
     * @param result Output values array
     * @param size Number of points
     * @param extrapolate Whether to allow extrapolation
     */
    void batchEvaluateWithSIMD(
        const double* x, 
        double* result, 
        size_t size,
        bool extrapolate) const {
        
        // Map standard domain
        std::vector<double> t(size);
        const double domain_start = domainStart();
        const double domain_end = domainEnd();
        
        for (size_t i = 0; i < size; ++i) {
            t[i] = 2.0 * (x[i] - domain_start) / (domain_end - domain_start) - 1.0;
            
            // Check domain bounds if not extrapolating
            if (!extrapolate && (t[i] < -1.0 || t[i] > 1.0)) {
                throw std::domain_error("Evaluation point outside interpolation domain");
            }
        }
        
        // Get coefficients
        const auto& coeffs = coefficients();
        
        if (kind() == FIRST_KIND) {
            // Process using Clenshaw algorithm with SIMD
            
            // Initialize recurrence
            std::vector<double> b_kp2(size, 0.0);
            std::vector<double> b_kp1(size, 0.0);
            std::vector<double> b_k(size);
            
            // Process in reverse order (Clenshaw algorithm)
            for (int k = static_cast<int>(coeffs.size()) - 1; k >= 0; --k) {
                // b_k = coeffs[k] + 2.0 * t * b_kp1 - b_kp2
                // Process in SIMD chunks
                size_t i = 0;
                for (; i + 3 < size; i += 4) {
                    __m256d t_vec = _mm256_loadu_pd(t.data() + i);
                    __m256d b_kp1_vec = _mm256_loadu_pd(b_kp1.data() + i);
                    __m256d b_kp2_vec = _mm256_loadu_pd(b_kp2.data() + i);
                    __m256d coeff_vec = _mm256_set1_pd(coeffs[k]);
                    __m256d two = _mm256_set1_pd(2.0);
                    
                    // 2.0 * t
                    __m256d two_t = _mm256_mul_pd(two, t_vec);
                    
                    // 2.0 * t * b_kp1
                    __m256d term = _mm256_mul_pd(two_t, b_kp1_vec);
                    
                    // coeffs[k] + 2.0 * t * b_kp1
                    __m256d sum = _mm256_add_pd(coeff_vec, term);
                    
                    // coeffs[k] + 2.0 * t * b_kp1 - b_kp2
                    __m256d b_k_vec = _mm256_sub_pd(sum, b_kp2_vec);
                    
                    _mm256_storeu_pd(b_k.data() + i, b_k_vec);
                }
                
                // Handle remaining elements
                for (; i < size; ++i) {
                    b_k[i] = coeffs[k] + 2.0 * t[i] * b_kp1[i] - b_kp2[i];
                }
                
                // Update for next iteration
                b_kp2 = b_kp1;
                b_kp1 = b_k;
            }
            
            // Compute final result: b_kp1 - t * b_kp2
            size_t i = 0;
            for (; i + 3 < size; i += 4) {
                __m256d t_vec = _mm256_loadu_pd(t.data() + i);
                __m256d b_kp1_vec = _mm256_loadu_pd(b_kp1.data() + i);
                __m256d b_kp2_vec = _mm256_loadu_pd(b_kp2.data() + i);
                
                // t * b_kp2
                __m256d product = _mm256_mul_pd(t_vec, b_kp2_vec);
                
                // b_kp1 - t * b_kp2
                __m256d result_vec = _mm256_sub_pd(b_kp1_vec, product);
                
                _mm256_storeu_pd(result + i, result_vec);
            }
            
            // Handle remaining elements
            for (; i < size; ++i) {
                result[i] = b_kp1[i] - t[i] * b_kp2[i];
            }
        } else {  // SECOND_KIND
            // Direct evaluation using Horner's method with SIMD
            
            // Initialize result to zero
            std::fill(result, result + size, 0.0);
            
            // Apply Horner's method in reverse
            for (int k = coeffs.size() - 1; k >= 0; --k) {
                // Process in SIMD chunks
                size_t i = 0;
                for (; i + 3 < size; i += 4) {
                    __m256d t_vec = _mm256_loadu_pd(t.data() + i);
                    __m256d result_vec = _mm256_loadu_pd(result + i);
                    __m256d coeff_vec = _mm256_set1_pd(coeffs[k]);
                    
                    // result * t
                    __m256d product = _mm256_mul_pd(result_vec, t_vec);
                    
                    // result * t + coeffs[k]
                    __m256d sum = _mm256_add_pd(product, coeff_vec);
                    
                    _mm256_storeu_pd(result + i, sum);
                }
                
                // Handle remaining elements
                for (; i < size; ++i) {
                    result[i] = result[i] * t[i] + coeffs[k];
                }
            }
        }
    }
    
private:
    bool use_simd_;
};

/**
 * @brief Factory function to create SIMD-accelerated Chebyshev interpolation
 * 
 * @param num_points Number of interpolation points
 * @param func Function to interpolate
 * @param kind Chebyshev polynomial kind
 * @param domain_start Start of domain
 * @param domain_end End of domain
 * @param use_simd Whether to use SIMD acceleration
 * @return Shared pointer to SIMD Chebyshev interpolation
 */
std::shared_ptr<SimdChebyshevInterpolation> createSimdChebyshevInterpolation(
    size_t num_points, 
    const std::function<double(double)>& func, 
    ChebyshevKind kind = SECOND_KIND,
    double domain_start = -1.0,
    double domain_end = 1.0,
    bool use_simd = true) {
    
    return std::make_shared<SimdChebyshevInterpolation>(
        num_points, func, kind, domain_start, domain_end, use_simd);
}

/**
 * @brief Factory function to create SIMD-accelerated Chebyshev interpolation from nodes and values
 * 
 * @param nodes Interpolation nodes
 * @param values Function values at nodes
 * @param kind Chebyshev polynomial kind
 * @param domain_start Start of domain
 * @param domain_end End of domain
 * @param use_simd Whether to use SIMD acceleration
 * @return Shared pointer to SIMD Chebyshev interpolation
 */
std::shared_ptr<SimdChebyshevInterpolation> createSimdChebyshevInterpolation(
    const std::vector<double>& nodes,
    const std::vector<double>& values,
    ChebyshevKind kind = SECOND_KIND,
    double domain_start = -1.0,
    double domain_end = 1.0,
    bool use_simd = true) {
    
    return std::make_shared<SimdChebyshevInterpolation>(
        nodes, values, kind, domain_start, domain_end, use_simd);
}

} // namespace numerics

#endif