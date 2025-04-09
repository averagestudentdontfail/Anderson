#ifndef INTEGRATE_H
#define INTEGRATE_H

#include "integrate.h"
#include "../common/simd/simdops.h"
#include "../common/simd/vectmth.h"
#include <immintrin.h>
#include <vector>
#include <memory>
#include <functional>

namespace numerics {

/**
 * @brief SIMD-accelerated Gauss-Legendre integration
 * 
 * This class implements Gauss-Legendre quadrature with SIMD acceleration
 * for improved performance when integrating functions at multiple points.
 */
class SimdGaussLegendreIntegrator : public Integrator {
public:
    /**
     * @brief Constructor with specified order
     * 
     * @param order Number of integration points
     * @param use_simd Whether to use SIMD acceleration (default: true)
     */
    explicit SimdGaussLegendreIntegrator(size_t order, bool use_simd = true)
        : order_(order), use_simd_(use_simd) {
        if (order_ < 1) {
            throw std::invalid_argument("SimdGaussLegendreIntegrator: Order must be at least 1");
        }
        initializeNodesAndWeights();
    }
    
    /**
     * @brief Destructor
     */
    ~SimdGaussLegendreIntegrator() override = default;
    
    /**
     * @brief Integrate a function over an interval
     * 
     * @param f Function to integrate
     * @param a Lower bound
     * @param b Upper bound
     * @return Approximate integral value
     */
    double integrate(const std::function<double(double)>& f, double a, double b) const override {
        // Check if SIMD should be used
        if (use_simd_) {
            return integrateWithSIMD(f, a, b);
        } else {
            return integrateStandard(f, a, b);
        }
    }
    
    /**
     * @brief Batch integration of a function at multiple intervals
     * 
     * @param f Function to integrate
     * @param intervals Vector of (lower, upper) bound pairs
     * @return Vector of approximate integral values
     */
    std::vector<double> batchIntegrate(
        const std::function<double(double)>& f,
        const std::vector<std::pair<double, double>>& intervals) const {
        
        std::vector<double> results(intervals.size());
        
        // Process batch in parallel chunks if large enough
        if (intervals.size() >= 8) {
            // TODO: Add parallel processing
            // For now, just iterate
            for (size_t i = 0; i < intervals.size(); ++i) {
                results[i] = integrate(f, intervals[i].first, intervals[i].second);
            }
        } else {
            // For small batches, just use standard iteration
            for (size_t i = 0; i < intervals.size(); ++i) {
                results[i] = integrate(f, intervals[i].first, intervals[i].second);
            }
        }
        
        return results;
    }
    
    /**
     * @brief Get the name of the integrator
     * 
     * @return Integrator name
     */
    std::string name() const override {
        return use_simd_ ? "SIMD-Gauss-Legendre" : "Gauss-Legendre";
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
     * @brief Initialize Gauss-Legendre nodes and weights
     */
    void initializeNodesAndWeights() {
        // Precomputed Gauss-Legendre nodes and weights for common orders
        if (order_ == 7) {
            nodes_ = {
                -0.9491079123427585, 
                -0.7415311855993944,
                -0.4058451513773972,
                0.0,
                0.4058451513773972,
                0.7415311855993944,
                0.9491079123427585
            };
            
            weights_ = {
                0.1294849661688697,
                0.2797053914892767,
                0.3818300505051189,
                0.4179591836734694,
                0.3818300505051189,
                0.2797053914892767,
                0.1294849661688697
            };
        } else if (order_ == 25) {
            // 25-point Gauss-Legendre quadrature nodes and weights
            nodes_ = {
                -0.9955569697904981, -0.9766639214595175, -0.9429745712289743, -0.8949919978782753, -0.8334426287608340,
                -0.7592592630373576, -0.6735663684734684, -0.5776629302412229, -0.4731469662935845, -0.3611723058093879,
                -0.2429801799032639, -0.1207530708447741, 0.0, 0.1207530708447741, 0.2429801799032639,
                0.3611723058093879, 0.4731469662935845, 0.5776629302412229, 0.6735663684734684, 0.7592592630373576,
                0.8334426287608340, 0.8949919978782753, 0.9429745712289743, 0.9766639214595175, 0.9955569697904981
            };
            
            weights_ = {
                0.0113937985010262, 0.0263549866150321, 0.0409391567013063, 0.0549046959758351, 0.0680383338123569,
                0.0801407003350010, 0.0910282619829636, 0.1005359490670506, 0.1085196244742637, 0.1148582591457116,
                0.1194557635357847, 0.1222424429903100, 0.1231760537267154, 0.1222424429903100, 0.1194557635357847,
                0.1148582591457116, 0.1085196244742637, 0.1005359490670506, 0.0910282619829636, 0.0801407003350010,
                0.0680383338123569, 0.0549046959758351, 0.0409391567013063, 0.0263549866150321, 0.0113937985010262
            };
        } else {
            // For other orders, use a simple but less accurate approach
            nodes_.resize(order_);
            weights_.resize(order_);
            
            for (size_t i = 0; i < order_; ++i) {
                double theta = M_PI * (i + 0.5) / order_;
                nodes_[i] = std::cos(theta);
                weights_[i] = M_PI / order_;
            }
        }
    }
    
    /**
     * @brief Standard implementation of Gauss-Legendre integration
     * 
     * @param f Function to integrate
     * @param a Lower bound
     * @param b Upper bound
     * @return Approximate integral value
     */
    double integrateStandard(const std::function<double(double)>& f, double a, double b) const {
        double result = 0.0;
        
        // Change of variable to map [a,b] to [-1,1]
        const double half_length = 0.5 * (b - a);
        const double mid_point = 0.5 * (a + b);
        
        for (size_t i = 0; i < order_; ++i) {
            const double x = mid_point + half_length * nodes_[i];
            result += weights_[i] * f(x);
        }
        
        result *= half_length;
        return result;
    }
    
    /**
     * @brief SIMD-accelerated implementation of Gauss-Legendre integration
     * 
     * @param f Function to integrate
     * @param a Lower bound
     * @param b Upper bound
     * @return Approximate integral value
     */
    double integrateWithSIMD(const std::function<double(double)>& f, double a, double b) const {
        // Change of variable to map [a,b] to [-1,1]
        const double half_length = 0.5 * (b - a);
        const double mid_point = 0.5 * (a + b);
        
        double result = 0.0;
        
        // Process nodes in groups of 4 using SIMD
        size_t i = 0;
        for (; i + 3 < order_; i += 4) {
            __m256d nodes_vec = _mm256_set_pd(nodes_[i+3], nodes_[i+2], nodes_[i+1], nodes_[i]);
            __m256d weights_vec = _mm256_set_pd(weights_[i+3], weights_[i+2], weights_[i+1], weights_[i]);
            
            // Calculate x values: x = mid_point + half_length * nodes
            __m256d half_length_vec = _mm256_set1_pd(half_length);
            __m256d mid_point_vec = _mm256_set1_pd(mid_point);
            __m256d nodes_scaled = _mm256_mul_pd(nodes_vec, half_length_vec);
            __m256d x_vec = _mm256_add_pd(mid_point_vec, nodes_scaled);
            
            // Evaluate function at each x value (no SIMD for function evaluation)
            double x_values[4];
            _mm256_storeu_pd(x_values, x_vec);
            
            __m256d f_vec = _mm256_set_pd(
                f(x_values[0]), 
                f(x_values[1]), 
                f(x_values[2]),
                f(x_values[3])
            );
            
            // Multiply by weights and accumulate
            __m256d weighted = _mm256_mul_pd(f_vec, weights_vec);
            
            // Sum the vector elements
            __m256d sum_vec = _mm256_hadd_pd(weighted, weighted);
            __m128d lo = _mm256_extractf128_pd(sum_vec, 0);
            __m128d hi = _mm256_extractf128_pd(sum_vec, 1);
            __m128d sum_128 = _mm_add_pd(lo, hi);
            double sum_values[2];
            _mm_storeu_pd(sum_values, sum_128);
            
            result += sum_values[0] + sum_values[1];
        }
        
        // Handle remaining nodes (if any)
        for (; i < order_; ++i) {
            const double x = mid_point + half_length * nodes_[i];
            result += weights_[i] * f(x);
        }
        
        result *= half_length;
        return result;
    }
    
private:
    size_t order_;
    bool use_simd_;
    std::vector<double> nodes_;
    std::vector<double> weights_;
};

/**
 * @brief Factory function to create SIMD-accelerated integrator
 * 
 * @param scheme_type Type of integration scheme
 * @param order Number of integration points
 * @param tolerance Error tolerance for adaptive schemes
 * @param use_simd Whether to use SIMD acceleration
 * @return Shared pointer to integrator instance
 */
std::shared_ptr<Integrator> createSimdIntegrator(
    const std::string& scheme_type, 
    size_t order = 0, 
    double tolerance = 0.0,
    bool use_simd = true) {
    
    if (scheme_type == "GaussLegendre") {
        return std::make_shared<SimdGaussLegendreIntegrator>(order, use_simd);
    } else {
        // If the requested integrator doesn't have a SIMD version,
        // fall back to the standard implementation
        return createIntegrator(scheme_type, order, tolerance);
    }
}

} // namespace numerics

#endif 