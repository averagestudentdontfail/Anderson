#include "vector.h"
#include "simd.h"
#include <immintrin.h>
#include <sleef.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstring> 
#include <cstdint>

namespace engine {
namespace alo {
namespace opt {

// SIMD detection implementation
SIMDSupport detectSIMDSupport() {
    int info[4];
    
    // Check for SSE2
    __cpuid(info, 1);
    if (!(info[3] & (1 << 26))) return NONE;
    
    // Check for AVX
    if (!(info[2] & (1 << 28))) return SSE2;
    
    // Check for AVX2
    __cpuid(info, 7);
    if (!(info[1] & (1 << 5))) return AVX;
    
    // Check for AVX-512F
    if (!(info[1] & (1 << 16))) return AVX2;
    
    return AVX512;
}

// Data conversion utilities
std::vector<float> VectorMath::convertToFloat(const std::vector<double>& input) {
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = static_cast<float>(input[i]);
    }
    return result;
}

std::vector<double> VectorMath::convertToDouble(const std::vector<float>& input) {
    std::vector<double> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = static_cast<double>(input[i]);
    }
    return result;
}

// Since we removed SIMD detection, we redefine shouldUseSimd to just check the size threshold
inline bool shouldUseSimd(size_t size) {
    return size >= 32;  // Only use SIMD for operations with size >= 32
}

/**
 * @brief Vectorized operations for arrays of data
 */
void VectorMath::exp(const double* x, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::exp(x[i]);
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF's AVX2 implementation
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        // Use SLEEF's optimized exponential function
        __m256d res = Sleef_expd4_u10avx2(vec);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::exp(x[i]);
    }
}

void VectorMath::log(const double* x, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::log(x[i]);
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        // Use SLEEF's optimized logarithm function
        __m256d res = Sleef_logd4_u10avx2(vec);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::log(x[i]);
    }
}

void VectorMath::sqrt(const double* x, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::sqrt(x[i]);
        }
        return;
    }
    
    // Process in chunks of 4 using AVX2 native sqrt
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = _mm256_sqrt_pd(vec);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::sqrt(x[i]);
    }
}

void VectorMath::erf(const double* x, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::erf(x[i]);
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF's native erf function
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = Sleef_erfd4_u10avx2(vec);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::erf(x[i]);
    }
}

void VectorMath::normalCDF(const double* x, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            // Choose the most numerically stable formula based on the value of x
            if (x[i] < -8.0) {
                // For large negative x, use erfc-based formula
                result[i] = 0.5 * std::erfc(-x[i] / std::sqrt(2.0));
            } else if (x[i] > 8.0) {
                // For large positive x, result is very close to 1
                result[i] = 1.0 - 0.5 * std::erfc(x[i] / std::sqrt(2.0));
            } else {
                // For moderate values, use erf-based formula
                result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
            }
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF with appropriate formula
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        
        // Get absolute value and sign masks for condition testing
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x_vec);
        
        // Create masks for extreme values
        __m256d large_neg_mask = _mm256_cmp_pd(x_vec, _mm256_set1_pd(-8.0), _CMP_LT_OS);
        __m256d large_pos_mask = _mm256_cmp_pd(x_vec, _mm256_set1_pd(8.0), _CMP_GT_OS);
        
        // Scale for normal CDF: x/sqrt(2)
        __m256d scaled_x = _mm256_div_pd(x_vec, _mm256_set1_pd(M_SQRT2));
        
        // For normal range values (-8 to 8), use erf
        __m256d erf_scaled = Sleef_erfd4_u10avx2(scaled_x);
        __m256d one = _mm256_set1_pd(1.0);
        __m256d half = _mm256_set1_pd(0.5);
        __m256d normal_result = _mm256_mul_pd(_mm256_add_pd(one, erf_scaled), half);
        
        // For extreme values, calculate using erfc for better numerical stability
        __m256d neg_scaled_x = _mm256_sub_pd(_mm256_setzero_pd(), scaled_x);
        
        // For negative large x: N(x) = 0.5*erfc(-x/sqrt(2))
        __m256d large_neg_result = _mm256_mul_pd(half, Sleef_erfcd4_u15avx2(neg_scaled_x));
        
        // For positive large x: N(x) = 1 - 0.5*erfc(x/sqrt(2))
        __m256d large_pos_result = _mm256_sub_pd(one, 
                                _mm256_mul_pd(half, Sleef_erfcd4_u15avx2(scaled_x)));
        
        // Blend results based on extreme value masks
        __m256d result_vec = normal_result;
        result_vec = _mm256_blendv_pd(result_vec, large_neg_result, large_neg_mask);
        result_vec = _mm256_blendv_pd(result_vec, large_pos_result, large_pos_mask);
        
        // Store result
        _mm256_storeu_pd(result + i, result_vec);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        if (x[i] < -8.0) {
            result[i] = 0.5 * std::erfc(-x[i] / std::sqrt(2.0));
        } else if (x[i] > 8.0) {
            result[i] = 1.0 - 0.5 * std::erfc(x[i] / std::sqrt(2.0));
        } else {
            result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
        }
    }
}

void VectorMath::normalPDF(const double* x, double* result, size_t size) {
    // Normal PDF: (1/sqrt(2π)) * exp(-0.5 * x²)
    const double INV_SQRT_2PI = 0.3989422804014327; // 1/sqrt(2*PI)
    
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = INV_SQRT_2PI * std::exp(-0.5 * x[i] * x[i]);
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d xvec = _mm256_loadu_pd(x + i);
        
        // Square the values
        __m256d x_squared = _mm256_mul_pd(xvec, xvec);
        
        // Multiply by -0.5
        __m256d scaled = _mm256_mul_pd(x_squared, _mm256_set1_pd(-0.5));
        
        // Use SLEEF exp function
        __m256d exp_term = Sleef_expd4_u10avx2(scaled);
        
        // Multiply by 1/sqrt(2π)
        __m256d pdf = _mm256_mul_pd(exp_term, _mm256_set1_pd(INV_SQRT_2PI));
        
        _mm256_storeu_pd(result + i, pdf);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = INV_SQRT_2PI * std::exp(-0.5 * x[i] * x[i]);
    }
}

void VectorMath::multiply(const double* a, const double* b, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
        return;
    }
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(a + i);
        __m256d vec_b = _mm256_loadu_pd(b + i);
        __m256d res = _mm256_mul_pd(vec_a, vec_b);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

void VectorMath::add(const double* a, const double* b, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
        return;
    }
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(a + i);
        __m256d vec_b = _mm256_loadu_pd(b + i);
        __m256d res = _mm256_add_pd(vec_a, vec_b);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void VectorMath::subtract(const double* a, const double* b, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] - b[i];
        }
        return;
    }
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(a + i);
        __m256d vec_b = _mm256_loadu_pd(b + i);
        __m256d res = _mm256_sub_pd(vec_a, vec_b);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] - b[i];
    }
}

void VectorMath::divide(const double* a, const double* b, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] / b[i];
        }
        return;
    }
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(a + i);
        __m256d vec_b = _mm256_loadu_pd(b + i);
        __m256d res = _mm256_div_pd(vec_a, vec_b);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] / b[i];
    }
}

void VectorMath::bsD1(const double* S, const double* K, const double* r, const double* q, 
                    const double* vol, const double* T, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            double vol_sqrt_T = vol[i] * std::sqrt(T[i]);
            double half_vol_squared = 0.5 * vol[i] * vol[i];
            double log_S_div_K = std::log(S[i] / K[i]);
            double drift = r[i] - q[i] + half_vol_squared;
            double drift_T = drift * T[i];
            double numerator = log_S_div_K + drift_T;
            result[i] = numerator / vol_sqrt_T;
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        // Load vectors
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Calculate sqrt(T)
        __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
        
        // Calculate vol * sqrt(T)
        __m256d vol_sqrt_T = _mm256_mul_pd(vol_vec, sqrt_T);
        
        // Calculate S/K
        __m256d S_div_K = _mm256_div_pd(S_vec, K_vec);
        
        // Calculate log(S/K) using SLEEF
        __m256d log_S_div_K = Sleef_logd4_u10avx2(S_div_K);
        
        // Calculate 0.5 * vol^2
        __m256d vol_squared = _mm256_mul_pd(vol_vec, vol_vec);
        __m256d half_vol_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), vol_squared);
        
        // Calculate (r-q)
        __m256d r_minus_q = _mm256_sub_pd(r_vec, q_vec);
        
        // Calculate (r-q) + 0.5*vol^2
        __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
        
        // Calculate drift * T
        __m256d drift_T = _mm256_mul_pd(drift, T_vec);
        
        // Calculate numerator
        __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
        
        // Calculate d1 = numerator / (vol * sqrt(T))
        __m256d d1 = _mm256_div_pd(numerator, vol_sqrt_T);
        
        // Store result
        _mm256_storeu_pd(result + i, d1);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        double vol_sqrt_T = vol[i] * std::sqrt(T[i]);
        double half_vol_squared = 0.5 * vol[i] * vol[i];
        double log_S_div_K = std::log(S[i] / K[i]);
        double drift = r[i] - q[i] + half_vol_squared;
        double drift_T = drift * T[i];
        double numerator = log_S_div_K + drift_T;
        result[i] = numerator / vol_sqrt_T;
    }
}

void VectorMath::bsD2(const double* d1, const double* vol, const double* T, 
                    double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = d1[i] - vol[i] * std::sqrt(T[i]);
        }
        return;
    }
    
    // Process in chunks of 4
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d d1_vec = _mm256_loadu_pd(d1 + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Calculate sqrt(T)
        __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
        
        // Calculate vol * sqrt(T)
        __m256d vol_sqrt_T = _mm256_mul_pd(vol_vec, sqrt_T);
        
        // Calculate d2 = d1 - vol * sqrt(T)
        __m256d d2 = _mm256_sub_pd(d1_vec, vol_sqrt_T);
        
        // Store result
        _mm256_storeu_pd(result + i, d2);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = d1[i] - vol[i] * std::sqrt(T[i]);
    }
}

void VectorMath::bsD1D2(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* d1, double* d2, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            // Handle degenerate cases
            if (vol[i] <= 0.0 || T[i] <= 0.0) {
                d1[i] = 0.0;  // Default values for degenerate cases
                d2[i] = 0.0;
                continue;
            }
            
            // Standard Black-Scholes d1 and d2 calculation
            double vol_sqrt_T = vol[i] * std::sqrt(T[i]);
            double half_vol_squared = 0.5 * vol[i] * vol[i];
            double log_S_div_K = std::log(S[i] / K[i]);
            double drift = r[i] - q[i] + half_vol_squared;
            double drift_T = drift * T[i];
            
            d1[i] = (log_S_div_K + drift_T) / vol_sqrt_T;
            d2[i] = d1[i] - vol_sqrt_T;
        }
        return;
    }
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        // Load vectors
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Check for degenerate cases
        __m256d eps = _mm256_set1_pd(1e-10);
        __m256d vol_mask = _mm256_cmp_pd(vol_vec, eps, _CMP_LE_OQ);
        __m256d T_mask = _mm256_cmp_pd(T_vec, eps, _CMP_LE_OQ);
        __m256d degenerate_mask = _mm256_or_pd(vol_mask, T_mask);
        
        // For non-degenerate cases, calculate d1 and d2
        __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
        __m256d vol_sqrt_T = _mm256_mul_pd(vol_vec, sqrt_T);
        
        // Calculate log(S/K) with high precision
        __m256d S_div_K = _mm256_div_pd(S_vec, K_vec);
        __m256d log_S_div_K = Sleef_logd4_u10avx2(S_div_K);
        
        // Calculate drift term
        __m256d vol_squared = _mm256_mul_pd(vol_vec, vol_vec);
        __m256d half_vol_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), vol_squared);
        __m256d r_minus_q = _mm256_sub_pd(r_vec, q_vec);
        __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
        __m256d drift_T = _mm256_mul_pd(drift, T_vec);
        
        // Calculate d1 and d2
        __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
        __m256d d1_vec = _mm256_div_pd(numerator, vol_sqrt_T);
        __m256d d2_vec = _mm256_sub_pd(d1_vec, vol_sqrt_T);
        
        // Set default values for degenerate cases
        __m256d zero = _mm256_setzero_pd();
        d1_vec = _mm256_blendv_pd(d1_vec, zero, degenerate_mask);
        d2_vec = _mm256_blendv_pd(d2_vec, zero, degenerate_mask);
        
        // Store results
        _mm256_storeu_pd(d1 + i, d1_vec);
        _mm256_storeu_pd(d2 + i, d2_vec);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        if (vol[i] <= 0.0 || T[i] <= 0.0) {
            d1[i] = 0.0;
            d2[i] = 0.0;
            continue;
        }
        
        double vol_sqrt_T = vol[i] * std::sqrt(T[i]);
        double half_vol_squared = 0.5 * vol[i] * vol[i];
        double log_S_div_K = std::log(S[i] / K[i]);
        double drift = r[i] - q[i] + half_vol_squared;
        double drift_T = drift * T[i];
        
        d1[i] = (log_S_div_K + drift_T) / vol_sqrt_T;
        d2[i] = d1[i] - vol_sqrt_T;
    }
}

void VectorMath::calculateD1D2Fused(
    const double* S, const double* K, const double* r, const double* q, 
    const double* vol, const double* T, double* d1, double* d2, size_t size) {
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        // Load vectors
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Calculate sqrt(time) once
        __m256d sqrt_time = _mm256_sqrt_pd(T_vec);
        
        // Calculate vol * sqrt(time)
        __m256d vol_sqrt_time = _mm256_mul_pd(vol_vec, sqrt_time);
        
        // Calculate volatility squared / 2
        __m256d vol_squared = _mm256_mul_pd(vol_vec, vol_vec);
        __m256d half_vol_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), vol_squared);
        
        // Calculate log(spot/strike)
        __m256d spot_div_strike = _mm256_div_pd(S_vec, K_vec);
        __m256d log_term = Sleef_logd4_u10avx2(spot_div_strike);
        
        // Calculate r-q
        __m256d r_minus_q = _mm256_sub_pd(r_vec, q_vec);
        
        // Calculate (r-q) + vol^2/2
        __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
        
        // Calculate (r-q + vol^2/2) * time
        __m256d drift_time = _mm256_mul_pd(drift, T_vec);
        
        // Calculate numerator for d1: log(S/K) + (r-q+vol^2/2)*time
        __m256d d1_num = _mm256_add_pd(log_term, drift_time);
        
        // Calculate d1 = numerator / (vol*sqrt(time))
        __m256d d1_vec = _mm256_div_pd(d1_num, vol_sqrt_time);
        
        // Calculate d2 = d1 - vol*sqrt(time)
        __m256d d2_vec = _mm256_sub_pd(d1_vec, vol_sqrt_time);
        
        // Store results
        _mm256_storeu_pd(d1 + i, d1_vec);
        _mm256_storeu_pd(d2 + i, d2_vec);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        double vol_sqrt_T = vol[i] * std::sqrt(T[i]);
        double half_vol_squared = 0.5 * vol[i] * vol[i];
        double log_S_div_K = std::log(S[i] / K[i]);
        double drift = r[i] - q[i] + half_vol_squared;
        double drift_T = drift * T[i];
        
        d1[i] = (log_S_div_K + drift_T) / vol_sqrt_T;
        d2[i] = d1[i] - vol_sqrt_T;
    }
}

void VectorMath::bsPut(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* result, size_t size) {
    // Allocate temporary arrays for intermediate results
    std::vector<double> d1(size);
    std::vector<double> d2(size);
    std::vector<double> neg_d1(size);
    std::vector<double> neg_d2(size);
    std::vector<double> Nd1(size);
    std::vector<double> Nd2(size);

    // Calculate d1 and d2 in a single pass
    bsD1D2(S, K, r, q, vol, T, d1.data(), d2.data(), size);

    // Negate d1 and d2 for put formula
    for (size_t i = 0; i < size; i++) {
        neg_d1[i] = -d1[i];
        neg_d2[i] = -d2[i];
    }

    // Use normalCDF implementation
    normalCDF(neg_d1.data(), Nd1.data(), size);
    normalCDF(neg_d2.data(), Nd2.data(), size);
    
    // Calculate discount factors
    std::vector<double> discount_r(size);
    std::vector<double> discount_q(size);

    for (size_t i = 0; i < size; i++) {
        discount_r[i] = std::exp(-r[i] * T[i]);
        discount_q[i] = std::exp(-q[i] * T[i]);
    }

    // Calculate final put price
    for (size_t i = 0; i < size; i++) {
        // Handle degenerate cases
        if (vol[i] <= 0.0 || T[i] <= 0.0) {
            result[i] = std::max(0.0, K[i] - S[i]);
            continue;
        }

        double term1 = K[i] * discount_r[i] * Nd2[i];
        double term2 = S[i] * discount_q[i] * Nd1[i];
        result[i] = term1 - term2;
    }
}

void VectorMath::bsCall(const double* S, const double* K, const double* r,
                       const double* q, const double* vol, const double* T,
                       double* result, size_t size) {
    // Allocate temporary arrays for intermediate results
    std::vector<double> d1(size);
    std::vector<double> d2(size);
    std::vector<double> Nd1(size);
    std::vector<double> Nd2(size);
    std::vector<double> discount_r(size);
    std::vector<double> discount_q(size);
    
    // Calculate d1 and d2 in a single pass
    bsD1D2(S, K, r, q, vol, T, d1.data(), d2.data(), size);
    
    // Calculate N(d1) and N(d2) using high precision function
    normalCDF(d1.data(), Nd1.data(), size);
    normalCDF(d2.data(), Nd2.data(), size);
    
    // Calculate discount factors
    for (size_t i = 0; i < size; i++) {
        discount_r[i] = std::exp(-r[i] * T[i]);
        discount_q[i] = std::exp(-q[i] * T[i]);
    }
    
    // Calculate final call price: S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
    for (size_t i = 0; i < size; i++) {
        // Handle degenerate cases
        if (vol[i] <= 0.0 || T[i] <= 0.0) {
            result[i] = std::max(0.0, S[i] - K[i]);
            continue;
        }
        
        double term1 = S[i] * discount_q[i] * Nd1[i];
        double term2 = K[i] * discount_r[i] * Nd2[i];
        result[i] = term1 - term2;
    }
}

void VectorMath::expMultSqrt(const double* x, const double* y, double* result, size_t size) {
    // Process larger chunks if SIMD is available
    constexpr size_t SIMD_THRESHOLD = 32;
    constexpr size_t CACHE_LINE_SIZE = 64; // Bytes, typical L1 cache line
    constexpr size_t DOUBLES_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(double);
    
    // Use scalar operations for small data sizes
    if (size < SIMD_THRESHOLD) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::exp(x[i]) * std::sqrt(y[i]);
        }
        return;
    }
    
    // Ensure memory is aligned for SIMD operations
    size_t i = 0;
    
    // Process unaligned start portion
    while (i < size && (reinterpret_cast<uintptr_t>(result + i) % 32) != 0) {
        result[i] = std::exp(x[i]) * std::sqrt(y[i]);
        i++;
    }
    
    // Process main chunk with SIMD - process in larger blocks to improve cache usage
    constexpr size_t BLOCK_SIZE = 1024; // Process in blocks of 1KB
    
    for (; i + BLOCK_SIZE <= size; i += BLOCK_SIZE) {
        // Process the block with SIMD
        for (size_t j = 0; j < BLOCK_SIZE; j += 4) {
            // Load 4 values from each array
            __m256d x_vec = _mm256_loadu_pd(x + i + j);
            __m256d y_vec = _mm256_loadu_pd(y + i + j);
            
            // Calculate exp(x) using SLEEF
            __m256d exp_x = Sleef_expd4_u10avx2(x_vec);
            
            // Calculate sqrt(y) using AVX2 native sqrt (very fast)
            __m256d sqrt_y = _mm256_sqrt_pd(y_vec);
            
            // Multiply the results
            __m256d result_vec = _mm256_mul_pd(exp_x, sqrt_y);
            
            // Store the result with aligned store if possible
            if ((reinterpret_cast<uintptr_t>(result + i + j) % 32) == 0) {
                _mm256_store_pd(result + i + j, result_vec);
            } else {
                _mm256_storeu_pd(result + i + j, result_vec);
            }
        }
    }
    
    // Process remaining blocks of 4
    for (; i + 3 < size; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        
        __m256d exp_x = Sleef_expd4_u10avx2(x_vec);
        __m256d sqrt_y = _mm256_sqrt_pd(y_vec);
        __m256d result_vec = _mm256_mul_pd(exp_x, sqrt_y);
        
        _mm256_storeu_pd(result + i, result_vec);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::exp(x[i]) * std::sqrt(y[i]);
    }
}

void VectorMath::discountedNormal(const double* x, const double* r, const double* T, 
                                 double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            // Calculate discount factor
            double discount = std::exp(-r[i] * T[i]);
            
            // Calculate normal CDF with maximum precision
            double normal = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
            
            // Combine results
            result[i] = discount * normal;
        }
        return;
    }
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Calculate discount factor e^(-r*T)
        __m256d neg_r_T = _mm256_mul_pd(
            _mm256_mul_pd(r_vec, T_vec),
            _mm256_set1_pd(-1.0)
        );
        __m256d discount = Sleef_expd4_u10avx2(neg_r_T);
        
        // Calculate normal CDF
        __m256d scaled_x = _mm256_div_pd(x_vec, _mm256_set1_pd(M_SQRT2));
        __m256d erf_scaled = Sleef_erfd4_u10avx2(scaled_x);
        __m256d normal = _mm256_mul_pd(
            _mm256_set1_pd(0.5),
            _mm256_add_pd(_mm256_set1_pd(1.0), erf_scaled)
        );
        
        // Multiply discount by normal
        __m256d result_vec = _mm256_mul_pd(discount, normal);
        
        // Store results
        _mm256_storeu_pd(result + i, result_vec);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        double discount = std::exp(-r[i] * T[i]);
        double normal = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
        result[i] = discount * normal;
    }
}

void VectorMath::processUnrolledBatch(const double* S, const double* K, const double* r,
                                    const double* q, const double* vol, const double* T,
                                    double* results, size_t size) {
    // Process in blocks of 32 (8 SIMD vectors of 4 doubles each)
    for (size_t i = 0; i < size; i += 32) {
        // Only process full blocks of 32
        if (i + 32 <= size) {
            // Process in chunks of 4 (AVX2 256-bit vectors)
            for (size_t j = 0; j < 32; j += 4) {
                __m256d S_vec = _mm256_set1_pd(S[0]);  // Assuming constant S
                __m256d K_vec = _mm256_loadu_pd(K + i + j);
                __m256d r_vec = _mm256_set1_pd(r[0]);  // Assuming constant r
                __m256d q_vec = _mm256_set1_pd(q[0]);  // Assuming constant q
                __m256d vol_vec = _mm256_set1_pd(vol[0]);  // Assuming constant vol
                __m256d T_vec = _mm256_set1_pd(T[0]);  // Assuming constant T
                
                // Calculate d1
                __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
                __m256d vol_sqrt_T = _mm256_mul_pd(vol_vec, sqrt_T);
                
                // Calculate log(S/K)
                __m256d S_div_K = _mm256_div_pd(S_vec, K_vec);
                __m256d log_S_div_K = Sleef_logd4_u10avx2(S_div_K);
                
                // Calculate drift term
                __m256d vol_squared = _mm256_mul_pd(vol_vec, vol_vec);
                __m256d half_vol_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), vol_squared);
                __m256d r_minus_q = _mm256_sub_pd(r_vec, q_vec);
                __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
                __m256d drift_T = _mm256_mul_pd(drift, T_vec);
                
                // Calculate d1 and d2
                __m256d d1_num = _mm256_add_pd(log_S_div_K, drift_T);
                __m256d d1 = _mm256_div_pd(d1_num, vol_sqrt_T);
                __m256d d2 = _mm256_sub_pd(d1, vol_sqrt_T);
                
                // Calculate -d1 and -d2 for put formula
                __m256d neg_d1 = _mm256_sub_pd(_mm256_setzero_pd(), d1);
                __m256d neg_d2 = _mm256_sub_pd(_mm256_setzero_pd(), d2);
                
                // Calculate N(-d1) and N(-d2)
                __m256d Nd1 = SimdOps::normalCDF(neg_d1);
                __m256d Nd2 = SimdOps::normalCDF(neg_d2);
                
                // Calculate discount factors
                __m256d neg_r_T = _mm256_mul_pd(_mm256_mul_pd(r_vec, T_vec), _mm256_set1_pd(-1.0));
                __m256d neg_q_T = _mm256_mul_pd(_mm256_mul_pd(q_vec, T_vec), _mm256_set1_pd(-1.0));
                __m256d dr = Sleef_expd4_u10avx2(neg_r_T);
                __m256d dq = Sleef_expd4_u10avx2(neg_q_T);
                
                // Calculate K * e^(-rT) * N(-d2)
                __m256d term1 = _mm256_mul_pd(K_vec, dr);
                term1 = _mm256_mul_pd(term1, Nd2);
                
                // Calculate S * e^(-qT) * N(-d1)
                __m256d term2 = _mm256_mul_pd(S_vec, dq);
                term2 = _mm256_mul_pd(term2, Nd1);
                
                // Calculate put price
                __m256d result = _mm256_sub_pd(term1, term2);
                
                // Store results
                _mm256_storeu_pd(results + i + j, result);
            }
        } else {
            // Handle remaining elements (less than 32)
            for (size_t j = i; j < size; ++j) {
                // Use scalar calculation for tail elements
                double d1 = (std::log(S[0] / K[j]) + (r[0] - q[0] + 0.5 * vol[0] * vol[0]) * T[0]) / 
                           (vol[0] * std::sqrt(T[0]));
                double d2 = d1 - vol[0] * std::sqrt(T[0]);
                
                double Nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
                double Nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
                
                double dr = std::exp(-r[0] * T[0]);
                double dq = std::exp(-q[0] * T[0]);
                
                results[j] = K[j] * dr * Nd2 - S[0] * dq * Nd1;
            }
        }
    }
}

void VectorMath::bsPutWithGreeks(
    const double* S, const double* K, const double* r, const double* q, 
    const double* vol, const double* T, BSGreeks* results, size_t size) {
    
    std::vector<double> d1(size);
    std::vector<double> d2(size);
    
    // Calculate d1 and d2 in a single pass
    calculateD1D2Fused(S, K, r, q, vol, T, d1.data(), d2.data(), size);
    
    // Parallelize with OpenMP if available
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (size_t i = 0; i < size; ++i) {
        // Handle degenerate cases
        if (vol[i] <= 0.0 || T[i] <= 0.0) {
            results[i].price = std::max(0.0, K[i] - S[i]);
            results[i].delta = (S[i] < K[i]) ? -1.0 : 0.0;
            results[i].gamma = 0.0;
            results[i].vega = 0.0;
            results[i].theta = 0.0;
            results[i].rho = 0.0;
            continue;
        }
        
        // Calculate discount factors
        double dr = std::exp(-r[i] * T[i]);
        double dq = std::exp(-q[i] * T[i]);
        
        // Calculate N(-d1) and N(-d2)
        double Nd1 = 0.5 * (1.0 + std::erf(-d1[i] / std::sqrt(2.0)));
        double Nd2 = 0.5 * (1.0 + std::erf(-d2[i] / std::sqrt(2.0)));
        
        // Calculate option price
        results[i].price = K[i] * dr * Nd2 - S[i] * dq * Nd1;
        
        // Calculate Greeks
        double sqrt_T = std::sqrt(T[i]);
        double pdf_d1 = std::exp(-0.5 * d1[i] * d1[i]) / std::sqrt(2.0 * M_PI);
        
        // Delta = -e^(-q*T) * N(-d1)
        results[i].delta = -dq * Nd1;
        
        // Gamma = e^(-q*T) * PDF(d1) / (S * vol * sqrt(T))
        results[i].gamma = dq * pdf_d1 / (S[i] * vol[i] * sqrt_T);
        
        // Vega = S * e^(-q*T) * PDF(d1) * sqrt(T) / 100
        results[i].vega = 0.01 * S[i] * dq * pdf_d1 * sqrt_T;
        
        // Theta = -(S * e^(-q*T) * PDF(d1) * vol / (2*sqrt(T)) + q*S*e^(-q*T)*N(-d1) - r*K*e^(-r*T)*N(-d2)) / 365
        double term1 = -S[i] * dq * pdf_d1 * vol[i] / (2.0 * sqrt_T);
        double term2 = q[i] * S[i] * dq * Nd1;
        double term3 = -r[i] * K[i] * dr * Nd2;
        results[i].theta = (term1 + term2 + term3) / 365.0;
        
        // Rho = -K * T * e^(-r*T) * N(-d2) / 100
        results[i].rho = -0.01 * K[i] * T[i] * dr * Nd2;
    }
}

void VectorMath::americanPutApprox(const double* S, const double* K, const double* r,
                                const double* q, const double* vol, const double* T,
                                double* results, size_t size) {
    // Process options in chunks of 4 using SIMD
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        // Load parameters
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // First calculate European price
        __m256d bs_price = SimdOps::bsPut(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);
        
        // Approximate early exercise premium using quadratic approximation
        // This is a simplified approximation for illustration
        __m256d one = _mm256_set1_pd(1.0);
        __m256d zero = _mm256_setzero_pd();
        
        // Check if early exercise might be valuable (r > q)
        __m256d r_gt_q = _mm256_cmp_pd(r_vec, q_vec, _CMP_GT_OQ);
        
        // Calculate critical ratio b* ≈ K * (1 - exp(-r*T))/(1 - exp(-q*T))
        __m256d neg_rT = _mm256_mul_pd(_mm256_mul_pd(r_vec, T_vec), _mm256_set1_pd(-1.0));
        __m256d neg_qT = _mm256_mul_pd(_mm256_mul_pd(q_vec, T_vec), _mm256_set1_pd(-1.0));
        
        __m256d exp_negr = Sleef_expd4_u10avx2(neg_rT);
        __m256d exp_negq = Sleef_expd4_u10avx2(neg_qT);
        
        __m256d num = _mm256_sub_pd(one, exp_negr);
        __m256d denom = _mm256_sub_pd(one, exp_negq);
        
        // Handle division by zero by using a mask
        __m256d denom_is_zero = _mm256_cmp_pd(denom, zero, _CMP_EQ_OQ);
        denom = _mm256_blendv_pd(denom, _mm256_set1_pd(1.0), denom_is_zero);
        
        __m256d b_star = _mm256_mul_pd(K_vec, _mm256_div_pd(num, denom));
        
        // If S <= b*, calculate approximate early exercise premium
        // Premium ≈ K - S - (K - b*) * (S/b*)^((2*r)/vol^2)
        __m256d S_le_b = _mm256_cmp_pd(S_vec, b_star, _CMP_LE_OQ);
        
        __m256d K_minus_S = _mm256_sub_pd(K_vec, S_vec);
        __m256d K_minus_b = _mm256_sub_pd(K_vec, b_star);
        
        __m256d S_div_b = _mm256_div_pd(S_vec, b_star);
        __m256d r2 = _mm256_mul_pd(_mm256_set1_pd(2.0), r_vec);
        __m256d vol2 = _mm256_mul_pd(vol_vec, vol_vec);
        __m256d power = _mm256_div_pd(r2, vol2);
        
        // Need to calculate (S/b*)^power
        // Use log and exp: (S/b*)^power = exp(power * log(S/b*))
        __m256d log_ratio = Sleef_logd4_u10avx2(S_div_b);
        __m256d log_pow = _mm256_mul_pd(power, log_ratio);
        __m256d ratio_pow = Sleef_expd4_u10avx2(log_pow);
        
        __m256d second_term = _mm256_mul_pd(K_minus_b, ratio_pow);
        __m256d premium = _mm256_sub_pd(K_minus_S, second_term);
        
        // Ensure premium is non-negative
        premium = _mm256_max_pd(premium, zero);
        
        // Add premium only if r > q and S <= b*
        __m256d apply_premium = _mm256_and_pd(r_gt_q, S_le_b);
        __m256d american_price = _mm256_add_pd(
            bs_price, 
            _mm256_and_pd(premium, _mm256_castsi256_pd(_mm256_castpd_si256(apply_premium)))
        );
        
        // Store results
        _mm256_storeu_pd(results + i, american_price);
    }
    
    // Handle remaining elements with scalar code
    for (; i < size; ++i) {
        // First calculate European price
        double bs_price = 0.0;
        
        // Use Black-Scholes formula directly 
        if (vol[i] <= 0.0 || T[i] <= 0.0) {
            bs_price = std::max(0.0, K[i] - S[i]);
        } else {
            double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) / 
                       (vol[i] * std::sqrt(T[i]));
            double d2 = d1 - vol[i] * std::sqrt(T[i]);
            
            double Nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
            double Nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
            
            bs_price = K[i] * std::exp(-r[i] * T[i]) * Nd2 - S[i] * std::exp(-q[i] * T[i]) * Nd1;
        }
        
        // Check if early exercise might be valuable
        double premium = 0.0;
        if (r[i] > q[i]) {
            // Calculate approximate critical price
            double num = 1.0 - std::exp(-r[i] * T[i]);
            double denom = 1.0 - std::exp(-q[i] * T[i]);
            if (denom == 0.0) denom = 1.0;
            
            double b_star = K[i] * num / denom;
            
            // If S <= b*, calculate approximate early exercise premium
            if (S[i] <= b_star) {
                double power = 2.0 * r[i] / (vol[i] * vol[i]);
                double ratio_pow = std::pow(S[i] / b_star, power);
                premium = std::max(0.0, K[i] - S[i] - (K[i] - b_star) * ratio_pow);
            }
        }
        
        results[i] = bs_price + premium;
    }
}

} // namespace opt
} // namespace alo
} // namespace engine