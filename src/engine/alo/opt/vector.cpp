#include "vector.h"
#include "simd.h"
#include <immintrin.h>
#include <sleef.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace engine {
namespace alo {
namespace opt {

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
    
    // Constants for tanh-based approximation of erf
    const double SCALE = 1.2732395447351628; // 4/π
    const double SQRT_2 = M_SQRT2;
    
    // Process in chunks of 4 using SLEEF-based approximation
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        
        // Scale inputs for tanh approximation
        __m256d scaled = _mm256_mul_pd(vec, _mm256_set1_pd(SCALE * SQRT_2));
        
        // Use SLEEF tanh for erf approximation
        __m256d res = Sleef_tanhd4_u10avx2(scaled);
        
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
            result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
        }
        return;
    }
    
    // Process in chunks of 4 using the improved A&S implementation
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        
        // Constants for the A&S 26.2.17 approximation
        const __m256d b1 = _mm256_set1_pd(0.31938153);
        const __m256d b2 = _mm256_set1_pd(-0.356563782);
        const __m256d b3 = _mm256_set1_pd(1.781477937);
        const __m256d b4 = _mm256_set1_pd(-1.821255978);
        const __m256d b5 = _mm256_set1_pd(1.330274429);
        const __m256d p = _mm256_set1_pd(0.2316419);
        const __m256d ONE = _mm256_set1_pd(1.0);
        const __m256d INV_SQRT_2PI = _mm256_set1_pd(0.3989422804); // 1/sqrt(2π)
        
        // Get absolute values and sign mask
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x_vec);
        __m256d sign_mask = _mm256_cmp_pd(x_vec, _mm256_setzero_pd(), _CMP_GE_OQ);
        
        // Handle extreme values
        __m256d large_mask = _mm256_cmp_pd(abs_x, _mm256_set1_pd(8.0), _CMP_GT_OQ);
        __m256d extreme_result = _mm256_blendv_pd(_mm256_setzero_pd(), ONE, sign_mask);
        
        // Calculate t = 1.0 / (1.0 + p * |x|)
        __m256d t = _mm256_div_pd(ONE, _mm256_add_pd(ONE, _mm256_mul_pd(p, abs_x)));
        
        // Calculate polynomial
        __m256d poly = b1;
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), b2);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), b3);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), b4);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), b5);
        poly = _mm256_mul_pd(poly, t);
        
        // Calculate exp(-x²/2) term
        __m256d x_squared = _mm256_mul_pd(abs_x, abs_x);
        __m256d neg_half_x_squared = _mm256_mul_pd(_mm256_set1_pd(-0.5), x_squared);
        __m256d exp_term = Sleef_expd4_u10avx2(neg_half_x_squared);
        
        // Calculate Z = exp(-x²/2) / sqrt(2π)
        __m256d z = _mm256_mul_pd(exp_term, INV_SQRT_2PI);
        
        // Calculate poly * z
        __m256d term = _mm256_mul_pd(poly, z);
        
        // For x >= 0: 1.0 - term
        // For x < 0: term
        __m256d normal_result = _mm256_blendv_pd(term, _mm256_sub_pd(ONE, term), sign_mask);
        
        // Blend extreme values and computed values
        __m256d result_vec = _mm256_blendv_pd(normal_result, extreme_result, large_mask);
        
        _mm256_storeu_pd(result + i, result_vec);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
    }
}

void VectorMath::normalCDFHighPrecision(const double* x, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF's native erf function
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        
        // Scale for normal CDF: x/sqrt(2)
        __m256d scaled_x = _mm256_div_pd(x_vec, _mm256_set1_pd(M_SQRT2));
        
        // Use SLEEF's native erf function
        __m256d erf_x = Sleef_erfd4_u10avx2(scaled_x);
        
        // Calculate 0.5 * (1 + erf(x/sqrt(2)))
        __m256d one = _mm256_set1_pd(1.0);
        __m256d half = _mm256_set1_pd(0.5);
        __m256d result_vec = _mm256_mul_pd(_mm256_add_pd(one, erf_x), half);
        
        // Store results
        _mm256_storeu_pd(result + i, result_vec);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
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

void VectorMath::bsPut(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* result, size_t size) {
    // Just call our high-precision implementation
    bsPutHighPrecision(S, K, r, q, vol, T, result, size);
}

void VectorMath::bsCall(const double* S, const double* K, const double* r, const double* q, 
                       const double* vol, const double* T, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            // Handle degenerate cases
            if (vol[i] <= 0.0 || T[i] <= 0.0) {
                result[i] = std::max(0.0, S[i] - K[i]);
                continue;
            }
            
            // Calculate Black-Scholes
            double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) 
                      / (vol[i] * std::sqrt(T[i]));
            double d2 = d1 - vol[i] * std::sqrt(T[i]);
            
            double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
            double Nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
            
            result[i] = S[i] * std::exp(-q[i] * T[i]) * Nd1 - K[i] * std::exp(-r[i] * T[i]) * Nd2;
        }
        return;
    }
    
    // Use high-precision implementation here too
    bsCallHighPrecision(S, K, r, q, vol, T, result, size);
}

void VectorMath::expMultSqrt(const double* x, const double* y, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::exp(x[i]) * std::sqrt(y[i]);
        }
        return;
    }
    
    // Process in chunks of 4 using AVX2 and SLEEF
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        // Load 4 values from each array
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        
        // Calculate exp(x) using SLEEF
        __m256d exp_x = Sleef_expd4_u10avx2(x_vec);
        
        // Calculate sqrt(y) using AVX2 native sqrt (very fast)
        __m256d sqrt_y = _mm256_sqrt_pd(y_vec);
        
        // Multiply the results
        __m256d result_vec = _mm256_mul_pd(exp_x, sqrt_y);
        
        // Store the result
        _mm256_storeu_pd(result + i, result_vec);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::exp(x[i]) * std::sqrt(y[i]);
    }
}

void VectorMath::bsD1D2(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* d1, double* d2, size_t size) {
    // Use scalar operations for small data sizes or when SIMD isn't available
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

void VectorMath::discountedNormal(const double* x, const double* r, const double* T, 
                                 double* result, size_t size) {
    // Use scalar operations for small data sizes or when SIMD isn't available
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
        
        // Calculate normal CDF with temporary array
        double normal_values[4];
        _mm256_storeu_pd(normal_values, x_vec);
        
        for (int j = 0; j < 4; j++) {
            normal_values[j] = 0.5 * (1.0 + std::erf(normal_values[j] / std::sqrt(2.0)));
        }
        
        __m256d normal = _mm256_loadu_pd(normal_values);
        
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

void VectorMath::bsPutHighPrecision(const double *S, const double *K,
                                    const double *r, const double *q,
                                    const double *vol, const double *T,
                                    double *result, size_t size) {
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

  // Use the SLEEF-based normalCDFHighPrecision implementation
  normalCDFHighPrecision(neg_d1.data(), Nd1.data(), size);
  normalCDFHighPrecision(neg_d2.data(), Nd2.data(), size);

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

void VectorMath::bsCallHighPrecision(const double* S, const double* K, const double* r,
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
    normalCDFHighPrecision(d1.data(), Nd1.data(), size);
    normalCDFHighPrecision(d2.data(), Nd2.data(), size);
    
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

} // namespace opt
} // namespace alo
} // namespace engine