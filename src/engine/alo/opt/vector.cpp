#include "vector.h"
#include "simd.h"
#include <immintrin.h>
#include <sleef.h>
#include <algorithm>
#include <cmath>

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
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            // Handle degenerate cases
            if (vol[i] <= 0.0 || T[i] <= 0.0) {
                result[i] = std::max(0.0, K[i] - S[i]);
                continue;
            }
            
            // Calculate Black-Scholes
            double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) 
                      / (vol[i] * std::sqrt(T[i]));
            double d2 = d1 - vol[i] * std::sqrt(T[i]);
            
            double Nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
            double Nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
            
            result[i] = K[i] * std::exp(-r[i] * T[i]) * Nd2 - S[i] * std::exp(-q[i] * T[i]) * Nd1;
        }
        return;
    }
    
    // Process in chunks of 4 using the improved SIMD implementation
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
        __m256d zero = _mm256_setzero_pd();
        __m256d eps = _mm256_set1_pd(1e-10);
        
        // Create mask for vol <= 0 or T <= 0
        __m256d vol_mask = _mm256_cmp_pd(vol_vec, eps, _CMP_LE_OQ);
        __m256d t_mask = _mm256_cmp_pd(T_vec, eps, _CMP_LE_OQ);
        __m256d degenerate_mask = _mm256_or_pd(vol_mask, t_mask);
        
        // Calculate K-S for degenerate cases
        __m256d K_minus_S = _mm256_sub_pd(K_vec, S_vec);
        __m256d degenerate_value = _mm256_max_pd(zero, K_minus_S);
        
        // If all are degenerate, just store intrinsic and continue
        if (_mm256_movemask_pd(degenerate_mask) == 0xF) {
            _mm256_storeu_pd(result + i, degenerate_value);
            continue;
        }
        
        // Calculate d1 components
        __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
        __m256d vol_sqrt_T = _mm256_mul_pd(vol_vec, sqrt_T);
        __m256d S_div_K = _mm256_div_pd(S_vec, K_vec);
        __m256d log_S_div_K = Sleef_logd4_u10avx2(S_div_K);
        
        __m256d vol_squared = _mm256_mul_pd(vol_vec, vol_vec);
        __m256d half_vol_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), vol_squared);
        __m256d r_minus_q = _mm256_sub_pd(r_vec, q_vec);
        __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
        __m256d drift_T = _mm256_mul_pd(drift, T_vec);
        __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
        
        // Calculate d1 and d2
        __m256d d1 = _mm256_div_pd(numerator, vol_sqrt_T);
        __m256d d2 = _mm256_sub_pd(d1, vol_sqrt_T);
        
        // Negate d1 and d2 for put calculation
        __m256d neg_d1 = _mm256_sub_pd(zero, d1);
        __m256d neg_d2 = _mm256_sub_pd(zero, d2);
        
        // Calculate N(-d1) and N(-d2) using our improved normalCDF
        
        // Constants for the A&S 26.2.17 approximation
        const __m256d b1 = _mm256_set1_pd(0.31938153);
        const __m256d b2 = _mm256_set1_pd(-0.356563782);
        const __m256d b3 = _mm256_set1_pd(1.781477937);
        const __m256d b4 = _mm256_set1_pd(-1.821255978);
        const __m256d b5 = _mm256_set1_pd(1.330274429);
        const __m256d p = _mm256_set1_pd(0.2316419);
        const __m256d one = _mm256_set1_pd(1.0);
        const __m256d inv_sqrt_2pi = _mm256_set1_pd(0.3989422804); // 1/sqrt(2π)
        
        // Process N(-d1)
        __m256d abs_neg_d1 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), neg_d1);
        __m256d sign_mask_d1 = _mm256_cmp_pd(neg_d1, _mm256_setzero_pd(), _CMP_GE_OQ);
        __m256d large_mask_d1 = _mm256_cmp_pd(abs_neg_d1, _mm256_set1_pd(8.0), _CMP_GT_OQ);
        __m256d extreme_result_d1 = _mm256_blendv_pd(_mm256_setzero_pd(), one, sign_mask_d1);
        
        __m256d t_d1 = _mm256_div_pd(one, _mm256_add_pd(one, _mm256_mul_pd(p, abs_neg_d1)));
        __m256d poly_d1 = b1;
        poly_d1 = _mm256_add_pd(_mm256_mul_pd(poly_d1, t_d1), b2);
        poly_d1 = _mm256_add_pd(_mm256_mul_pd(poly_d1, t_d1), b3);
        poly_d1 = _mm256_add_pd(_mm256_mul_pd(poly_d1, t_d1), b4);
        poly_d1 = _mm256_add_pd(_mm256_mul_pd(poly_d1, t_d1), b5);
        poly_d1 = _mm256_mul_pd(poly_d1, t_d1);
        
        __m256d x_squared_d1 = _mm256_mul_pd(abs_neg_d1, abs_neg_d1);
        __m256d neg_half_x_squared_d1 = _mm256_mul_pd(_mm256_set1_pd(-0.5), x_squared_d1);
        __m256d exp_term_d1 = Sleef_expd4_u10avx2(neg_half_x_squared_d1);
        __m256d z_d1 = _mm256_mul_pd(exp_term_d1, inv_sqrt_2pi);
        __m256d term_d1 = _mm256_mul_pd(poly_d1, z_d1);
        __m256d Nd1 = _mm256_blendv_pd(term_d1, _mm256_sub_pd(one, term_d1), sign_mask_d1);
        Nd1 = _mm256_blendv_pd(Nd1, extreme_result_d1, large_mask_d1);
        
        // Process N(-d2) - same approach
        __m256d abs_neg_d2 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), neg_d2);
        __m256d sign_mask_d2 = _mm256_cmp_pd(neg_d2, _mm256_setzero_pd(), _CMP_GE_OQ);
        __m256d large_mask_d2 = _mm256_cmp_pd(abs_neg_d2, _mm256_set1_pd(8.0), _CMP_GT_OQ);
        __m256d extreme_result_d2 = _mm256_blendv_pd(_mm256_setzero_pd(), one, sign_mask_d2);
        
        __m256d t_d2 = _mm256_div_pd(one, _mm256_add_pd(one, _mm256_mul_pd(p, abs_neg_d2)));
        __m256d poly_d2 = b1;
        poly_d2 = _mm256_add_pd(_mm256_mul_pd(poly_d2, t_d2), b2);
        poly_d2 = _mm256_add_pd(_mm256_mul_pd(poly_d2, t_d2), b3);
        poly_d2 = _mm256_add_pd(_mm256_mul_pd(poly_d2, t_d2), b4);
        poly_d2 = _mm256_add_pd(_mm256_mul_pd(poly_d2, t_d2), b5);
        poly_d2 = _mm256_mul_pd(poly_d2, t_d2);
        
        __m256d x_squared_d2 = _mm256_mul_pd(abs_neg_d2, abs_neg_d2);
        __m256d neg_half_x_squared_d2 = _mm256_mul_pd(_mm256_set1_pd(-0.5), x_squared_d2);
        __m256d exp_term_d2 = Sleef_expd4_u10avx2(neg_half_x_squared_d2);
        __m256d z_d2 = _mm256_mul_pd(exp_term_d2, inv_sqrt_2pi);
        __m256d term_d2 = _mm256_mul_pd(poly_d2, z_d2);
        __m256d Nd2 = _mm256_blendv_pd(term_d2, _mm256_sub_pd(one, term_d2), sign_mask_d2);
        Nd2 = _mm256_blendv_pd(Nd2, extreme_result_d2, large_mask_d2);
        
        // Calculate discount factors
        __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r_vec), T_vec);
        __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q_vec), T_vec);
        __m256d dr = Sleef_expd4_u10avx2(neg_r_T);
        __m256d dq = Sleef_expd4_u10avx2(neg_q_T);
        
        // K * e^(-rT) * N(-d2)
        __m256d term1 = _mm256_mul_pd(K_vec, dr);
        term1 = _mm256_mul_pd(term1, Nd2);
        
        // S * e^(-qT) * N(-d1)
        __m256d term2 = _mm256_mul_pd(S_vec, dq);
        term2 = _mm256_mul_pd(term2, Nd1);
        
        // put = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        __m256d put_value = _mm256_sub_pd(term1, term2);
        
        // Blend degenerate and computed values
        __m256d final_value = _mm256_blendv_pd(put_value, degenerate_value, degenerate_mask);
        
        // Store result
        _mm256_storeu_pd(result + i, final_value);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        if (vol[i] <= 0.0 || T[i] <= 0.0) {
            result[i] = std::max(0.0, K[i] - S[i]);
            continue;
        }
        
        double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) 
                  / (vol[i] * std::sqrt(T[i]));
        double d2 = d1 - vol[i] * std::sqrt(T[i]);
        
        double Nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
        double Nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
        
        result[i] = K[i] * std::exp(-r[i] * T[i]) * Nd2 - S[i] * std::exp(-q[i] * T[i]) * Nd1;
    }
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
    
    // Process in chunks of 4 using our improved SIMD approach
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        // Load vectors
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Check for degenerate cases (vol <= 0 or T <= 0)
        __m256d zero = _mm256_setzero_pd();
        __m256d eps = _mm256_set1_pd(1e-10);
        __m256d vol_mask = _mm256_cmp_pd(vol_vec, eps, _CMP_LE_OQ);
        __m256d t_mask = _mm256_cmp_pd(T_vec, eps, _CMP_LE_OQ);
        __m256d degenerate_mask = _mm256_or_pd(vol_mask, t_mask);
        
        // Calculate S-K for degenerate cases
        __m256d S_minus_K = _mm256_sub_pd(S_vec, K_vec);
        __m256d degenerate_value = _mm256_max_pd(zero, S_minus_K);
        
        // If all are degenerate, just store and continue
        if (_mm256_movemask_pd(degenerate_mask) == 0xF) {
            _mm256_storeu_pd(result + i, degenerate_value);
            continue;
        }
        
        // Calculate d1 and d2
        __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
        __m256d vol_sqrt_T = _mm256_mul_pd(vol_vec, sqrt_T);
        __m256d S_div_K = _mm256_div_pd(S_vec, K_vec);
        __m256d log_S_div_K = Sleef_logd4_u10avx2(S_div_K);
        __m256d vol_squared = _mm256_mul_pd(vol_vec, vol_vec);
        __m256d half_vol_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), vol_squared);
        __m256d r_minus_q = _mm256_sub_pd(r_vec, q_vec);
        __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
        __m256d drift_T = _mm256_mul_pd(drift, T_vec);
        __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
        __m256d d1 = _mm256_div_pd(numerator, vol_sqrt_T);
        __m256d d2 = _mm256_sub_pd(d1, vol_sqrt_T);
        
        // Calculate N(d1) and N(d2) using our improved normalCDF implementation
        // Using the same A&S 26.2.17 formula as in the put case
        
        // Constants for the A&S approximation
        const __m256d b1 = _mm256_set1_pd(0.31938153);
        const __m256d b2 = _mm256_set1_pd(-0.356563782);
        const __m256d b3 = _mm256_set1_pd(1.781477937);
        const __m256d b4 = _mm256_set1_pd(-1.821255978);
        const __m256d b5 = _mm256_set1_pd(1.330274429);
        const __m256d p = _mm256_set1_pd(0.2316419);
        const __m256d one = _mm256_set1_pd(1.0);
        const __m256d inv_sqrt_2pi = _mm256_set1_pd(0.3989422804); // 1/sqrt(2π)
        
        // Calculate N(d1)
        __m256d abs_d1 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), d1);
        __m256d sign_mask_d1 = _mm256_cmp_pd(d1, _mm256_setzero_pd(), _CMP_GE_OQ);
        __m256d large_mask_d1 = _mm256_cmp_pd(abs_d1, _mm256_set1_pd(8.0), _CMP_GT_OQ);
        __m256d extreme_result_d1 = _mm256_blendv_pd(_mm256_setzero_pd(), one, sign_mask_d1);
        
        __m256d t_d1 = _mm256_div_pd(one, _mm256_add_pd(one, _mm256_mul_pd(p, abs_d1)));
        __m256d poly_d1 = b1;
        poly_d1 = _mm256_add_pd(_mm256_mul_pd(poly_d1, t_d1), b2);
        poly_d1 = _mm256_add_pd(_mm256_mul_pd(poly_d1, t_d1), b3);
        poly_d1 = _mm256_add_pd(_mm256_mul_pd(poly_d1, t_d1), b4);
        poly_d1 = _mm256_add_pd(_mm256_mul_pd(poly_d1, t_d1), b5);
        poly_d1 = _mm256_mul_pd(poly_d1, t_d1);
        
        __m256d x_squared_d1 = _mm256_mul_pd(abs_d1, abs_d1);
        __m256d neg_half_x_squared_d1 = _mm256_mul_pd(_mm256_set1_pd(-0.5), x_squared_d1);
        __m256d exp_term_d1 = Sleef_expd4_u10avx2(neg_half_x_squared_d1);
        __m256d z_d1 = _mm256_mul_pd(exp_term_d1, inv_sqrt_2pi);
        __m256d term_d1 = _mm256_mul_pd(poly_d1, z_d1);
        
        // For d1 >= 0: 1.0 - term_d1, otherwise term_d1
        __m256d Nd1 = _mm256_blendv_pd(term_d1, _mm256_sub_pd(one, term_d1), sign_mask_d1);
        Nd1 = _mm256_blendv_pd(Nd1, extreme_result_d1, large_mask_d1);
        
        // Calculate N(d2) - same approach
        __m256d abs_d2 = _mm256_andnot_pd(_mm256_set1_pd(-0.0), d2);
        __m256d sign_mask_d2 = _mm256_cmp_pd(d2, _mm256_setzero_pd(), _CMP_GE_OQ);
        __m256d large_mask_d2 = _mm256_cmp_pd(abs_d2, _mm256_set1_pd(8.0), _CMP_GT_OQ);
        __m256d extreme_result_d2 = _mm256_blendv_pd(_mm256_setzero_pd(), one, sign_mask_d2);
        
        __m256d t_d2 = _mm256_div_pd(one, _mm256_add_pd(one, _mm256_mul_pd(p, abs_d2)));
        __m256d poly_d2 = b1;
        poly_d2 = _mm256_add_pd(_mm256_mul_pd(poly_d2, t_d2), b2);
        poly_d2 = _mm256_add_pd(_mm256_mul_pd(poly_d2, t_d2), b3);
        poly_d2 = _mm256_add_pd(_mm256_mul_pd(poly_d2, t_d2), b4);
        poly_d2 = _mm256_add_pd(_mm256_mul_pd(poly_d2, t_d2), b5);
        poly_d2 = _mm256_mul_pd(poly_d2, t_d2);
        
        __m256d x_squared_d2 = _mm256_mul_pd(abs_d2, abs_d2);
        __m256d neg_half_x_squared_d2 = _mm256_mul_pd(_mm256_set1_pd(-0.5), x_squared_d2);
        __m256d exp_term_d2 = Sleef_expd4_u10avx2(neg_half_x_squared_d2);
        __m256d z_d2 = _mm256_mul_pd(exp_term_d2, inv_sqrt_2pi);
        __m256d term_d2 = _mm256_mul_pd(poly_d2, z_d2);
        
        // For d2 >= 0: 1.0 - term_d2, otherwise term_d2
        __m256d Nd2 = _mm256_blendv_pd(term_d2, _mm256_sub_pd(one, term_d2), sign_mask_d2);
        Nd2 = _mm256_blendv_pd(Nd2, extreme_result_d2, large_mask_d2);
        
        // Calculate discount factors
        __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r_vec), T_vec);
        __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q_vec), T_vec);
        __m256d dr = Sleef_expd4_u10avx2(neg_r_T);
        __m256d dq = Sleef_expd4_u10avx2(neg_q_T);
        
        // S * e^(-qT) * N(d1)
        __m256d term1 = _mm256_mul_pd(S_vec, dq);
        term1 = _mm256_mul_pd(term1, Nd1);
        
        // K * e^(-rT) * N(d2)
        __m256d term2 = _mm256_mul_pd(K_vec, dr);
        term2 = _mm256_mul_pd(term2, Nd2);
        
        // call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
        __m256d call_value = _mm256_sub_pd(term1, term2);
        
        // Blend degenerate and computed values
        __m256d final_value = _mm256_blendv_pd(call_value, degenerate_value, degenerate_mask);
        
        // Store result
        _mm256_storeu_pd(result + i, final_value);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        if (vol[i] <= 0.0 || T[i] <= 0.0) {
            result[i] = std::max(0.0, S[i] - K[i]);
            continue;
        }
        
        double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) 
                  / (vol[i] * std::sqrt(T[i]));
        double d2 = d1 - vol[i] * std::sqrt(T[i]);
        
        double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
        double Nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
        
        result[i] = S[i] * std::exp(-q[i] * T[i]) * Nd1 - K[i] * std::exp(-r[i] * T[i]) * Nd2;
    }
}

/**
 * @brief Compute exp(x)*sqrt(y) in a single pass with SIMD optimization
 */
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

} // namespace opt
} // namespace alo
} // namespace engine