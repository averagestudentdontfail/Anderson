#include "vector.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cstdint>  // For uint64_t type
#include <cstring>  // For memcpy function

namespace engine {
namespace alo {
namespace opt {

// Fast polynomial approximation of exp(x) using AVX2
inline __m256d fast_exp_avx2(__m256d x) {
    // Constants for polynomial approximation
    const __m256d LOG2E = _mm256_set1_pd(1.4426950408889634);
    const __m256d LN2 = _mm256_set1_pd(0.6931471805599453);
    const __m256d ONE = _mm256_set1_pd(1.0);
    const __m256d C1 = _mm256_set1_pd(1.0);
    const __m256d C2 = _mm256_set1_pd(0.5);
    const __m256d C3 = _mm256_set1_pd(0.1666666666666667);
    const __m256d C4 = _mm256_set1_pd(0.041666666666666664);
    const __m256d C5 = _mm256_set1_pd(0.008333333333333333);
    
    // Apply range reduction: exp(x) = 2^i * exp(f) where i = floor(x/ln(2)), f = x - i*ln(2)
    __m256d tx = _mm256_mul_pd(x, LOG2E);
    __m256d ti = _mm256_floor_pd(tx);
    __m256d tf = _mm256_sub_pd(x, _mm256_mul_pd(ti, LN2));
    
    // Compute polynomial approximation of exp(f)
    __m256d result = ONE;
    result = _mm256_add_pd(result, _mm256_mul_pd(C1, tf));
    __m256d tf2 = _mm256_mul_pd(tf, tf);
    result = _mm256_add_pd(result, _mm256_mul_pd(C2, tf2));
    __m256d tf3 = _mm256_mul_pd(tf2, tf);
    result = _mm256_add_pd(result, _mm256_mul_pd(C3, tf3));
    __m256d tf4 = _mm256_mul_pd(tf3, tf);
    result = _mm256_add_pd(result, _mm256_mul_pd(C4, tf4));
    __m256d tf5 = _mm256_mul_pd(tf4, tf);
    result = _mm256_add_pd(result, _mm256_mul_pd(C5, tf5));
    
    // Scale by 2^i using scalar operations for simplicity and reliability
    alignas(32) double int_parts[4];
    alignas(32) double result_array[4];
    _mm256_store_pd(int_parts, ti);  // Store the integer parts
    _mm256_store_pd(result_array, result);  // Store the polynomial result
    
    for (int i = 0; i < 4; i++) {
        int exp = (int)int_parts[i] + 1023;  // IEEE-754 exponent bias
        exp = std::min(std::max(exp, 0), 2046);  // Clamp to valid range
        uint64_t bits = (uint64_t)exp << 52;
        double scale;
        memcpy(&scale, &bits, sizeof(double));
        result_array[i] *= scale;
    }
    
    return _mm256_load_pd(result_array);
}

// Fast approximation of erf(x) using polynomial
inline __m256d fast_erf_avx2(__m256d x) {
    // Abramowitz and Stegun approximation (error < 5e-4)
    const __m256d ONE = _mm256_set1_pd(1.0);
    const __m256d NEG_ONE = _mm256_set1_pd(-1.0);
    
    // Get absolute value and sign
    __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
    __m256d sign_x = _mm256_and_pd(x, _mm256_set1_pd(-0.0));
    sign_x = _mm256_or_pd(sign_x, ONE);  // sign(x) as +1.0 or -1.0
    
    // Constants for approximation
    const __m256d a1 = _mm256_set1_pd(0.254829592);
    const __m256d a2 = _mm256_set1_pd(-0.284496736);
    const __m256d a3 = _mm256_set1_pd(1.421413741);
    const __m256d a4 = _mm256_set1_pd(-1.453152027);
    const __m256d a5 = _mm256_set1_pd(1.061405429);
    const __m256d p = _mm256_set1_pd(0.3275911);
    
    // t = 1.0 / (1.0 + p * |x|)
    __m256d t = _mm256_div_pd(ONE, _mm256_add_pd(ONE, _mm256_mul_pd(p, abs_x)));
    
    // Polynomial approximation
    __m256d result = a5;
    result = _mm256_add_pd(_mm256_mul_pd(result, t), a4);
    result = _mm256_add_pd(_mm256_mul_pd(result, t), a3);
    result = _mm256_add_pd(_mm256_mul_pd(result, t), a2);
    result = _mm256_add_pd(_mm256_mul_pd(result, t), a1);
    result = _mm256_mul_pd(result, t);
    
    // erf(x) = sign(x) * (1 - exp(-x^2) * polynomial)
    __m256d x_squared = _mm256_mul_pd(x, x);
    __m256d neg_x_squared = _mm256_mul_pd(NEG_ONE, x_squared);
    __m256d exp_term = fast_exp_avx2(neg_x_squared);
    result = _mm256_mul_pd(exp_term, result);
    result = _mm256_sub_pd(ONE, result);
    
    // Apply sign
    return _mm256_mul_pd(sign_x, result);
}

// Fast normalCDF calculation using fast_erf_avx2
inline __m256d fast_normalCDF_avx2(__m256d x) {
    const __m256d HALF = _mm256_set1_pd(0.5);
    const __m256d ONE = _mm256_set1_pd(1.0);
    const __m256d SQRT_2_INV = _mm256_set1_pd(1.0 / std::sqrt(2.0));
    
    // normalCDF(x) = 0.5 * (1 + erf(x/sqrt(2)))
    __m256d scaled_x = _mm256_mul_pd(x, SQRT_2_INV);
    __m256d erf_term = fast_erf_avx2(scaled_x);
    __m256d term1 = _mm256_add_pd(ONE, erf_term);
    return _mm256_mul_pd(HALF, term1);
}

// Fast normalPDF calculation using fast_exp_avx2
inline __m256d fast_normalPDF_avx2(__m256d x) {
    const __m256d NEG_HALF = _mm256_set1_pd(-0.5);
    const __m256d INV_SQRT_2PI = _mm256_set1_pd(1.0 / std::sqrt(2.0 * M_PI));
    
    // normalPDF(x) = exp(-0.5 * x^2) / sqrt(2*PI)
    __m256d x_squared = _mm256_mul_pd(x, x);
    __m256d exponent = _mm256_mul_pd(NEG_HALF, x_squared);
    __m256d exp_term = fast_exp_avx2(exponent);
    
    return _mm256_mul_pd(exp_term, INV_SQRT_2PI);
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
    
    // For larger sizes, use AVX2 with vectorized processing
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        // Use fast approximation for exponential
        __m256d res = fast_exp_avx2(vec);
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
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        
        // Log doesn't have a direct AVX instruction, use scalar for now
        // This could be optimized further with polynomial approximation
        alignas(32) double values[4];
        _mm256_store_pd(values, vec);
        
        for (int j = 0; j < 4; j++) {
            values[j] = std::log(values[j]);
        }
        
        __m256d res = _mm256_load_pd(values);
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
    
    // Process in chunks of 4 using AVX2 built-in sqrt
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
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = fast_erf_avx2(vec);
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
    
    // Process in chunks of 4 using optimized AVX2 implementation
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = fast_normalCDF_avx2(vec);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
    }
}

void VectorMath::normalPDF(const double* x, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x[i] * x[i]);
        }
        return;
    }
    
    // Process in chunks of 4 using optimized AVX2 implementation
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = fast_normalPDF_avx2(vec);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x[i] * x[i]);
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
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Calculate vol * sqrt(T)
        __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
        __m256d vol_sqrt_T = _mm256_mul_pd(vol_vec, sqrt_T);
        
        // Calculate 0.5 * vol^2
        __m256d half = _mm256_set1_pd(0.5);
        __m256d vol_squared = _mm256_mul_pd(vol_vec, vol_vec);
        __m256d half_vol_squared = _mm256_mul_pd(half, vol_squared);
        
        // Calculate log(S/K)
        __m256d S_div_K = _mm256_div_pd(S_vec, K_vec);
        
        // Use aligned buffer for log calculation (no direct AVX2 log instruction)
        alignas(32) double values[4];
        _mm256_store_pd(values, S_div_K);
        for (int j = 0; j < 4; j++) {
            values[j] = std::log(values[j]);
        }
        __m256d log_S_div_K = _mm256_load_pd(values);
        
        // Calculate r - q + 0.5 * vol^2
        __m256d r_minus_q = _mm256_sub_pd(r_vec, q_vec);
        __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
        
        // Calculate (r - q + 0.5 * vol^2) * T
        __m256d drift_T = _mm256_mul_pd(drift, T_vec);
        
        // Calculate log(S/K) + (r - q + 0.5 * vol^2) * T
        __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
        
        // Calculate d1 = (log(S/K) + (r - q + 0.5 * vol^2) * T) / (vol * sqrt(T))
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
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d d1_vec = _mm256_loadu_pd(d1 + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Calculate vol * sqrt(T)
        __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
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

// Optimized implementation of Black-Scholes put calculation
void VectorMath::bsPut(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            if (vol[i] <= 0.0 || T[i] <= 0.0) {
                result[i] = std::max(0.0, K[i] - S[i]);
                continue;
            }
            
            double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) 
                      / (vol[i] * std::sqrt(T[i]));
            double d2 = d1 - vol[i] * std::sqrt(T[i]);
            
            double nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
            double nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
            
            result[i] = K[i] * std::exp(-r[i] * T[i]) * nd2 - S[i] * std::exp(-q[i] * T[i]) * nd1;
        }
        return;
    }
    
    // Allocate temporary arrays for intermediate calculations
    alignas(32) double d1_arr[size];
    alignas(32) double d2_arr[size];
    
    // Prepare vectors for bulk computing
    constexpr size_t CHUNK_SIZE = 4;
    
    // Calculate d1 and d2 for all values
    bsD1(S, K, r, q, vol, T, d1_arr, size);
    bsD2(d1_arr, vol, T, d2_arr, size);
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + CHUNK_SIZE <= size; i += CHUNK_SIZE) {
        // Negate d1 and d2 for put option
        __m256d d1_vec = _mm256_loadu_pd(d1_arr + i);
        __m256d d2_vec = _mm256_loadu_pd(d2_arr + i);
        __m256d neg_d1 = _mm256_sub_pd(_mm256_set1_pd(0.0), d1_vec);
        __m256d neg_d2 = _mm256_sub_pd(_mm256_set1_pd(0.0), d2_vec);
        
        // Calculate N(-d1) and N(-d2) using optimized normalCDF
        __m256d nd1 = fast_normalCDF_avx2(neg_d1);
        __m256d nd2 = fast_normalCDF_avx2(neg_d2);
        
        // Load input vectors
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Discount factors
        __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(_mm256_set1_pd(0.0), r_vec), T_vec);
        __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(_mm256_set1_pd(0.0), q_vec), T_vec);
        __m256d dr = fast_exp_avx2(neg_r_T);
        __m256d dq = fast_exp_avx2(neg_q_T);
        
        // K * e^(-rT) * N(-d2)
        __m256d term1 = _mm256_mul_pd(K_vec, dr);
        term1 = _mm256_mul_pd(term1, nd2);
        
        // S * e^(-qT) * N(-d1)
        __m256d term2 = _mm256_mul_pd(S_vec, dq);
        term2 = _mm256_mul_pd(term2, nd1);
        
        // Calculate put value: K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        __m256d put_val = _mm256_sub_pd(term1, term2);
        
        // Store the results
        _mm256_storeu_pd(result + i, put_val);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        double d1 = d1_arr[i];
        double d2 = d2_arr[i];
        
        double nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
        double nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
        
        result[i] = K[i] * std::exp(-r[i] * T[i]) * nd2 - S[i] * std::exp(-q[i] * T[i]) * nd1;
    }
}

// Optimized implementation of Black-Scholes call calculation
void VectorMath::bsCall(const double* S, const double* K, const double* r, const double* q, 
                       const double* vol, const double* T, double* result, size_t size) {
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            if (vol[i] <= 0.0 || T[i] <= 0.0) {
                result[i] = std::max(0.0, S[i] - K[i]);
                continue;
            }
            
            double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) 
                      / (vol[i] * std::sqrt(T[i]));
            double d2 = d1 - vol[i] * std::sqrt(T[i]);
            
            double nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
            double nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
            
            result[i] = S[i] * std::exp(-q[i] * T[i]) * nd1 - K[i] * std::exp(-r[i] * T[i]) * nd2;
        }
        return;
    }
    
    // Allocate temporary arrays for intermediate calculations
    alignas(32) double d1_arr[size];
    alignas(32) double d2_arr[size];
    
    // Prepare vectors for bulk computing
    constexpr size_t CHUNK_SIZE = 4;
    
    // Calculate d1 and d2 for all values
    bsD1(S, K, r, q, vol, T, d1_arr, size);
    bsD2(d1_arr, vol, T, d2_arr, size);
    
    // Process in chunks of 4 using AVX2
    size_t i = 0;
    for (; i + CHUNK_SIZE <= size; i += CHUNK_SIZE) {
        // Load d1 and d2
        __m256d d1_vec = _mm256_loadu_pd(d1_arr + i);
        __m256d d2_vec = _mm256_loadu_pd(d2_arr + i);
        
        // Calculate N(d1) and N(d2) using optimized normalCDF
        __m256d nd1 = fast_normalCDF_avx2(d1_vec);
        __m256d nd2 = fast_normalCDF_avx2(d2_vec);
        
        // Load input vectors
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Discount factors
        __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(_mm256_set1_pd(0.0), r_vec), T_vec);
        __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(_mm256_set1_pd(0.0), q_vec), T_vec);
        __m256d dr = fast_exp_avx2(neg_r_T);
        __m256d dq = fast_exp_avx2(neg_q_T);
        
        // S * exp(-q*T) * N(d1)
        __m256d term1 = _mm256_mul_pd(S_vec, dq);
        term1 = _mm256_mul_pd(term1, nd1);
        
        // K * exp(-r*T) * N(d2)
        __m256d term2 = _mm256_mul_pd(K_vec, dr);
        term2 = _mm256_mul_pd(term2, nd2);
        
        // Calculate call value: S * exp(-q*T) * N(d1) - K * exp(-r*T) * N(d2)
        __m256d call_val = _mm256_sub_pd(term1, term2);
        
        // Store the results
        _mm256_storeu_pd(result + i, call_val);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        double d1 = d1_arr[i];
        double d2 = d2_arr[i];
        
        double nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
        double nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
        
        result[i] = S[i] * std::exp(-q[i] * T[i]) * nd1 - K[i] * std::exp(-r[i] * T[i]) * nd2;
    }
}

} // namespace opt
} // namespace alo
} // namespace engine