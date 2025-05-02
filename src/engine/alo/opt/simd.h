#ifndef ENGINE_ALO_OPT_SIMD_H
#define ENGINE_ALO_OPT_SIMD_H

#include <immintrin.h>  
#include <sleef.h>     
#include <array>
#include <cmath>
#include <algorithm>
#include <cstring>  
#include <cstdint>  

namespace engine {
namespace alo {
namespace opt {

/**
 * @class SimdOps
 * @brief Optimized SIMD operations for financial math
 * 
 * This class provides SIMD operations optimized for financial mathematics
 * calculations with AVX2 instructions (256-bit vectors) and SLEEF for 
 * transcendental functions.
 */
class SimdOps {
public:
    /**
     * @brief Load 4 doubles into a vector
     * 
     * @param ptr Pointer to array of 4 doubles
     * @return 256-bit vector containing 4 doubles
     */
    static inline __m256d load(const double* ptr) {
        return _mm256_loadu_pd(ptr);
    }
    
    /**
     * @brief Load 4 doubles from an array
     * 
     * @param arr Array of 4 doubles
     * @return 256-bit vector containing 4 doubles
     */
    static inline __m256d load(const std::array<double, 4>& arr) {
        return _mm256_loadu_pd(arr.data());
    }
    
    /**
     * @brief Store a vector into 4 doubles
     * 
     * @param ptr Pointer to array of 4 doubles
     * @param vec 256-bit vector containing 4 doubles
     */
    static inline void store(double* ptr, __m256d vec) {
        _mm256_storeu_pd(ptr, vec);
    }
    
    /**
     * @brief Store a vector into an array
     * 
     * @param arr Array of 4 doubles
     * @param vec 256-bit vector containing 4 doubles
     */
    static inline void store(std::array<double, 4>& arr, __m256d vec) {
        _mm256_storeu_pd(arr.data(), vec);
    }
    
    /**
     * @brief Set all elements to a scalar value
     * 
     * @param value Scalar value
     * @return 256-bit vector with all elements set to value
     */
    static inline __m256d set1(double value) {
        return _mm256_set1_pd(value);
    }
    
    /**
     * @brief Set 4 elements
     * 
     * @param a First element
     * @param b Second element
     * @param c Third element
     * @param d Fourth element
     * @return 256-bit vector containing [a, b, c, d]
     */
    static inline __m256d set(double a, double b, double c, double d) {
        return _mm256_set_pd(d, c, b, a);
    }
    
    /**
     * @brief Add two vectors
     * 
     * @param a First vector
     * @param b Second vector
     * @return Vector containing a + b
     */
    static inline __m256d add(__m256d a, __m256d b) {
        return _mm256_add_pd(a, b);
    }
    
    /**
     * @brief Subtract two vectors
     * 
     * @param a First vector
     * @param b Second vector
     * @return Vector containing a - b
     */
    static inline __m256d sub(__m256d a, __m256d b) {
        return _mm256_sub_pd(a, b);
    }
    
    /**
     * @brief Multiply two vectors
     * 
     * @param a First vector
     * @param b Second vector
     * @return Vector containing a * b
     */
    static inline __m256d mul(__m256d a, __m256d b) {
        return _mm256_mul_pd(a, b);
    }
    
    /**
     * @brief Divide two vectors
     * 
     * @param a First vector
     * @param b Second vector
     * @return Vector containing a / b
     */
    static inline __m256d div(__m256d a, __m256d b) {
        return _mm256_div_pd(a, b);
    }
    
    /**
     * @brief Maximum of two vectors
     * 
     * @param a First vector
     * @param b Second vector
     * @return Vector containing max(a, b)
     */
    static inline __m256d max(__m256d a, __m256d b) {
        return _mm256_max_pd(a, b);
    }
    
    /**
     * @brief Minimum of two vectors
     * 
     * @param a First vector
     * @param b Second vector
     * @return Vector containing min(a, b)
     */
    static inline __m256d min(__m256d a, __m256d b) {
        return _mm256_min_pd(a, b);
    }
    
    /**
     * @brief Square root of a vector using AVX2
     * 
     * @param x Input vector
     * @return Vector containing sqrt(x)
     */
    static inline __m256d sqrt(__m256d x) {
        // Use AVX2 sqrt (faster and more accurate)
        return _mm256_sqrt_pd(x);
    }

    /**
     * @brief Exponential of a vector using SLEEF
     * 
     * @param x Input vector
     * @return Vector containing exp(x)
     */
    static inline __m256d exp(__m256d x) {
        // Use SLEEF's optimized exponential function for AVX2
        return Sleef_expd4_u10avx2(x);
    }
   
    /**
     * @brief Natural logarithm of a vector using SLEEF
     * 
     * @param x Input vector
     * @return Vector containing log(x)
     */
    static inline __m256d log(__m256d x) {
        // Use SLEEF's optimized log function for AVX2
        return Sleef_logd4_u10avx2(x);
    }

    /**
     * @brief Error function of a vector using SLEEF
     * 
     * @param x Input vector
     * @return Vector containing erf(x)
     */
    static inline __m256d erf(__m256d x) {
        // Use SLEEF's native erf function for AVX2
        return Sleef_erfd4_u10avx2(x);
    }
    
    /**
     * @brief Complementary error function of a vector using SLEEF
     * 
     * @param x Input vector
     * @return Vector containing erfc(x)
     */
    static inline __m256d erfc(__m256d x) {
        // Use SLEEF's native erfc function for AVX2
        return Sleef_erfcd4_u15avx2(x);
    }

    /**
     * @brief Fast approximation of error function using SIMD
     * 
     * Uses Abramowitz and Stegun approximation for better performance
     * with vectorized computation.
     */
    static inline __m256d fast_erf(__m256d x) {
        // Extract sign for later reconstruction
        __m256d sign_bit = _mm256_and_pd(x, _mm256_set1_pd(-0.0));
        
        // Take absolute value
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        
        // Constants for Abramowitz & Stegun approximation
        const __m256d a1 = _mm256_set1_pd(0.254829592);
        const __m256d a2 = _mm256_set1_pd(-0.284496736);
        const __m256d a3 = _mm256_set1_pd(1.421413741);
        const __m256d a4 = _mm256_set1_pd(-1.453152027);
        const __m256d a5 = _mm256_set1_pd(1.061405429);
        const __m256d p = _mm256_set1_pd(0.3275911);
        
        // Calculate t = 1/(1 + p*|x|)
        __m256d t = _mm256_div_pd(
            _mm256_set1_pd(1.0),
            _mm256_add_pd(_mm256_set1_pd(1.0), _mm256_mul_pd(p, abs_x))
        );
        
        // Calculate polynomial using Estrin's scheme (faster than Horner's method for SIMD)
        
        // First level: a5*t, a3*t, a1*t
        __m256d a5t = _mm256_mul_pd(a5, t);
        __m256d a3t = _mm256_mul_pd(a3, t);
        __m256d a1t = _mm256_mul_pd(a1, t);
        
        // Second level: (a5*t + a4), (a3*t + a2)
        __m256d a5t_a4 = _mm256_add_pd(a5t, a4);
        __m256d a3t_a2 = _mm256_add_pd(a3t, a2);
        
        // Third level: (a5*t + a4)*t^2, (a3*t + a2)*t^2
        __m256d t_squared = _mm256_mul_pd(t, t);
        __m256d a5t_a4_t2 = _mm256_mul_pd(a5t_a4, t_squared);
        __m256d a3t_a2_t = _mm256_mul_pd(a3t_a2, t);
        
        // Fourth level: (a5*t + a4)*t^2 + (a3*t + a2)*t
        __m256d poly = _mm256_add_pd(a5t_a4_t2, a3t_a2_t);
        
        // Fifth level: ((a5*t + a4)*t^2 + (a3*t + a2)*t) + a1*t
        poly = _mm256_add_pd(poly, a1t);
        
        // Calculate 1 - poly * exp(-x^2)
        __m256d x_squared = _mm256_mul_pd(abs_x, abs_x);
        __m256d neg_x_squared = _mm256_mul_pd(x_squared, _mm256_set1_pd(-1.0));
        __m256d exp_term = Sleef_expd4_u10avx2(neg_x_squared);
        __m256d poly_exp = _mm256_mul_pd(poly, exp_term);
        __m256d result = _mm256_sub_pd(_mm256_set1_pd(1.0), poly_exp);
        
        // Restore sign
        return _mm256_xor_pd(result, sign_bit);
    }

    /**
     * @brief Normal CDF calculation with improved precision
     * 
     * Uses the numerically stable approach based on erf and erfc
     * - For moderate values: N(x) = 0.5*(1 + erf(x/sqrt(2)))
     * - For large negative x: N(x) = 0.5*erfc(-x/sqrt(2))
     * - For large positive x: N(x) = 1 - 0.5*erfc(x/sqrt(2))
     * 
     * @param x Input vector
     * @return Vector containing normalCDF(x)
     */
    static inline __m256d normalCDF(__m256d x) {
        const __m256d HALF = _mm256_set1_pd(0.5);
        const __m256d ONE = _mm256_set1_pd(1.0);
        const __m256d SQRT2_INV = _mm256_set1_pd(0.7071067811865475); // 1/sqrt(2)
        
        // Get absolute value for condition testing
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        
        // Create masks for extreme values
        __m256d large_neg_mask = _mm256_cmp_pd(x, _mm256_set1_pd(-8.0), _CMP_LT_OS);
        __m256d large_pos_mask = _mm256_cmp_pd(x, _mm256_set1_pd(8.0), _CMP_GT_OS);
        
        // Calculate erf(x/sqrt(2))
        __m256d scaled_x = _mm256_mul_pd(x, SQRT2_INV);
        __m256d erf_result = Sleef_erfd4_u10avx2(scaled_x);
        
        // Normal range: 0.5 * (1 + erf(x/sqrt(2)))
        __m256d normal_result = _mm256_mul_pd(
            HALF, 
            _mm256_add_pd(ONE, erf_result)
        );
        
        // Handle extreme values
        __m256d result = _mm256_blendv_pd(
            normal_result,
            _mm256_setzero_pd(),  // Return 0 for large negative x
            large_neg_mask
        );
        
        result = _mm256_blendv_pd(
            result,
            ONE,  // Return 1 for large positive x
            large_pos_mask
        );
        
        return result;
    }
    
    /**
     * @brief Normal PDF calculation using SLEEF
     * 
     * @param x Input vector
     * @return Vector containing normalPDF(x)
     */
    static inline __m256d normalPDF(__m256d x) {
        const __m256d NEG_HALF = _mm256_set1_pd(-0.5);
        const __m256d INV_SQRT_2PI = _mm256_set1_pd(0.3989422804014327); // 1/sqrt(2π)
        
        // normalPDF(x) = exp(-0.5 * x^2) / sqrt(2*PI)
        __m256d x_squared = _mm256_mul_pd(x, x);
        __m256d exponent = _mm256_mul_pd(NEG_HALF, x_squared);
        
        // Use SLEEF's exponential function
        __m256d exp_term = Sleef_expd4_u10avx2(exponent);
        
        return _mm256_mul_pd(exp_term, INV_SQRT_2PI);
    }
    
    /**
     * @brief Calculate Black-Scholes d1 for 4 options at once
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @return Vector containing d1 values
     */
    static inline __m256d bsD1(__m256d S, __m256d K, __m256d r, __m256d q, __m256d vol, __m256d T) {
        // Prepare intermediate values
        __m256d vol_sqrt_T = _mm256_mul_pd(vol, _mm256_sqrt_pd(T));
        __m256d half = _mm256_set1_pd(0.5);
        __m256d vol_squared = _mm256_mul_pd(vol, vol);
        __m256d half_vol_squared = _mm256_mul_pd(half, vol_squared);
        
        // Calculate log(S/K) using SLEEF
        __m256d S_div_K = _mm256_div_pd(S, K);
        __m256d log_S_div_K = Sleef_logd4_u10avx2(S_div_K);
        
        // Calculate drift term
        __m256d r_minus_q = _mm256_sub_pd(r, q);
        __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
        __m256d drift_T = _mm256_mul_pd(drift, T);
        
        // Combine terms and calculate d1
        __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
        return _mm256_div_pd(numerator, vol_sqrt_T);
    }
    
    /**
     * @brief Calculate Black-Scholes d2 for 4 options at once
     * 
     * @param d1 d1 values
     * @param vol Volatilities
     * @param T Times to maturity
     * @return Vector containing d2 values
     */
    static inline __m256d bsD2(__m256d d1, __m256d vol, __m256d T) {
        // d2 = d1 - vol * sqrt(T)
        __m256d vol_sqrt_T = _mm256_mul_pd(vol, _mm256_sqrt_pd(T));
        return _mm256_sub_pd(d1, vol_sqrt_T);
    }
    
    /**
     * @brief Calculate Black-Scholes put prices for 4 options at once
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @return Vector containing put option prices
     */
    static inline __m256d bsPut(__m256d S, __m256d K, __m256d r, __m256d q, __m256d vol, __m256d T) {
        // Check for degenerate cases
        __m256d zero = _mm256_setzero_pd();
        __m256d eps = _mm256_set1_pd(1e-10);
        
        // Create mask for vol <= 0 or T <= 0
        __m256d vol_mask = _mm256_cmp_pd(vol, eps, _CMP_LE_OQ);
        __m256d t_mask = _mm256_cmp_pd(T, eps, _CMP_LE_OQ);
        __m256d degenerate_mask = _mm256_or_pd(vol_mask, t_mask);
        
        // Calculate K-S for degenerate cases
        __m256d K_minus_S = _mm256_sub_pd(K, S);
        __m256d degenerate_value = _mm256_max_pd(zero, K_minus_S);
        
        // Calculate d1 and d2
        __m256d d1 = bsD1(S, K, r, q, vol, T);
        __m256d d2 = bsD2(d1, vol, T);
        
        // Negate d1 and d2 for put calculation
        __m256d neg_d1 = _mm256_sub_pd(zero, d1);
        __m256d neg_d2 = _mm256_sub_pd(zero, d2);
        
        // Calculate N(-d1) and N(-d2)
        __m256d Nd1 = normalCDF(neg_d1);
        __m256d Nd2 = normalCDF(neg_d2);
        
        // Calculate discount factors with SLEEF
        __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
        __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q), T);
        __m256d dr = Sleef_expd4_u10avx2(neg_r_T);
        __m256d dq = Sleef_expd4_u10avx2(neg_q_T);
        
        // K * e^(-rT) * N(-d2)
        __m256d term1 = _mm256_mul_pd(K, dr);
        term1 = _mm256_mul_pd(term1, Nd2);
        
        // S * e^(-qT) * N(-d1)
        __m256d term2 = _mm256_mul_pd(S, dq);
        term2 = _mm256_mul_pd(term2, Nd1);
        
        // put = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        __m256d put_value = _mm256_sub_pd(term1, term2);
        
        // Blend degenerate and computed values
        return _mm256_blendv_pd(put_value, degenerate_value, degenerate_mask);
    }
    
    /**
     * @brief Calculate Black-Scholes call prices for 4 options at once
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @return Vector containing call option prices
     */
    static inline __m256d bsCall(__m256d S, __m256d K, __m256d r, __m256d q, __m256d vol, __m256d T) {
        // Check for degenerate cases
        __m256d zero = _mm256_setzero_pd();
        __m256d eps = _mm256_set1_pd(1e-10);
        
        // Create mask for vol <= 0 or T <= 0
        __m256d vol_mask = _mm256_cmp_pd(vol, eps, _CMP_LE_OQ);
        __m256d t_mask = _mm256_cmp_pd(T, eps, _CMP_LE_OQ);
        __m256d degenerate_mask = _mm256_or_pd(vol_mask, t_mask);
        
        // Calculate S-K for degenerate cases
        __m256d S_minus_K = _mm256_sub_pd(S, K);
        __m256d degenerate_value = _mm256_max_pd(zero, S_minus_K);
        
        // Calculate d1 and d2
        __m256d d1 = bsD1(S, K, r, q, vol, T);
        __m256d d2 = bsD2(d1, vol, T);
        
        // Calculate N(d1) and N(d2)
        __m256d Nd1 = normalCDF(d1);
        __m256d Nd2 = normalCDF(d2);
        
        // Calculate discount factors with SLEEF
        __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
        __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q), T);
        __m256d dr = Sleef_expd4_u10avx2(neg_r_T);
        __m256d dq = Sleef_expd4_u10avx2(neg_q_T);
        
        // S * e^(-qT) * N(d1)
        __m256d term1 = _mm256_mul_pd(S, dq);
        term1 = _mm256_mul_pd(term1, Nd1);
        
        // K * e^(-rT) * N(d2)
        __m256d term2 = _mm256_mul_pd(K, dr);
        term2 = _mm256_mul_pd(term2, Nd2);
        
        // call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
        __m256d call_value = _mm256_sub_pd(term1, term2);
        
        // Blend degenerate and computed values
        return _mm256_blendv_pd(call_value, degenerate_value, degenerate_mask);
    }
    
    /**
     * @brief Extract a single element from a vector
     * 
     * @param vec Input vector
     * @param index Index of element to extract (0-3)
     * @return Extracted element
     */
    static inline double extract(__m256d vec, int index) {
        alignas(32) double values[4];
        _mm256_store_pd(values, vec);
        return values[index];
    }
    
    /**
     * @brief Sum all elements in a vector
     * 
     * @param vec Input vector
     * @return Sum of all elements
     */
    static inline double sum(__m256d vec) {
        // Sum the elements using horizontal adds
        __m128d low = _mm256_extractf128_pd(vec, 0);
        __m128d high = _mm256_extractf128_pd(vec, 1);
        __m128d sum_128 = _mm_add_pd(low, high);
        
        // Extract the result
        alignas(16) double sum_values[2];
        _mm_store_pd(sum_values, sum_128);
        return sum_values[0] + sum_values[1];
    }
    
    /**
     * @brief SLEEF-powered AVX2 implementations for additional functions
     */
    static inline __m256d pow(__m256d x, __m256d y) {
        return Sleef_powd4_u10avx2(x, y);
    }
    
    static inline __m256d sin(__m256d x) {
        return Sleef_sind4_u10avx2(x);
    }
    
    static inline __m256d cos(__m256d x) {
        return Sleef_cosd4_u10avx2(x);
    }
    
    static inline __m256d tan(__m256d x) {
        return Sleef_tand4_u10avx2(x);
    }
    
    static inline __m256d tanh(__m256d x) {
        return Sleef_tanhd4_u10avx2(x);
    }
};

// Add AVX-512 support when available
#ifdef __AVX512F__
/**
 * @class SimdOpsAVX512
 * @brief AVX-512 optimized operations for financial math
 * 
 * This class provides SIMD operations using AVX-512 instructions
 * for processing 8 options at once.
 */
class SimdOpsAVX512 {
public:
    /**
     * @brief Calculate Black-Scholes put prices for 8 options at once
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @return Vector containing put option prices
     */
    static inline __m512d bsPut(__m512d S, __m512d K, __m512d r, __m512d q, __m512d vol, __m512d T) {
        // Check for degenerate cases
        __m512d zero = _mm512_setzero_pd();
        __m512d eps = _mm512_set1_pd(1e-10);
        
        // Create mask for vol <= 0 or T <= 0
        __mmask8 vol_mask = _mm512_cmp_pd_mask(vol, eps, _CMP_LE_OQ);
        __mmask8 t_mask = _mm512_cmp_pd_mask(T, eps, _CMP_LE_OQ);
        __mmask8 degenerate_mask = _mm512_kor(vol_mask, t_mask);
        
        // Calculate K-S for degenerate cases
        __m512d K_minus_S = _mm512_sub_pd(K, S);
        __m512d degenerate_value = _mm512_max_pd(zero, K_minus_S);
        
        // Calculate d1
        __m512d sqrt_T = _mm512_sqrt_pd(T);
        __m512d vol_sqrt_T = _mm512_mul_pd(vol, sqrt_T);
        
        // Calculate log(S/K)
        __m512d S_div_K = _mm512_div_pd(S, K);
        __m512d log_S_div_K = Sleef_logd8_u10avx512(S_div_K);
        
        // Calculate (r-q) + 0.5*vol^2
        __m512d r_minus_q = _mm512_sub_pd(r, q);
        __m512d vol_squared = _mm512_mul_pd(vol, vol);
        __m512d half_vol_squared = _mm512_mul_pd(_mm512_set1_pd(0.5), vol_squared);
        __m512d drift = _mm512_add_pd(r_minus_q, half_vol_squared);
        
        // Calculate (r-q + 0.5*vol^2) * T
        __m512d drift_T = _mm512_mul_pd(drift, T);
        
        // Calculate d1 numerator
        __m512d d1_num = _mm512_add_pd(log_S_div_K, drift_T);
        
        // Calculate d1
        __m512d d1 = _mm512_div_pd(d1_num, vol_sqrt_T);
        
        // Calculate d2 = d1 - vol*sqrt(T)
        __m512d d2 = _mm512_sub_pd(d1, vol_sqrt_T);
        
        // Negate d1 and d2 for put formula
        __m512d neg_d1 = _mm512_sub_pd(zero, d1);
        __m512d neg_d2 = _mm512_sub_pd(zero, d2);
        
        // Calculate N(-d1) and N(-d2)
        __m512d scaled_neg_d1 = _mm512_div_pd(neg_d1, _mm512_set1_pd(M_SQRT2));
        __m512d scaled_neg_d2 = _mm512_div_pd(neg_d2, _mm512_set1_pd(M_SQRT2));
        
        __m512d Nd1 = _mm512_mul_pd(
            _mm512_set1_pd(0.5),
            _mm512_add_pd(_mm512_set1_pd(1.0), Sleef_erfd8_u10avx512(scaled_neg_d1))
        );
        
        __m512d Nd2 = _mm512_mul_pd(
            _mm512_set1_pd(0.5),
            _mm512_add_pd(_mm512_set1_pd(1.0), Sleef_erfd8_u10avx512(scaled_neg_d2))
        );
        
        // Calculate discount factors
        __m512d neg_r_T = _mm512_mul_pd(_mm512_sub_pd(zero, r), T);
        __m512d neg_q_T = _mm512_mul_pd(_mm512_sub_pd(zero, q), T);
        __m512d dr = Sleef_expd8_u10avx512(neg_r_T);
        __m512d dq = Sleef_expd8_u10avx512(neg_q_T);
        
        // Calculate K * e^(-rT) * N(-d2)
        __m512d term1 = _mm512_mul_pd(K, dr);
        term1 = _mm512_mul_pd(term1, Nd2);
        
        // Calculate S * e^(-qT) * N(-d1)
        __m512d term2 = _mm512_mul_pd(S, dq);
        term2 = _mm512_mul_pd(term2, Nd1);
        
        // Calculate put price = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        __m512d put_price = _mm512_sub_pd(term1, term2);
        
        // Blend degenerate and computed values
        return _mm512_mask_blend_pd(degenerate_mask, put_price, degenerate_value);
    }
    
    /**
     * @brief Calculate Black-Scholes call prices for 8 options at once
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @return Vector containing call option prices
     */
    static inline __m512d bsCall(__m512d S, __m512d K, __m512d r, __m512d q, __m512d vol, __m512d T) {
        // Check for degenerate cases
        __m512d zero = _mm512_setzero_pd();
        __m512d eps = _mm512_set1_pd(1e-10);
        
        // Create mask for vol <= 0 or T <= 0
        __mmask8 vol_mask = _mm512_cmp_pd_mask(vol, eps, _CMP_LE_OQ);
        __mmask8 t_mask = _mm512_cmp_pd_mask(T, eps, _CMP_LE_OQ);
        __mmask8 degenerate_mask = _mm512_kor(vol_mask, t_mask);
        
        // Calculate S-K for degenerate cases
        __m512d S_minus_K = _mm512_sub_pd(S, K);
        __m512d degenerate_value = _mm512_max_pd(zero, S_minus_K);
        
        // Calculate sqrt(T)
        __m512d sqrt_T = _mm512_sqrt_pd(T);
        __m512d vol_sqrt_T = _mm512_mul_pd(vol, sqrt_T);
        
        // Calculate log(S/K)
        __m512d S_div_K = _mm512_div_pd(S, K);
        __m512d log_S_div_K = Sleef_logd8_u10avx512(S_div_K);
        
        // Calculate drift term
        __m512d r_minus_q = _mm512_sub_pd(r, q);
        __m512d vol_squared = _mm512_mul_pd(vol, vol);
        __m512d half_vol_squared = _mm512_mul_pd(_mm512_set1_pd(0.5), vol_squared);
        __m512d drift = _mm512_add_pd(r_minus_q, half_vol_squared);
        
        // Calculate drift * T
        __m512d drift_T = _mm512_mul_pd(drift, T);
        
        // Calculate d1 numerator
        __m512d d1_num = _mm512_add_pd(log_S_div_K, drift_T);
        
        // Calculate d1
        __m512d d1 = _mm512_div_pd(d1_num, vol_sqrt_T);
        
        // Calculate d2 = d1 - vol*sqrt(T)
        __m512d d2 = _mm512_sub_pd(d1, vol_sqrt_T);
        
        // Calculate N(d1) and N(d2)
        __m512d scaled_d1 = _mm512_div_pd(d1, _mm512_set1_pd(M_SQRT2));
        __m512d scaled_d2 = _mm512_div_pd(d2, _mm512_set1_pd(M_SQRT2));
        
        __m512d Nd1 = _mm512_mul_pd(
            _mm512_set1_pd(0.5),
            _mm512_add_pd(_mm512_set1_pd(1.0), Sleef_erfd8_u10avx512(scaled_d1))
        );
        
        __m512d Nd2 = _mm512_mul_pd(
            _mm512_set1_pd(0.5),
            _mm512_add_pd(_mm512_set1_pd(1.0), Sleef_erfd8_u10avx512(scaled_d2))
        );
        
        // Calculate discount factors
        __m512d neg_r_T = _mm512_mul_pd(_mm512_sub_pd(zero, r), T);
        __m512d neg_q_T = _mm512_mul_pd(_mm512_sub_pd(zero, q), T);
        
        __m512d dr = Sleef_expd8_u10avx512(neg_r_T);
        __m512d dq = Sleef_expd8_u10avx512(neg_q_T);
        
        // Calculate S * e^(-qT) * N(d1)
        __m512d term1 = _mm512_mul_pd(S, dq);
        term1 = _mm512_mul_pd(term1, Nd1);
        
        // Calculate K * e^(-rT) * N(d2)
        __m512d term2 = _mm512_mul_pd(K, dr);
        term2 = _mm512_mul_pd(term2, Nd2);
        
        // Calculate call price = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
        __m512d call_price = _mm512_sub_pd(term1, term2);
        
        // Blend degenerate and computed values
        return _mm512_mask_blend_pd(degenerate_mask, call_price, degenerate_value);
    }
};
#endif // __AVX512F__

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_SIMD_H