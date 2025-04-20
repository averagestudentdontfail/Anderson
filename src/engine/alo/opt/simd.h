#ifndef ENGINE_ALO_OPT_SIMD_H
#define ENGINE_ALO_OPT_SIMD_H

#include <immintrin.h>  
#include <sleef.h>     
#include <array>
#include <cmath>
#include <algorithm>
#include <cstring>  
#include <cstdint>  

// Platform-specific headers for CPU feature detection
#if defined(_MSC_VER)
    // MSVC
    #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
    // GCC or Clang
    #include <cpuid.h>
#endif

namespace engine {
namespace alo {
namespace opt {

/**
 * @brief SIMD support detection and CPU feature detection
 */
class SimdDetect {
public:
    /**
     * @brief Check if AVX2 instructions are supported
     * 
     * @return True if AVX2 is supported, false otherwise
     */
    static bool hasAVX2() {
        static const bool has_avx2 = checkAVX2Support();
        return has_avx2;
    }
    
    /**
     * @brief Check if AVX-512 instructions are supported
     * 
     * @return True if AVX-512 is supported, false otherwise
     */
    static bool hasAVX512() {
        static const bool has_avx512 = checkAVX512Support();
        return has_avx512;
    }
    
private:
    static bool checkAVX2Support() {
        // Cross-platform implementation of AVX2 detection
#if defined(_MSC_VER)
        // MSVC implementation
        int cpuInfo[4];
        int nIds = 0;
        
        __cpuidex(cpuInfo, 0, 0);
        nIds = cpuInfo[0];
        
        bool avxSupported = false;
        bool avx2Supported = false;
        
        if (nIds >= 1) {
            __cpuidex(cpuInfo, 1, 0);
            avxSupported = (cpuInfo[2] & (1 << 28)) != 0;  // Check for AVX
        }
        
        if (nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            avx2Supported = (cpuInfo[1] & (1 << 5)) != 0;  // Check for AVX2
        }
        
        return avxSupported && avx2Supported;
#elif defined(__GNUC__) || defined(__clang__)
        // GCC or Clang implementation
        unsigned int eax, ebx, ecx, edx;
        
        // Check for basic CPUID support
        if (__get_cpuid_max(0, nullptr) < 7) {
            return false;
        }
        
        // Check for AVX support (CPUID.1:ECX.AVX[bit 28])
        __cpuid(1, eax, ebx, ecx, edx);
        bool avxSupported = (ecx & (1 << 28)) != 0;
        
        // Check for AVX2 support (CPUID.7.0:EBX.AVX2[bit 5])
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        bool avx2Supported = (ebx & (1 << 5)) != 0;
        
        return avxSupported && avx2Supported;
#else
        // Fallback for other platforms - assume AVX2 is not available
        return false;
#endif
    }

    static bool checkAVX512Support() {
        // Cross-platform implementation of AVX-512 detection
#if defined(_MSC_VER)
        // MSVC implementation
        int cpuInfo[4];
        int nIds = 0;
        
        __cpuidex(cpuInfo, 0, 0);
        nIds = cpuInfo[0];
        
        if (nIds >= 7) {
            __cpuidex(cpuInfo, 7, 0);
            // Check for AVX-512 Foundation
            return (cpuInfo[1] & (1 << 16)) != 0;
        }
        
        return false;
#elif defined(__GNUC__) || defined(__clang__)
        // GCC or Clang implementation
        unsigned int eax, ebx, ecx, edx;
        
        // Check for basic CPUID support
        if (__get_cpuid_max(0, nullptr) < 7) {
            return false;
        }
        
        // Check for AVX-512 Foundation (CPUID.7.0:EBX.AVX512F[bit 16])
        __cpuid_count(7, 0, eax, ebx, ecx, edx);
        return (ebx & (1 << 16)) != 0;
#else
        // Fallback for other platforms - assume AVX-512 is not available
        return false;
#endif
    }
};

/**
 * @brief Check if SIMD optimization is available
 * 
 * @return True if SIMD optimization is available, false otherwise
 */
inline bool isSimdAvailable() {
    return SimdDetect::hasAVX2();
}

/**
 * @brief Threshold for using SIMD operations
 * 
 * Only use SIMD for operations with size >= this threshold
 */
constexpr size_t SIMD_THRESHOLD = 32;

/**
 * @brief Check if we should use SIMD for a given data size
 * 
 * @param size Size of the data to process
 * @return True if SIMD should be used, false otherwise
 */
inline bool shouldUseSimd(size_t size) {
    return isSimdAvailable() && size >= SIMD_THRESHOLD;
}

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
     * @brief Square root of a vector using SLEEF or AVX2
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
     * @brief Error function of a vector using polynomial approximation
     * 
     * @param x Input vector
     * @return Vector containing erf(x)
     */
    static inline __m256d erf(__m256d x) {
        // Using Abramowitz and Stegun 7.1.26 approximation for erf
        // erf(x) = 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵)e^(-x²) 
        // where t = 1/(1 + px), p = 0.3275911
        const __m256d ONE = _mm256_set1_pd(1.0);
        const __m256d NEG_ONE = _mm256_set1_pd(-1.0);
        
        // Handle sign of x separately - erf is an odd function
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x); // |x|
        __m256d sign_x = _mm256_or_pd(
            _mm256_and_pd(x, _mm256_set1_pd(-0.0)),
            ONE
        ); // sign(x) = x < 0 ? -1 : 1
        
        // Constants for A&S approximation
        const __m256d a1 = _mm256_set1_pd(0.254829592);
        const __m256d a2 = _mm256_set1_pd(-0.284496736);
        const __m256d a3 = _mm256_set1_pd(1.421413741);
        const __m256d a4 = _mm256_set1_pd(-1.453152027);
        const __m256d a5 = _mm256_set1_pd(1.061405429);
        const __m256d p = _mm256_set1_pd(0.3275911);
        
        // Calculate t = 1.0 / (1.0 + p * |x|)
        __m256d t = _mm256_div_pd(ONE, _mm256_add_pd(ONE, _mm256_mul_pd(p, abs_x)));
        
        // Calculate polynomial
        __m256d poly = a1;
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), a2);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), a3);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), a4);
        poly = _mm256_add_pd(_mm256_mul_pd(poly, t), a5);
        poly = _mm256_mul_pd(poly, t);
        
        // Calculate exp(-x²)
        __m256d x_squared = _mm256_mul_pd(abs_x, abs_x);
        __m256d exp_term = Sleef_expd4_u10avx2(_mm256_mul_pd(NEG_ONE, x_squared));
        
        // Multiply polynomial by exp term
        __m256d error_term = _mm256_mul_pd(poly, exp_term);
        
        // Calculate erf(x) = sign(x) * (1 - error_term)
        __m256d abs_erf = _mm256_sub_pd(ONE, error_term);
        
        // Apply sign - erf is an odd function
        __m256d pos_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ);
        __m256d result = _mm256_blendv_pd(
            _mm256_mul_pd(abs_erf, NEG_ONE), // negative x: -abs_erf
            abs_erf,                         // positive x: abs_erf
            pos_mask
        );
        
        return result;
    }

    /**
    * @brief Normal CDF calculation using Abramowitz & Stegun approximation
    * 
    * @param x Input vector
    * @return Vector containing normalCDF(x)
    */
    static inline __m256d normalCDF(__m256d x) {
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
        __m256d abs_x = _mm256_andnot_pd(_mm256_set1_pd(-0.0), x);
        __m256d sign_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_GE_OQ);
        
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
        return _mm256_blendv_pd(normal_result, extreme_result, large_mask);
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
        
        // If all are degenerate, just return the intrinsic values
        if (_mm256_movemask_pd(degenerate_mask) == 0xF) {
            return degenerate_value;
        }
        
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
        
        // If all are degenerate, just return the intrinsic values
        if (_mm256_movemask_pd(degenerate_mask) == 0xF) {
            return degenerate_value;
        }
        
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

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_SIMD_H