#ifndef SIMD_OPS_H
#define SIMD_OPS_H

#include <immintrin.h>
#include <array>
#include <cmath>

namespace simd {

/**
 * @brief SIMD operations for AVX2 (256-bit vectors)
 */
class SimdOps {
public:
    // Load 4 doubles into a vector
    static inline __m256d load(const double* ptr) {
        return _mm256_loadu_pd(ptr);
    }
    
    // Load 4 doubles from an array
    static inline __m256d load(const std::array<double, 4>& arr) {
        return _mm256_loadu_pd(arr.data());
    }
    
    // Store a vector into 4 doubles
    static inline void store(double* ptr, __m256d vec) {
        _mm256_storeu_pd(ptr, vec);
    }
    
    // Store a vector into an array
    static inline void store(std::array<double, 4>& arr, __m256d vec) {
        _mm256_storeu_pd(arr.data(), vec);
    }
    
    // Set all elements to a scalar value
    static inline __m256d set1(double value) {
        return _mm256_set1_pd(value);
    }
    
    // Set 4 elements
    static inline __m256d set(double a, double b, double c, double d) {
        return _mm256_set_pd(d, c, b, a);
    }
    
    // Math operations
    static inline __m256d add(__m256d a, __m256d b) {
        return _mm256_add_pd(a, b);
    }
    
    static inline __m256d sub(__m256d a, __m256d b) {
        return _mm256_sub_pd(a, b);
    }
    
    static inline __m256d mul(__m256d a, __m256d b) {
        return _mm256_mul_pd(a, b);
    }
    
    static inline __m256d div(__m256d a, __m256d b) {
        return _mm256_div_pd(a, b);
    }
    
    static inline __m256d max(__m256d a, __m256d b) {
        return _mm256_max_pd(a, b);
    }
    
    static inline __m256d min(__m256d a, __m256d b) {
        return _mm256_min_pd(a, b);
    }
    
    // Natural logarithm approximation
    static inline __m256d log(__m256d x) {
        // AVX2 doesn't have a direct log function, use polynomial approximation
        // or call scalar function 4 times
        std::array<double, 4> values;
        store(values, x);
        
        values[0] = std::log(values[0]);
        values[1] = std::log(values[1]);
        values[2] = std::log(values[2]);
        values[3] = std::log(values[3]);
        
        return load(values);
    }
    
    // Exponential function approximation
    static inline __m256d exp(__m256d x) {
        // AVX2 doesn't have a direct exp function, use polynomial approximation
        // or call scalar function 4 times
        std::array<double, 4> values;
        store(values, x);
        
        values[0] = std::exp(values[0]);
        values[1] = std::exp(values[1]);
        values[2] = std::exp(values[2]);
        values[3] = std::exp(values[3]);
        
        return load(values);
    }
    
    // Square root
    static inline __m256d sqrt(__m256d x) {
        return _mm256_sqrt_pd(x);
    }
    
    // Error function approximation for normal CDF calculation
    static inline __m256d erf(__m256d x) {
        std::array<double, 4> values;
        store(values, x);
        
        values[0] = std::erf(values[0]);
        values[1] = std::erf(values[1]);
        values[2] = std::erf(values[2]);
        values[3] = std::erf(values[3]);
        
        return load(values);
    }
    
    // Normal CDF approximation
    static inline __m256d normalCDF(__m256d x) {
        __m256d half = set1(0.5);
        __m256d sqrt2inv = set1(1.0 / std::sqrt(2.0));
        
        // normalCDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
        return add(half, mul(half, erf(mul(x, sqrt2inv))));
    }
    
    // Normal PDF approximation
    static inline __m256d normalPDF(__m256d x) {
        __m256d minusHalf = set1(-0.5);
        __m256d invSqrt2Pi = set1(1.0 / std::sqrt(2.0 * M_PI));
        
        // normalPDF(x) = exp(-0.5 * x^2) / sqrt(2 * PI)
        return mul(exp(mul(minusHalf, mul(x, x))), invSqrt2Pi);
    }
    
    // Black-Scholes d1 calculation for 4 options at once
    static inline __m256d bsD1(__m256d S, __m256d K, __m256d r, __m256d q, __m256d vol, __m256d T) {
        __m256d vol_sqrt_T = mul(vol, sqrt(T));
        __m256d half_vol_squared = mul(set1(0.5), mul(vol, vol));
        
        // d1 = (ln(S/K) + (r - q + 0.5 * vol^2) * T) / (vol * sqrt(T))
        return div(
            add(
                log(div(S, K)),
                mul(add(sub(r, q), half_vol_squared), T)
            ),
            vol_sqrt_T
        );
    }
    
    // Black-Scholes d2 calculation for 4 options at once
    static inline __m256d bsD2(__m256d d1, __m256d vol, __m256d T) {
        // d2 = d1 - vol * sqrt(T)
        return sub(d1, mul(vol, sqrt(T)));
    }
    
    // Black-Scholes put pricing for 4 options at once
    static inline __m256d bsPut(__m256d S, __m256d K, __m256d r, __m256d q, __m256d vol, __m256d T) {
        __m256d zero = set1(0.0);
        
        // Handle degenerate cases
        __m256d vol_mask = _mm256_cmp_pd(vol, zero, _CMP_GT_OQ);
        __m256d T_mask = _mm256_cmp_pd(T, zero, _CMP_GT_OQ);
        __m256d valid_mask = _mm256_and_pd(vol_mask, T_mask);
        
        if (_mm256_movemask_pd(valid_mask) == 0) {
            // All elements are degenerate, return max(0, K-S)
            return max(zero, sub(K, S));
        }
        
        __m256d d1 = bsD1(S, K, r, q, vol, T);
        __m256d d2 = bsD2(d1, vol, T);
        
        // Negative d1 and d2 for put calculation
        __m256d neg_d1 = sub(zero, d1);
        __m256d neg_d2 = sub(zero, d2);
        
        // N(-d1) and N(-d2)
        __m256d Nd1 = normalCDF(neg_d1);
        __m256d Nd2 = normalCDF(neg_d2);
        
        // discount factors
        __m256d dr = exp(mul(sub(zero, r), T));
        __m256d dq = exp(mul(sub(zero, q), T));
        
        // put = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
        __m256d put_value = sub(
            mul(mul(K, dr), Nd2),
            mul(mul(S, dq), Nd1)
        );
        
        // For degenerate cases, use max(0, K-S)
        __m256d intrinsic = max(zero, sub(K, S));
        
        return _mm256_blendv_pd(intrinsic, put_value, valid_mask);
    }
    
    // Extract a single element from a vector
    static inline double extract(__m256d vec, int index) {
        std::array<double, 4> values;
        store(values, vec);
        return values[index];
    }
    
    // Sum all elements in a vector
    static inline double sum(__m256d vec) {
        // Sum the elements of the vector
        __m128d low = _mm256_extractf128_pd(vec, 0);
        __m128d high = _mm256_extractf128_pd(vec, 1);
        __m128d sum_128 = _mm_add_pd(low, high);
        double sum_values[2];
        _mm_storeu_pd(sum_values, sum_128);
        return sum_values[0] + sum_values[1];
    }
};

} 

#endif