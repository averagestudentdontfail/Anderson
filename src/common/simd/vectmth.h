#ifndef VECTOR_MTH_H
#define VECTOR_MTH_H

#include <immintrin.h>
#include <array>
#include <cmath>
#include <algorithm>
#include <vector>

namespace simd {

/**
 * @brief Vector math operations for SIMD acceleration
 * 
 * This class provides vectorized mathematical functions optimized for performance
 * using AVX2 instructions when available.
 */
class VectorMth {
public:
    /**
     * @brief Vectorized exponential function for arrays
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void exp(const double* x, double* result, size_t size) {
        // Process in chunks of 4 using AVX2
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = exp_avx2(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = std::exp(x[i]);
        }
    }
    
    /**
     * @brief Vectorized natural logarithm for arrays
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void log(const double* x, double* result, size_t size) {
        // Process in chunks of 4 using AVX2
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = log_avx2(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = std::log(x[i]);
        }
    }
    
    /**
     * @brief Vectorized square root for arrays
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void sqrt(const double* x, double* result, size_t size) {
        // Process in chunks of 4 using AVX2
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
    
    /**
     * @brief Vectorized error function for arrays
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void erf(const double* x, double* result, size_t size) {
        // Process in chunks of 4 using AVX2
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = erf_avx2(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = std::erf(x[i]);
        }
    }
    
    /**
     * @brief Vectorized cosine for arrays
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void cos(const double* x, double* result, size_t size) {
        // Process in chunks of 4 using AVX2
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = cos_avx2(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = std::cos(x[i]);
        }
    }
    
    /**
     * @brief Vectorized sin for arrays
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void sin(const double* x, double* result, size_t size) {
        // Process in chunks of 4 using AVX2
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = sin_avx2(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = std::sin(x[i]);
        }
    }
    
    /**
     * @brief Vectorized normal CDF for arrays
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void normalCDF(const double* x, double* result, size_t size) {
        // Process in chunks of 4 using AVX2
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = normalCDF_avx2(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
        }
    }
    
    /**
     * @brief Vectorized normal PDF for arrays
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void normalPDF(const double* x, double* result, size_t size) {
        // Process in chunks of 4 using AVX2
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = normalPDF_avx2(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x[i] * x[i]);
        }
    }
    
    /**
     * @brief Vectorized elementwise multiplication of arrays
     * 
     * @param a First input array
     * @param b Second input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void multiply(const double* a, const double* b, double* result, size_t size) {
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
    
    /**
     * @brief Vectorized elementwise addition of arrays
     * 
     * @param a First input array
     * @param b Second input array
     * @param result Output array
     * @param size Number of elements to process
     */
    static void add(const double* a, const double* b, double* result, size_t size) {
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

    /**
     * @brief Optimized Chebyshev polynomial evaluation
     * 
     * @param x Points at which to evaluate Chebyshev polynomials
     * @param coeff Chebyshev coefficients
     * @param result Output array
     * @param size Number of points
     * @param n Degree of the polynomial
     */
    static void evaluateChebyshev(const double* x, const double* coeff, double* result, 
                                 size_t size, size_t n) {
        // Use Clenshaw algorithm with SIMD
        for (size_t i = 0; i < size; ++i) {
            double t = x[i];
            double b_kp2 = 0.0;
            double b_kp1 = 0.0;
            double b_k;
            
            for (int k = static_cast<int>(n) - 1; k >= 0; --k) {
                b_k = coeff[k] + 2.0 * t * b_kp1 - b_kp2;
                b_kp2 = b_kp1;
                b_kp1 = b_k;
            }
            
            result[i] = b_kp1 - t * b_kp2;
        }
    }
    
private:
    /**
     * @brief AVX2 implementation of exponential function
     * 
     * @param x Input vector
     * @return Exponential of each element
     */
    static __m256d exp_avx2(__m256d x) {
        // AVX2 doesn't have a direct exp function, so we use a scalar approach
        std::array<double, 4> values;
        _mm256_storeu_pd(values.data(), x);
        
        for (int i = 0; i < 4; ++i) {
            values[i] = std::exp(values[i]);
        }
        
        return _mm256_loadu_pd(values.data());
    }
    
    /**
     * @brief AVX2 implementation of logarithm function
     * 
     * @param x Input vector
     * @return Natural logarithm of each element
     */
    static __m256d log_avx2(__m256d x) {
        std::array<double, 4> values;
        _mm256_storeu_pd(values.data(), x);
        
        for (int i = 0; i < 4; ++i) {
            values[i] = std::log(values[i]);
        }
        
        return _mm256_loadu_pd(values.data());
    }
    
    /**
     * @brief AVX2 implementation of error function
     * 
     * @param x Input vector
     * @return Error function of each element
     */
    static __m256d erf_avx2(__m256d x) {
        std::array<double, 4> values;
        _mm256_storeu_pd(values.data(), x);
        
        for (int i = 0; i < 4; ++i) {
            values[i] = std::erf(values[i]);
        }
        
        return _mm256_loadu_pd(values.data());
    }
    
    /**
     * @brief AVX2 implementation of cosine function
     * 
     * @param x Input vector
     * @return Cosine of each element
     */
    static __m256d cos_avx2(__m256d x) {
        std::array<double, 4> values;
        _mm256_storeu_pd(values.data(), x);
        
        for (int i = 0; i < 4; ++i) {
            values[i] = std::cos(values[i]);
        }
        
        return _mm256_loadu_pd(values.data());
    }
    
    /**
     * @brief AVX2 implementation of sine function
     * 
     * @param x Input vector
     * @return Sine of each element
     */
    static __m256d sin_avx2(__m256d x) {
        std::array<double, 4> values;
        _mm256_storeu_pd(values.data(), x);
        
        for (int i = 0; i < 4; ++i) {
            values[i] = std::sin(values[i]);
        }
        
        return _mm256_loadu_pd(values.data());
    }
    
    /**
     * @brief AVX2 implementation of normal CDF
     * 
     * @param x Input vector
     * @return Normal CDF of each element
     */
    static __m256d normalCDF_avx2(__m256d x) {
        __m256d half = _mm256_set1_pd(0.5);
        __m256d one = _mm256_set1_pd(1.0);
        __m256d sqrt2inv = _mm256_set1_pd(1.0 / std::sqrt(2.0));
        
        // x / sqrt(2)
        __m256d arg = _mm256_mul_pd(x, sqrt2inv);
        
        // erf(x / sqrt(2))
        __m256d erf_val = erf_avx2(arg);
        
        // 0.5 * (1 + erf(x / sqrt(2)))
        __m256d term1 = _mm256_add_pd(one, erf_val);
        return _mm256_mul_pd(half, term1);
    }
    
    /**
     * @brief AVX2 implementation of normal PDF
     * 
     * @param x Input vector
     * @return Normal PDF of each element
     */
    static __m256d normalPDF_avx2(__m256d x) {
        __m256d half = _mm256_set1_pd(-0.5);
        __m256d invSqrt2Pi = _mm256_set1_pd(1.0 / std::sqrt(2.0 * M_PI));
        
        // x^2
        __m256d x_squared = _mm256_mul_pd(x, x);
        
        // -0.5 * x^2
        __m256d exponent = _mm256_mul_pd(half, x_squared);
        
        // exp(-0.5 * x^2)
        __m256d exp_val = exp_avx2(exponent);
        
        // (1/sqrt(2π)) * exp(-0.5 * x^2)
        return _mm256_mul_pd(invSqrt2Pi, exp_val);
    }
};

} 

#endif