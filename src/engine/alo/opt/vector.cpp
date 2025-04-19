#include "vector.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>

namespace engine {
namespace alo {
namespace opt {

/**
 * @brief Vectorized operations for arrays of data
 */
void VectorMath::exp(const double* x, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = simd_exp(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = std::exp(x[i]);
        }
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::exp(x[i]);
        }
    }
}

void VectorMath::log(const double* x, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = simd_log(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = std::log(x[i]);
        }
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::log(x[i]);
        }
    }
}

void VectorMath::sqrt(const double* x, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
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
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::sqrt(x[i]);
        }
    }
}

void VectorMath::erf(const double* x, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = simd_erf(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = std::erf(x[i]);
        }
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::erf(x[i]);
        }
    }
}

void VectorMath::normalCDF(const double* x, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = simd_normalCDF(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
        }
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
        }
    }
}

void VectorMath::normalPDF(const double* x, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
        size_t i = 0;
        for (; i + 3 < size; i += 4) {
            __m256d vec = _mm256_loadu_pd(x + i);
            __m256d res = simd_normalPDF(vec);
            _mm256_storeu_pd(result + i, res);
        }
        
        // Handle remaining elements
        for (; i < size; ++i) {
            result[i] = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x[i] * x[i]);
        }
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = (1.0 / std::sqrt(2.0 * M_PI)) * std::exp(-0.5 * x[i] * x[i]);
        }
    }
}

void VectorMath::multiply(const double* a, const double* b, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
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
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }
}

void VectorMath::add(const double* a, const double* b, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
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
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }
}

void VectorMath::subtract(const double* a, const double* b, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
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
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] - b[i];
        }
    }
}

void VectorMath::divide(const double* a, const double* b, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
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
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] / b[i];
        }
    }
}

void VectorMath::bsD1(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
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
            __m256d log_S_div_K = simd_log(S_div_K);
            
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
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            double vol_sqrt_T = vol[i] * std::sqrt(T[i]);
            double half_vol_squared = 0.5 * vol[i] * vol[i];
            double log_S_div_K = std::log(S[i] / K[i]);
            double drift = r[i] - q[i] + half_vol_squared;
            double drift_T = drift * T[i];
            double numerator = log_S_div_K + drift_T;
            result[i] = numerator / vol_sqrt_T;
        }
    }
}

void VectorMath::bsD2(const double* d1, const double* vol, const double* T, 
                      double* result, size_t size) {
    // Process in chunks of 4 using AVX2 if available
    if (SimdDetect::hasAVX2()) {
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
    } else {
        // Use scalar computation
        for (size_t i = 0; i < size; ++i) {
            result[i] = d1[i] - vol[i] * std::sqrt(T[i]);
        }
    }
}

// SIMD helper functions
__m256d VectorMath::simd_exp(__m256d x) {
    // AVX2 doesn't have a direct exp function, use scalar computation
    std::array<double, 4> values;
    _mm256_storeu_pd(values.data(), x);
    
    for (int i = 0; i < 4; ++i) {
        values[i] = std::exp(values[i]);
    }
    
    return _mm256_loadu_pd(values.data());
}

__m256d VectorMath::simd_log(__m256d x) {
    // AVX2 doesn't have a direct log function, use scalar computation
    std::array<double, 4> values;
    _mm256_storeu_pd(values.data(), x);
    
    for (int i = 0; i < 4; ++i) {
        values[i] = std::log(values[i]);
    }
    
    return _mm256_loadu_pd(values.data());
}

__m256d VectorMath::simd_erf(__m256d x) {
    // AVX2 doesn't have a direct erf function, use scalar computation
    std::array<double, 4> values;
    _mm256_storeu_pd(values.data(), x);
    
    for (int i = 0; i < 4; ++i) {
        values[i] = std::erf(values[i]);
    }
    
    return _mm256_loadu_pd(values.data());
}

__m256d VectorMath::simd_normalCDF(__m256d x) {
    __m256d half = _mm256_set1_pd(0.5);
    __m256d one = _mm256_set1_pd(1.0);
    __m256d sqrt2inv = _mm256_set1_pd(1.0 / std::sqrt(2.0));
    
    // x / sqrt(2)
    __m256d arg = _mm256_mul_pd(x, sqrt2inv);
    
    // erf(x / sqrt(2))
    __m256d erf_val = simd_erf(arg);
    
    // 0.5 * (1 + erf(x / sqrt(2)))
    __m256d term1 = _mm256_add_pd(one, erf_val);
    return _mm256_mul_pd(half, term1);
}

__m256d VectorMath::simd_normalPDF(__m256d x) {
    __m256d half = _mm256_set1_pd(-0.5);
    __m256d invSqrt2Pi = _mm256_set1_pd(1.0 / std::sqrt(2.0 * M_PI));
    
    // x^2
    __m256d x_squared = _mm256_mul_pd(x, x);
    
    // -0.5 * x^2
    __m256d exponent = _mm256_mul_pd(half, x_squared);
    
    // exp(-0.5 * x^2)
    __m256d exp_val = simd_exp(exponent);
    
    // (1/sqrt(2π)) * exp(-0.5 * x^2)
    return _mm256_mul_pd(invSqrt2Pi, exp_val);
}

} // namespace opt
} // namespace alo
} // namespace engine