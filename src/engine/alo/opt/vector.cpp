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
        __m256d res = Sleef_expd4_u10(vec);
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
        __m256d res = Sleef_logd4_u10(vec);
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
        __m256d res = Sleef_tanhd4_u10(scaled);
        
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = std::erf(x[i]);
    }
}

void VectorMath::normalCDF(const double* x, double* result, size_t size) {
    // Normal CDF: 0.5 * (1 + erf(x / sqrt(2)))
    const double ONE_OVER_SQRT_2 = 1.0 / std::sqrt(2.0);
    
    // Use scalar operations for small data sizes
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = 0.5 * (1.0 + std::erf(x[i] * ONE_OVER_SQRT_2));
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF-based implementation
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = SimdOps::normalCDF(vec);
        _mm256_storeu_pd(result + i, res);
    }
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = 0.5 * (1.0 + std::erf(x[i] * ONE_OVER_SQRT_2));
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
        __m256d exp_term = Sleef_expd4_u10(scaled);
        
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
        __m256d log_S_div_K = Sleef_logd4_u10(S_div_K);
        
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
            
            double nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
            double nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
            
            result[i] = K[i] * std::exp(-r[i] * T[i]) * nd2 - S[i] * std::exp(-q[i] * T[i]) * nd1;
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF-powered implementation
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Use SLEEF-powered Black-Scholes implementation
        __m256d res = SimdOps::bsPut(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);
        
        // Store result
        _mm256_storeu_pd(result + i, res);
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
        
        double nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
        double nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
        
        result[i] = K[i] * std::exp(-r[i] * T[i]) * nd2 - S[i] * std::exp(-q[i] * T[i]) * nd1;
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
            
            double nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
            double nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
            
            result[i] = S[i] * std::exp(-q[i] * T[i]) * nd1 - K[i] * std::exp(-r[i] * T[i]) * nd2;
        }
        return;
    }
    
    // Process in chunks of 4 using SLEEF-powered implementation
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d S_vec = _mm256_loadu_pd(S + i);
        __m256d K_vec = _mm256_loadu_pd(K + i);
        __m256d r_vec = _mm256_loadu_pd(r + i);
        __m256d q_vec = _mm256_loadu_pd(q + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        
        // Use SLEEF-powered Black-Scholes implementation
        __m256d res = SimdOps::bsCall(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);
        
        // Store result
        _mm256_storeu_pd(result + i, res);
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
        
        double nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
        double nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
        
        result[i] = S[i] * std::exp(-q[i] * T[i]) * nd1 - K[i] * std::exp(-r[i] * T[i]) * nd2;
    }
}

} // namespace opt
} // namespace alo
} // namespace engine