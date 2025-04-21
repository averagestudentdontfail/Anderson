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

// Check size threshold for using SIMD loops (even if underlying func is std::)
inline bool shouldUseSimd(size_t size) {
    // Keep consistency, though the benefit might be reduced if std:: funcs aren't auto-vectorized well
    return size >= 4; 
}

/**
 * @brief Vectorized operations for arrays of data
 */
void VectorMath::exp(const double* x, double* result, size_t size) {
    // Keep using SLEEF u10 as it benchmarked faster
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::exp(x[i]);
        }
        return;
    }
    
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = Sleef_expd4_u10avx2(vec);
        _mm256_storeu_pd(result + i, res);
    }
    for (; i < size; ++i) { result[i] = std::exp(x[i]); } // Scalar fallback for remainder
}

void VectorMath::log(const double* x, double* result, size_t size) {
     // Keep using SLEEF u10 based on bsD1D2 profiling results being good.
     // The isolated benchmark might have been misleading due to context/compiler.
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::log(x[i]);
        }
        return;
    }
    
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = Sleef_logd4_u10avx2(vec);
        _mm256_storeu_pd(result + i, res);
    }
     for (; i < size; ++i) { result[i] = std::log(x[i]); } // Scalar fallback for remainder
}

void VectorMath::sqrt(const double* x, double* result, size_t size) {
    // Keep using AVX2 native sqrt - it's fast
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::sqrt(x[i]);
        }
        return;
    }
    
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec = _mm256_loadu_pd(x + i);
        __m256d res = _mm256_sqrt_pd(vec);
        _mm256_storeu_pd(result + i, res);
    }
     for (; i < size; ++i) { result[i] = std::sqrt(x[i]); } // Scalar fallback for remainder
}

void VectorMath::erf(const double* x, double* result, size_t size) {
    // *** CHANGE: Revert to std::erf based on benchmarks ***
    // Process element by element
    for (size_t i = 0; i < size; ++i) {
        result[i] = std::erf(x[i]);
    }
    // Note: We could try to vectorize the std::erf loop using #pragma omp simd
    // or explicit AVX calls if std::erf itself isn't auto-vectorized well by the compiler,
    // but let's start with the simple loop.
}

// ========================================================================
// !!! MODIFIED normalCDF IMPLEMENTATION !!!
// ========================================================================
void VectorMath::normalCDF(const double* x, double* result, size_t size) {
    // *** CHANGE: Use std::erf / std::erfc instead of SLEEF ***
    const double inv_sqrt2 = 1.0 / M_SQRT2; // Precompute 1/sqrt(2)

    // Process element by element using std functions
    // The branching logic for numerical stability is still valid.
    for (size_t i = 0; i < size; ++i) {
        double scaled_x = x[i] * inv_sqrt2;
        if (x[i] < -8.0) { // Use erfc for large negative x
            result[i] = 0.5 * std::erfc(-scaled_x); // erfc(-x/sqrt(2)) = erfc(negative_scaled_x)
        } else if (x[i] > 8.0) { // Use erfc for large positive x
             result[i] = 1.0 - 0.5 * std::erfc(scaled_x); // 1 - 0.5 * erfc(x/sqrt(2))
        } else { // Use erf for moderate values
             result[i] = 0.5 * (1.0 + std::erf(scaled_x)); // 0.5 * (1 + erf(x/sqrt(2)))
        }
    }

    /* // Original SLEEF implementation - commented out
    if (!shouldUseSimd(size)) {
         for (size_t i = 0; i < size; ++i) {
             if (x[i] < -8.0) { result[i] = 0.5 * std::erfc(-x[i] / std::sqrt(2.0)); } 
             else if (x[i] > 8.0) { result[i] = 1.0 - 0.5 * std::erfc(x[i] / std::sqrt(2.0)); } 
             else { result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0))); }
         }
         return;
     }
    
     size_t i = 0;
     const __m256d vec_inv_sqrt2 = _mm256_set1_pd(inv_sqrt2);
     const __m256d vec_one = _mm256_set1_pd(1.0);
     const __m256d vec_half = _mm256_set1_pd(0.5);
     const __m256d vec_zero = _mm256_setzero_pd();
     const __m256d vec_neg_zero = _mm256_set1_pd(-0.0); // To get absolute value
     const __m256d vec_8 = _mm256_set1_pd(8.0);

     for (; i + 3 < size; i += 4) {
         __m256d x_vec = _mm256_loadu_pd(x + i);
         __m256d scaled_x = _mm256_mul_pd(x_vec, vec_inv_sqrt2); // Use multiply instead of divide

         __m256d abs_x = _mm256_andnot_pd(vec_neg_zero, x_vec);
         __m256d neg_mask = _mm256_cmp_pd(x_vec, vec_zero, _CMP_LT_OQ);
         __m256d large_mask = _mm256_cmp_pd(abs_x, vec_8, _CMP_GT_OQ);
         
         // erf path
         __m256d erf_scaled = Sleef_erfd4_u10avx2(scaled_x);
         __m256d normal_result = _mm256_mul_pd(_mm256_add_pd(vec_one, erf_scaled), vec_half);
         
         // erfc path
         __m256d neg_scaled_x = _mm256_sub_pd(vec_zero, scaled_x);
         __m256d large_neg_result = _mm256_mul_pd(vec_half, Sleef_erfcd4_u15avx2(neg_scaled_x));
         __m256d large_pos_result = _mm256_sub_pd(vec_one, _mm256_mul_pd(vec_half, Sleef_erfcd4_u15avx2(scaled_x)));
         __m256d erfc_result = _mm256_blendv_pd(large_pos_result, large_neg_result, neg_mask);
         
         __m256d result_vec = _mm256_blendv_pd(normal_result, erfc_result, large_mask);
         _mm256_storeu_pd(result + i, result_vec);
     }
     // Scalar fallback for remainder (original logic using std funcs)
     for (; i < size; ++i) {
         if (x[i] < -8.0) { result[i] = 0.5 * std::erfc(-x[i] / std::sqrt(2.0)); }
         else if (x[i] > 8.0) { result[i] = 1.0 - 0.5 * std::erfc(x[i] / std::sqrt(2.0)); }
         else { result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0))); }
     }
     */ // End of original SLEEF implementation
}
// ========================================================================


void VectorMath::normalPDF(const double* x, double* result, size_t size) {
    // Keep using SLEEF exp as it's faster
    const double INV_SQRT_2PI = 0.3989422804014327; 
    
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = INV_SQRT_2PI * std::exp(-0.5 * x[i] * x[i]);
        }
        return;
    }
    
    size_t i = 0;
    const __m256d vec_inv_sqrt_2pi = _mm256_set1_pd(INV_SQRT_2PI);
    const __m256d vec_neg_half = _mm256_set1_pd(-0.5);
    for (; i + 3 < size; i += 4) {
        __m256d xvec = _mm256_loadu_pd(x + i);
        __m256d x_squared = _mm256_mul_pd(xvec, xvec);
        __m256d scaled = _mm256_mul_pd(x_squared, vec_neg_half);
        __m256d exp_term = Sleef_expd4_u10avx2(scaled);
        __m256d pdf = _mm256_mul_pd(exp_term, vec_inv_sqrt_2pi);
        _mm256_storeu_pd(result + i, pdf);
    }
     for (; i < size; ++i) { result[i] = INV_SQRT_2PI * std::exp(-0.5 * x[i] * x[i]); } // Scalar fallback
}

void VectorMath::multiply(const double* a, const double* b, double* result, size_t size) {
    // Keep AVX2 implementation
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) { result[i] = a[i] * b[i]; }
        return;
    }
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(a + i);
        __m256d vec_b = _mm256_loadu_pd(b + i);
        __m256d res = _mm256_mul_pd(vec_a, vec_b);
        _mm256_storeu_pd(result + i, res);
    }
     for (; i < size; ++i) { result[i] = a[i] * b[i]; } // Scalar fallback
}

void VectorMath::add(const double* a, const double* b, double* result, size_t size) {
    // Keep AVX2 implementation
     if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) { result[i] = a[i] + b[i]; }
        return;
    }
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(a + i);
        __m256d vec_b = _mm256_loadu_pd(b + i);
        __m256d res = _mm256_add_pd(vec_a, vec_b);
        _mm256_storeu_pd(result + i, res);
    }
    for (; i < size; ++i) { result[i] = a[i] + b[i]; } // Scalar fallback
}

void VectorMath::subtract(const double* a, const double* b, double* result, size_t size) {
    // Keep AVX2 implementation
     if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) { result[i] = a[i] - b[i]; }
        return;
    }
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(a + i);
        __m256d vec_b = _mm256_loadu_pd(b + i);
        __m256d res = _mm256_sub_pd(vec_a, vec_b);
        _mm256_storeu_pd(result + i, res);
    }
     for (; i < size; ++i) { result[i] = a[i] - b[i]; } // Scalar fallback
}

void VectorMath::divide(const double* a, const double* b, double* result, size_t size) {
     // Keep AVX2 implementation
     if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) { result[i] = a[i] / b[i]; }
        return;
    }
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d vec_a = _mm256_loadu_pd(a + i);
        __m256d vec_b = _mm256_loadu_pd(b + i);
        __m256d res = _mm256_div_pd(vec_a, vec_b);
        _mm256_storeu_pd(result + i, res);
    }
     for (; i < size; ++i) { result[i] = a[i] / b[i]; } // Scalar fallback
}

void VectorMath::bsD1(const double* S, const double* K, const double* r, const double* q, 
                    const double* vol, const double* T, double* result, size_t size) {
    // Keep using SLEEF log as bsD1D2 profile was good
     if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            double vsqrtT = vol[i] * std::sqrt(T[i]);
             if (vsqrtT < 1e-10) { result[i] = (S[i]>K[i]) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity(); continue; } // Avoid division by zero
            double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) / vsqrtT;
            result[i] = d1;
        }
        return;
    }
    size_t i = 0;
     for (; i + 3 < size; i += 4) {
         __m256d S_vec = _mm256_loadu_pd(S + i); __m256d K_vec = _mm256_loadu_pd(K + i);
         __m256d r_vec = _mm256_loadu_pd(r + i); __m256d q_vec = _mm256_loadu_pd(q + i);
         __m256d vol_vec = _mm256_loadu_pd(vol + i); __m256d T_vec = _mm256_loadu_pd(T + i);
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
         _mm256_storeu_pd(result + i, d1);
     }
      for (; i < size; ++i) { // Scalar fallback
         double vsqrtT = vol[i] * std::sqrt(T[i]);
         if (vsqrtT < 1e-10) { result[i] = (S[i]>K[i]) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity(); continue; }
         result[i] = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) / vsqrtT;
     }
}

void VectorMath::bsD2(const double* d1, const double* vol, const double* T, 
                    double* result, size_t size) {
     // Keep AVX2 implementation
     if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) { result[i] = d1[i] - vol[i] * std::sqrt(T[i]); }
        return;
    }
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d d1_vec = _mm256_loadu_pd(d1 + i);
        __m256d vol_vec = _mm256_loadu_pd(vol + i);
        __m256d T_vec = _mm256_loadu_pd(T + i);
        __m256d sqrt_T = _mm256_sqrt_pd(T_vec);
        __m256d vol_sqrt_T = _mm256_mul_pd(vol_vec, sqrt_T);
        __m256d d2 = _mm256_sub_pd(d1_vec, vol_sqrt_T);
        _mm256_storeu_pd(result + i, d2);
    }
     for (; i < size; ++i) { result[i] = d1[i] - vol[i] * std::sqrt(T[i]); } // Scalar fallback
}

void VectorMath::bsPut(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* result, size_t size) {
    // This function now relies on the modified normalCDF (using std::erf)
    std::vector<double> d1(size);
    std::vector<double> d2(size);
    std::vector<double> neg_d1(size);
    std::vector<double> neg_d2(size);
    std::vector<double> Nd1(size);
    std::vector<double> Nd2(size);
    std::vector<double> discount_r(size);
    std::vector<double> discount_q(size);
    std::vector<double> neg_rT(size), neg_qT(size); // Temps for exp inputs

    bsD1D2(S, K, r, q, vol, T, d1.data(), d2.data(), size);

    for (size_t i = 0; i < size; i++) {
        neg_d1[i] = -d1[i];
        neg_d2[i] = -d2[i];
        neg_rT[i] = -r[i] * T[i];
        neg_qT[i] = -q[i] * T[i];
    }

    normalCDF(neg_d1.data(), Nd1.data(), size); // Uses std::erf internally now
    normalCDF(neg_d2.data(), Nd2.data(), size); // Uses std::erf internally now
    
    exp(neg_rT.data(), discount_r.data(), size); // Uses SLEEF exp
    exp(neg_qT.data(), discount_q.data(), size); // Uses SLEEF exp

    // Combine - use vectorized multiply/subtract for potentially better performance
    std::vector<double> term1(size);
    std::vector<double> term2(size);
    multiply(K, discount_r.data(), term1.data(), size);          
    multiply(term1.data(), Nd2.data(), term1.data(), size); 
    multiply(S, discount_q.data(), term2.data(), size);          
    multiply(term2.data(), Nd1.data(), term2.data(), size); 
    subtract(term1.data(), term2.data(), result, size); 

    // Handle degenerate cases after calculation
    for(size_t i=0; i<size; ++i) {
         if (vol[i] <= 0.0 || T[i] <= 0.0) {
            result[i] = std::max(0.0, K[i] - S[i]);
         }
     }
}

void VectorMath::bsCall(const double* S, const double* K, const double* r,
                       const double* q, const double* vol, const double* T,
                       double* result, size_t size) {
    // This function now relies on the modified normalCDF (using std::erf)
    std::vector<double> d1(size);
    std::vector<double> d2(size);
    std::vector<double> Nd1(size);
    std::vector<double> Nd2(size);
    std::vector<double> discount_r(size);
    std::vector<double> discount_q(size);
    std::vector<double> neg_rT(size), neg_qT(size); // Temps for exp inputs
    
    bsD1D2(S, K, r, q, vol, T, d1.data(), d2.data(), size);
    
    normalCDF(d1.data(), Nd1.data(), size); // Uses std::erf internally now
    normalCDF(d2.data(), Nd2.data(), size); // Uses std::erf internally now
    
     for (size_t i = 0; i < size; i++) {
        neg_rT[i] = -r[i] * T[i];
        neg_qT[i] = -q[i] * T[i];
    }
    exp(neg_rT.data(), discount_r.data(), size); // Uses SLEEF exp
    exp(neg_qT.data(), discount_q.data(), size); // Uses SLEEF exp
    
    // Combine - use vectorized multiply/subtract
    std::vector<double> term1(size);
    std::vector<double> term2(size);
    multiply(S, discount_q.data(), term1.data(), size);          
    multiply(term1.data(), Nd1.data(), term1.data(), size); 
    multiply(K, discount_r.data(), term2.data(), size);          
    multiply(term2.data(), Nd2.data(), term2.data(), size); 
    subtract(term1.data(), term2.data(), result, size);

     // Handle degenerate cases after calculation
     for(size_t i=0; i<size; ++i) {
         if (vol[i] <= 0.0 || T[i] <= 0.0) {
            result[i] = std::max(0.0, S[i] - K[i]);
         }
     }
}

void VectorMath::expMultSqrt(const double* x, const double* y, double* result, size_t size) {
    // Keep fused implementation, using SLEEF for exp and AVX2 for sqrt
    if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            result[i] = std::exp(x[i]) * std::sqrt(y[i]);
        }
        return;
    }
    
    size_t i = 0;
    for (; i + 3 < size; i += 4) {
        __m256d x_vec = _mm256_loadu_pd(x + i);
        __m256d y_vec = _mm256_loadu_pd(y + i);
        __m256d exp_x = Sleef_expd4_u10avx2(x_vec); // Keep SLEEF exp
        __m256d sqrt_y = _mm256_sqrt_pd(y_vec); // Keep AVX2 sqrt
        __m256d result_vec = _mm256_mul_pd(exp_x, sqrt_y);
        _mm256_storeu_pd(result + i, result_vec);
    }
     for (; i < size; ++i) { result[i] = std::exp(x[i]) * std::sqrt(y[i]); } // Scalar fallback
}

void VectorMath::bsD1D2(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* d1, double* d2, size_t size) {
    // Keep using SLEEF log as profile was good
     if (!shouldUseSimd(size)) {
        for (size_t i = 0; i < size; ++i) {
            if (vol[i] <= 0.0 || T[i] <= 0.0) { d1[i] = 0.0; d2[i] = 0.0; continue; }
            double vsqrtT = vol[i] * std::sqrt(T[i]);
             if (vsqrtT < 1e-10) { d1[i] = (S[i]>K[i]) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity(); d2[i]=d1[i]; continue;}
            double d1_val = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) / vsqrtT;
            d1[i] = d1_val; d2[i] = d1_val - vsqrtT;
        }
        return;
    }
    size_t i = 0;
     for (; i + 3 < size; i += 4) {
         __m256d S_vec = _mm256_loadu_pd(S + i); __m256d K_vec = _mm256_loadu_pd(K + i);
         __m256d r_vec = _mm256_loadu_pd(r + i); __m256d q_vec = _mm256_loadu_pd(q + i);
         __m256d vol_vec = _mm256_loadu_pd(vol + i); __m256d T_vec = _mm256_loadu_pd(T + i);
         
         __m256d eps = _mm256_set1_pd(1e-10);
         __m256d vol_mask = _mm256_cmp_pd(vol_vec, eps, _CMP_LE_OQ);
         __m256d T_mask = _mm256_cmp_pd(T_vec, eps, _CMP_LE_OQ);
         __m256d degenerate_mask = _mm256_or_pd(vol_mask, T_mask);
         
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
         __m256d d1_vec = _mm256_div_pd(numerator, vol_sqrt_T);
         __m256d d2_vec = _mm256_sub_pd(d1_vec, vol_sqrt_T);
         
         __m256d zero = _mm256_setzero_pd();
         d1_vec = _mm256_blendv_pd(d1_vec, zero, degenerate_mask);
         d2_vec = _mm256_blendv_pd(d2_vec, zero, degenerate_mask);
         
         _mm256_storeu_pd(d1 + i, d1_vec);
         _mm256_storeu_pd(d2 + i, d2_vec);
     }
      for (; i < size; ++i) { // Scalar fallback
         if (vol[i] <= 0.0 || T[i] <= 0.0) { d1[i] = 0.0; d2[i] = 0.0; continue; }
         double vsqrtT = vol[i] * std::sqrt(T[i]);
         if (vsqrtT < 1e-10) { d1[i] = (S[i]>K[i]) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity(); d2[i]=d1[i]; continue;}
         double d1_val = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) / vsqrtT;
         d1[i] = d1_val; d2[i] = d1_val - vsqrtT;
     }
}

void VectorMath::discountedNormal(const double* x, const double* r, const double* T, 
                                 double* result, size_t size) {
    // Relies on modified normalCDF and SLEEF exp
     std::vector<double> n_x(size);
     std::vector<double> neg_rT(size);
     std::vector<double> discount(size);

     normalCDF(x, n_x.data(), size); // Uses std::erf internally now

     for (size_t i = 0; i < size; i++) {
         neg_rT[i] = -r[i] * T[i];
     }
     exp(neg_rT.data(), discount.data(), size); // Uses SLEEF exp

     multiply(discount.data(), n_x.data(), result, size); // Uses AVX2 multiply
}

} // namespace opt
} // namespace alo
} // namespace engine