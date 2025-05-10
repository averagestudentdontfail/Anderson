#ifndef ENGINE_ALO_OPT_SIMD_H
#define ENGINE_ALO_OPT_SIMD_H

#include "../num/float.h" 
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
 * @class SimdOperationDouble
 * @brief Optimized SIMD operations for financial math with double precision
 *
 * This class provides SIMD operations optimized for financial mathematics
 * calculations with AVX2 instructions (256-bit vectors) for double precision.
 */
class SimdOperationDouble {
public:
  /**
   * @brief Load 4 doubles into a vector
   *
   * @param ptr Pointer to array of 4 doubles
   * @return 256-bit vector containing 4 doubles
   */
  static inline __m256d load(const double *ptr) { return _mm256_loadu_pd(ptr); }

  /**
   * @brief Load 4 doubles from an array
   *
   * @param arr Array of 4 doubles
   * @return 256-bit vector containing 4 doubles
   */
  static inline __m256d load(const ::std::array<double, 4> &arr) {
    return _mm256_loadu_pd(arr.data());
  }

  /**
   * @brief Store a vector into 4 doubles
   *
   * @param ptr Pointer to array of 4 doubles
   * @param vec 256-bit vector containing 4 doubles
   */
  static inline void store(double *ptr, __m256d vec) {
    _mm256_storeu_pd(ptr, vec);
  }

  /**
   * @brief Store a vector into an array
   *
   * @param arr Array of 4 doubles
   * @param vec 256-bit vector containing 4 doubles
   */
  static inline void store(::std::array<double, 4> &arr, __m256d vec) {
    _mm256_storeu_pd(arr.data(), vec);
  }

  /**
   * @brief Set all elements to a scalar value
   *
   * @param value Scalar value
   * @return 256-bit vector with all elements set to value
   */
  static inline __m256d set1(double value) { return _mm256_set1_pd(value); }

  /**
   * @brief Set 4 elements
   *
   * @param d Fourth element (order is d, c, b, a for _mm256_set_pd)
   * @param c Third element
   * @param b Second element
   * @param a First element
   * @return 256-bit vector containing [a, b, c, d] (in memory order)
   */
  static inline __m256d set(double a, double b, double c, double d) {
    return _mm256_set_pd(d, c, b, a); 
  }

  /**
   * @brief Calculate European option d1 for 4 options at once
   */
  static inline __m256d EuropeanD1(__m256d S, __m256d K, __m256d r, __m256d q,
                                   __m256d vol, __m256d T) {
    __m256d sqrt_T = _mm256_sqrt_pd(T);
    __m256d vol_sqrt_T = _mm256_mul_pd(vol, sqrt_T);
    __m256d S_div_K = _mm256_div_pd(S, K);
    __m256d log_S_div_K = Sleef_logd4_u10avx2(S_div_K); // SLEEF for log

    __m256d vol_squared = _mm256_mul_pd(vol, vol);
    __m256d half_vol_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), vol_squared);
    __m256d r_minus_q = _mm256_sub_pd(r, q);
    __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
    __m256d drift_T = _mm256_mul_pd(drift, T);

    __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
    // Add a small epsilon to denominator to prevent division by zero if vol_sqrt_T is zero
    __m256d epsilon = _mm256_set1_pd(1e-16); 
    return _mm256_div_pd(numerator, _mm256_add_pd(vol_sqrt_T, epsilon));
  }

  /**
   * @brief Calculate European option d2 from d1
   */
  static inline __m256d EuropeanD2(__m256d d1, __m256d vol, __m256d T) {
    __m256d sqrt_T = _mm256_sqrt_pd(T);
    __m256d vol_sqrt_T = _mm256_mul_pd(vol, sqrt_T);
    return _mm256_sub_pd(d1, vol_sqrt_T);
  }
  
  /**
   * @brief Normal CDF calculation with improved precision using SLEEF
   */
  static inline __m256d normalCDF(__m256d x) {
    const __m256d HALF = _mm256_set1_pd(0.5);
    const __m256d ONE = _mm256_set1_pd(1.0);
    const __m256d SQRT2_INV_PD = _mm256_set1_pd(0.70710678118654752440); // 1/sqrt(2)

    // Using SLEEF's erf directly. SLEEF's erf is generally robust.
    __m256d scaled_x = _mm256_mul_pd(x, SQRT2_INV_PD);
    __m256d erf_result = Sleef_erfd4_u10avx2(scaled_x); 
    return _mm256_mul_pd(HALF, _mm256_add_pd(ONE, erf_result));
  }

  /**
   * @brief Normal PDF calculation using SLEEF
   */
  static inline __m256d normalPDF(__m256d x) {
    const __m256d NEG_HALF = _mm256_set1_pd(-0.5);
    const __m256d INV_SQRT_2PI_PD = _mm256_set1_pd(0.39894228040143267794); // 1/sqrt(2*PI)
    __m256d x_squared = _mm256_mul_pd(x, x);
    __m256d exponent = _mm256_mul_pd(NEG_HALF, x_squared);
    __m256d exp_term = Sleef_expd4_u10avx2(exponent); // SLEEF for exp
    return _mm256_mul_pd(exp_term, INV_SQRT_2PI_PD);
  }

  /**
   * @brief Calculate European put prices for 4 options at once
   */
  static inline __m256d EuropeanPut(__m256d S, __m256d K, __m256d r, __m256d q,
                                    __m256d vol, __m256d T) {
    __m256d zero = _mm256_setzero_pd();
    __m256d eps = _mm256_set1_pd(1e-12); // Epsilon for double precision checks

    __m256d vol_mask = _mm256_cmp_pd(vol, eps, _CMP_LE_OQ);
    __m256d t_mask = _mm256_cmp_pd(T, eps, _CMP_LE_OQ);
    __m256d degenerate_mask = _mm256_or_pd(vol_mask, t_mask);

    __m256d K_minus_S = _mm256_sub_pd(K, S);
    __m256d degenerate_value = _mm256_max_pd(zero, K_minus_S);

    __m256d d1_val = EuropeanD1(S, K, r, q, vol, T);
    __m256d d2_val = EuropeanD2(d1_val, vol, T);

    __m256d neg_d1 = _mm256_sub_pd(zero, d1_val);
    __m256d neg_d2 = _mm256_sub_pd(zero, d2_val);

    __m256d Nd1 = normalCDF(neg_d1);
    __m256d Nd2 = normalCDF(neg_d2);

    __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
    __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q), T);
    __m256d dr_val = Sleef_expd4_u10avx2(neg_r_T); // SLEEF for exp
    __m256d dq_val = Sleef_expd4_u10avx2(neg_q_T); // SLEEF for exp

    __m256d term1 = _mm256_mul_pd(K, dr_val);
    term1 = _mm256_mul_pd(term1, Nd2);
    __m256d term2 = _mm256_mul_pd(S, dq_val);
    term2 = _mm256_mul_pd(term2, Nd1);
    __m256d put_value = _mm256_sub_pd(term1, term2);

    return _mm256_blendv_pd(put_value, degenerate_value, degenerate_mask);
  }

  /**
   * @brief Calculate European call prices for 4 options at once
   */
  static inline __m256d EuropeanCall(__m256d S, __m256d K, __m256d r, __m256d q,
                                     __m256d vol, __m256d T) {
    __m256d zero = _mm256_setzero_pd();
    __m256d eps = _mm256_set1_pd(1e-12);

    __m256d vol_mask = _mm256_cmp_pd(vol, eps, _CMP_LE_OQ);
    __m256d t_mask = _mm256_cmp_pd(T, eps, _CMP_LE_OQ);
    __m256d degenerate_mask = _mm256_or_pd(vol_mask, t_mask);

    __m256d S_minus_K = _mm256_sub_pd(S, K);
    __m256d degenerate_value = _mm256_max_pd(zero, S_minus_K);

    __m256d d1_val = EuropeanD1(S, K, r, q, vol, T);
    __m256d d2_val = EuropeanD2(d1_val, vol, T);

    __m256d Nd1 = normalCDF(d1_val);
    __m256d Nd2 = normalCDF(d2_val);

    __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
    __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q), T);
    __m256d dr_val = Sleef_expd4_u10avx2(neg_r_T); // SLEEF for exp
    __m256d dq_val = Sleef_expd4_u10avx2(neg_q_T); // SLEEF for exp

    __m256d term1 = _mm256_mul_pd(S, dq_val);
    term1 = _mm256_mul_pd(term1, Nd1);
    __m256d term2 = _mm256_mul_pd(K, dr_val);
    term2 = _mm256_mul_pd(term2, Nd2);
    __m256d call_value = _mm256_sub_pd(term1, term2);

    return _mm256_blendv_pd(call_value, degenerate_value, degenerate_mask);
  }

  // Greeks for double precision (AVX2 - 4 doubles)
  static inline __m256d delta(__m256d S, __m256d K, __m256d r, __m256d q,
                              __m256d vol, __m256d T, bool isCall);
  static inline __m256d gamma(__m256d S, __m256d K, __m256d r, __m256d q,
                              __m256d vol, __m256d T);
  static inline __m256d vega(__m256d S, __m256d K, __m256d r, __m256d q,
                             __m256d vol, __m256d T);
  static inline __m256d theta(__m256d S, __m256d K, __m256d r, __m256d q,
                              __m256d vol, __m256d T, bool isCall);
  static inline __m256d rho(__m256d S, __m256d K, __m256d r, __m256d q,
                            __m256d vol, __m256d T, bool isCall);
};


/**
 * @class SimdOperationSingle
 * @brief Optimized SIMD operations for financial math with single precision
 *
 * This class provides SIMD operations optimized for financial mathematics
 * calculations with AVX2 instructions (256-bit vectors, 8 floats)
 */
class SimdOperationSingle {
public:
  /**
   * @brief Load 8 floats into a vector
   */
  static inline __m256 load(const float *ptr) { return _mm256_loadu_ps(ptr); }
  
  /**
   * @brief Load 8 floats from an array
   */
  static inline __m256 load(const ::std::array<float, 8> &arr) {
    return _mm256_loadu_ps(arr.data());
  }

  /**
   * @brief Store a vector into 8 floats
   */
  static inline void store(float *ptr, __m256 vec) {
    _mm256_storeu_ps(ptr, vec);
  }

  /**
   * @brief Store a vector into an array
   */
  static inline void store(::std::array<float, 8> &arr, __m256 vec) {
    _mm256_storeu_ps(arr.data(), vec);
  }


  /**
   * @brief Set all elements to a scalar value
   */
  static inline __m256 set1(float value) { return _mm256_set1_ps(value); }

  /**
   * @brief Calculate European option d1 for 8 options at once
   */
  static inline __m256 EuropeanD1(__m256 S, __m256 K, __m256 r, __m256 q,
                                  __m256 vol, __m256 T) {
    __m256 sqrt_T = num::simd::sqrt_ps(T); // num::simd::sqrt_ps
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);
    __m256 S_div_K = _mm256_div_ps(S, K);
    __m256 log_S_div_K = num::simd::log_ps(S_div_K); // num::simd::log_ps

    __m256 vol_squared = _mm256_mul_ps(vol, vol);
    __m256 half_vol_squared = _mm256_mul_ps(_mm256_set1_ps(0.5f), vol_squared);
    __m256 r_minus_q = _mm256_sub_ps(r, q);
    __m256 drift = _mm256_add_ps(r_minus_q, half_vol_squared);
    __m256 drift_T = _mm256_mul_ps(drift, T);

    __m256 numerator = _mm256_add_ps(log_S_div_K, drift_T);
    // Add a small epsilon to denominator to prevent division by zero
    __m256 epsilon = _mm256_set1_ps(1e-7f);
    return _mm256_div_ps(numerator, _mm256_add_ps(vol_sqrt_T, epsilon));
  }

  /**
   * @brief Calculate European option d2 from d1
   */
  static inline __m256 EuropeanD2(__m256 d1, __m256 vol, __m256 T) {
    __m256 sqrt_T = num::simd::sqrt_ps(T); // num::simd::sqrt_ps
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);
    return _mm256_sub_ps(d1, vol_sqrt_T);
  }

  /**
   * @brief Calculate European put prices for 8 options at once
   */
  static inline __m256 EuropeanPut(__m256 S, __m256 K, __m256 r, __m256 q,
                                   __m256 vol, __m256 T) {
    __m256 zero = _mm256_setzero_ps();
    __m256 eps = _mm256_set1_ps(1e-7f); 

    __m256 vol_mask = _mm256_cmp_ps(vol, eps, _CMP_LE_OQ);
    __m256 t_mask = _mm256_cmp_ps(T, eps, _CMP_LE_OQ);
    __m256 degenerate_mask = _mm256_or_ps(vol_mask, t_mask);

    __m256 K_minus_S = _mm256_sub_ps(K, S);
    __m256 degenerate_value = _mm256_max_ps(zero, K_minus_S);

    __m256 d1_val = EuropeanD1(S, K, r, q, vol, T);
    __m256 d2_val = EuropeanD2(d1_val, vol, T);

    __m256 neg_d1 = _mm256_sub_ps(zero, d1_val);
    __m256 neg_d2 = _mm256_sub_ps(zero, d2_val);

    __m256 Nd1 = num::simd::normal_cdf_ps(neg_d1); // num::simd::normal_cdf_ps
    __m256 Nd2 = num::simd::normal_cdf_ps(neg_d2); // num::simd::normal_cdf_ps

    __m256 neg_r_T = _mm256_mul_ps(_mm256_sub_ps(zero, r), T);
    __m256 neg_q_T = _mm256_mul_ps(_mm256_sub_ps(zero, q), T);
    __m256 dr_val = num::simd::exp_ps(neg_r_T); // num::simd::exp_ps
    __m256 dq_val = num::simd::exp_ps(neg_q_T); // num::simd::exp_ps

    __m256 term1 = _mm256_mul_ps(K, dr_val);
    term1 = _mm256_mul_ps(term1, Nd2);
    __m256 term2 = _mm256_mul_ps(S, dq_val);
    term2 = _mm256_mul_ps(term2, Nd1);
    __m256 put_value = _mm256_sub_ps(term1, term2);

    return _mm256_blendv_ps(put_value, degenerate_value, degenerate_mask);
  }

  /**
   * @brief Calculate European call prices for 8 options at once
   */
  static inline __m256 EuropeanCall(__m256 S, __m256 K, __m256 r, __m256 q,
                                    __m256 vol, __m256 T) {
    __m256 zero = _mm256_setzero_ps();
    __m256 eps = _mm256_set1_ps(1e-7f);

    __m256 vol_mask = _mm256_cmp_ps(vol, eps, _CMP_LE_OQ);
    __m256 t_mask = _mm256_cmp_ps(T, eps, _CMP_LE_OQ);
    __m256 degenerate_mask = _mm256_or_ps(vol_mask, t_mask);

    __m256 S_minus_K = _mm256_sub_ps(S, K);
    __m256 degenerate_value = _mm256_max_ps(zero, S_minus_K);

    __m256 d1_val = EuropeanD1(S, K, r, q, vol, T);
    __m256 d2_val = EuropeanD2(d1_val, vol, T);

    __m256 Nd1 = num::simd::normal_cdf_ps(d1_val); // num::simd::normal_cdf_ps
    __m256 Nd2 = num::simd::normal_cdf_ps(d2_val); // num::simd::normal_cdf_ps

    __m256 neg_r_T = _mm256_mul_ps(_mm256_sub_ps(zero, r), T);
    __m256 neg_q_T = _mm256_mul_ps(_mm256_sub_ps(zero, q), T);
    __m256 dr_val = num::simd::exp_ps(neg_r_T); // num::simd::exp_ps
    __m256 dq_val = num::simd::exp_ps(neg_q_T); // num::simd::exp_ps

    __m256 term1 = _mm256_mul_ps(S, dq_val);
    term1 = _mm256_mul_ps(term1, Nd1);
    __m256 term2 = _mm256_mul_ps(K, dr_val);
    term2 = _mm256_mul_ps(term2, Nd2);
    __m256 call_value = _mm256_sub_ps(term1, term2);

    return _mm256_blendv_ps(call_value, degenerate_value, degenerate_mask);
  }

  // Greeks for single precision (AVX2 - 8 floats)
  static inline __m256 delta(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                             __m256 T, bool isCall);
  static inline __m256 gamma(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                             __m256 T);
  static inline __m256 vega(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                            __m256 T);
  static inline __m256 theta(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                             __m256 T, bool isCall);
  static inline __m256 rho(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                           __m256 T, bool isCall);

  // American option approximations (single precision - AVX2 - 8 floats)
  static inline __m256 AmericanPut(__m256 S, __m256 K, __m256 r, __m256 q,
                                   __m256 vol, __m256 T);
  static inline __m256 AmericanCall(__m256 S, __m256 K, __m256 r, __m256 q,
                                    __m256 vol, __m256 T);
};

// Inline implementations for Greeks and American approximations (can be moved to .cpp if preferred)
// Double Precision Greeks
inline __m256d SimdOperationDouble::delta(__m256d S, __m256d K, __m256d r, __m256d q,
                                         __m256d vol, __m256d T, bool isCall) {
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d neg_q_T = _mm256_mul_pd(_mm256_mul_pd(q, T), _mm256_set1_pd(-1.0));
    __m256d dq = Sleef_expd4_u10avx2(neg_q_T);
    if (isCall) {
      __m256d Nd1 = normalCDF(d1);
      return _mm256_mul_pd(dq, Nd1);
    } else {
      __m256d neg_d1 = _mm256_sub_pd(_mm256_setzero_pd(), d1);
      __m256d Nneg_d1 = normalCDF(neg_d1);
      return _mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(dq, Nneg_d1));
    }
}

inline __m256d SimdOperationDouble::gamma(__m256d S, __m256d K, __m256d r, __m256d q,
                                         __m256d vol, __m256d T) {
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d pdf_d1 = normalPDF(d1);
    __m256d neg_q_T = _mm256_mul_pd(_mm256_mul_pd(q, T), _mm256_set1_pd(-1.0));
    __m256d dq = Sleef_expd4_u10avx2(neg_q_T);
    __m256d sqrt_T = _mm256_sqrt_pd(T);
    __m256d vol_sqrt_T = _mm256_mul_pd(vol, sqrt_T);
    __m256d denominator = _mm256_mul_pd(S, vol_sqrt_T);
    __m256d numerator = _mm256_mul_pd(dq, pdf_d1);
    // Add a small epsilon to denominator to prevent division by zero
    __m256d epsilon = _mm256_set1_pd(1e-16);
    return _mm256_div_pd(numerator, _mm256_add_pd(denominator, epsilon));
}

inline __m256d SimdOperationDouble::vega(__m256d S, __m256d K, __m256d r, __m256d q,
                                        __m256d vol, __m256d T) {
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d pdf_d1 = normalPDF(d1);
    __m256d neg_q_T = _mm256_mul_pd(_mm256_mul_pd(q, T), _mm256_set1_pd(-1.0));
    __m256d dq = Sleef_expd4_u10avx2(neg_q_T);
    __m256d sqrt_T = _mm256_sqrt_pd(T);
    __m256d result = _mm256_mul_pd(S, dq);
    result = _mm256_mul_pd(result, pdf_d1);
    result = _mm256_mul_pd(result, sqrt_T);
    return _mm256_mul_pd(result, _mm256_set1_pd(0.01));
}

inline __m256d SimdOperationDouble::theta(__m256d S, __m256d K, __m256d r, __m256d q,
                                         __m256d vol, __m256d T, bool isCall) {
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d d2 = EuropeanD2(d1, vol, T);
    __m256d pdf_d1 = normalPDF(d1);
    __m256d zero = _mm256_setzero_pd();
    __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
    __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q), T);
    __m256d dr_val = Sleef_expd4_u10avx2(neg_r_T);
    __m256d dq_val = Sleef_expd4_u10avx2(neg_q_T);
    __m256d sqrt_T = _mm256_sqrt_pd(T);

    __m256d term1_factor = _mm256_div_pd(vol, _mm256_mul_pd(_mm256_set1_pd(2.0), sqrt_T));
    __m256d S_dq_pdf = _mm256_mul_pd(_mm256_mul_pd(S, dq_val), pdf_d1);
    __m256d term1 = _mm256_mul_pd(_mm256_sub_pd(zero, S_dq_pdf), term1_factor);

    if (isCall) {
        __m256d Nd1 = normalCDF(d1);
        __m256d Nd2 = normalCDF(d2);
        __m256d term2 = _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(zero, q), S), _mm256_mul_pd(dq_val, Nd1));
        __m256d term3 = _mm256_mul_pd(_mm256_mul_pd(r, K), _mm256_mul_pd(dr_val, Nd2));
        __m256d theta_val = _mm256_add_pd(term1, _mm256_add_pd(term2, term3));
        return _mm256_div_pd(theta_val, _mm256_set1_pd(365.0));
    } else {
        __m256d neg_d1 = _mm256_sub_pd(zero, d1);
        __m256d neg_d2 = _mm256_sub_pd(zero, d2);
        __m256d Nneg_d1 = normalCDF(neg_d1);
        __m256d Nneg_d2 = normalCDF(neg_d2);
        __m256d term2 = _mm256_mul_pd(_mm256_mul_pd(q, S), _mm256_mul_pd(dq_val, Nneg_d1));
        __m256d term3 = _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(zero, r), K), _mm256_mul_pd(dr_val, Nneg_d2));
        __m256d theta_val = _mm256_add_pd(term1, _mm256_add_pd(term2, term3));
        return _mm256_div_pd(theta_val, _mm256_set1_pd(365.0));
    }
}

inline __m256d SimdOperationDouble::rho(__m256d S, __m256d K, __m256d r, __m256d q,
                                       __m256d vol, __m256d T, bool isCall) {
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d d2 = EuropeanD2(d1, vol, T);
    __m256d zero = _mm256_setzero_pd();
    __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
    __m256d dr_val = Sleef_expd4_u10avx2(neg_r_T);
    __m256d K_T = _mm256_mul_pd(K, T);
    __m256d base = _mm256_mul_pd(K_T, dr_val);
    __m256d scale = _mm256_set1_pd(0.01);
    if (isCall) {
        __m256d Nd2 = normalCDF(d2);
        return _mm256_mul_pd(_mm256_mul_pd(base, Nd2), scale);
    } else {
        __m256d neg_d2 = _mm256_sub_pd(zero, d2);
        __m256d Nneg_d2 = normalCDF(neg_d2);
        return _mm256_mul_pd(_mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), base), Nneg_d2), scale);
    }
}


// Single Precision Greeks
inline __m256 SimdOperationSingle::delta(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                                        __m256 T, bool isCall) {
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dq = num::simd::exp_ps(neg_q_T);
    if (isCall) {
      __m256 Nd1 = num::simd::normal_cdf_ps(d1);
      return _mm256_mul_ps(dq, Nd1);
    } else {
      __m256 neg_d1 = _mm256_sub_ps(_mm256_setzero_ps(), d1);
      __m256 Nneg_d1 = num::simd::normal_cdf_ps(neg_d1);
      return _mm256_mul_ps(_mm256_set1_ps(-1.0f), _mm256_mul_ps(dq, Nneg_d1));
    }
}

inline __m256 SimdOperationSingle::gamma(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                                        __m256 T) {
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256 pdf_d1 = num::simd::normal_pdf_ps(d1);
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dq = num::simd::exp_ps(neg_q_T);
    __m256 sqrt_T = num::simd::sqrt_ps(T);
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);
    __m256 denominator = _mm256_mul_ps(S, vol_sqrt_T);
    __m256 numerator = _mm256_mul_ps(dq, pdf_d1);
    // Add a small epsilon to denominator to prevent division by zero
    __m256 epsilon = _mm256_set1_ps(1e-7f);
    return _mm256_div_ps(numerator, _mm256_add_ps(denominator, epsilon));
}

inline __m256 SimdOperationSingle::vega(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                                       __m256 T) {
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256 pdf_d1 = num::simd::normal_pdf_ps(d1);
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dq = num::simd::exp_ps(neg_q_T);
    __m256 sqrt_T = num::simd::sqrt_ps(T);
    __m256 result = _mm256_mul_ps(S, dq);
    result = _mm256_mul_ps(result, pdf_d1);
    result = _mm256_mul_ps(result, sqrt_T);
    return _mm256_mul_ps(result, _mm256_set1_ps(0.01f));
}

inline __m256 SimdOperationSingle::theta(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                                        __m256 T, bool isCall) {
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256 d2 = EuropeanD2(d1, vol, T);
    __m256 pdf_d1 = num::simd::normal_pdf_ps(d1);
    __m256 zero = _mm256_setzero_ps();
    __m256 neg_r_T = _mm256_mul_ps(_mm256_sub_ps(zero, r), T);
    __m256 neg_q_T = _mm256_mul_ps(_mm256_sub_ps(zero, q), T);
    __m256 dr_val = num::simd::exp_ps(neg_r_T);
    __m256 dq_val = num::simd::exp_ps(neg_q_T);
    __m256 sqrt_T = num::simd::sqrt_ps(T);

    __m256 term1_factor = _mm256_div_ps(vol, _mm256_mul_ps(_mm256_set1_ps(2.0f), sqrt_T));
     __m256 S_dq_pdf = _mm256_mul_ps(_mm256_mul_ps(S, dq_val), pdf_d1);
    __m256 term1 = _mm256_mul_ps(_mm256_sub_ps(zero, S_dq_pdf), term1_factor);
    
    if (isCall) {
        __m256 Nd1 = num::simd::normal_cdf_ps(d1);
        __m256 Nd2 = num::simd::normal_cdf_ps(d2);
        __m256 term2 = _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(zero, q), S), _mm256_mul_ps(dq_val, Nd1));
        __m256 term3 = _mm256_mul_ps(_mm256_mul_ps(r, K), _mm256_mul_ps(dr_val, Nd2));
        __m256 theta_val = _mm256_add_ps(term1, _mm256_add_ps(term2, term3));
        return _mm256_div_ps(theta_val, _mm256_set1_ps(365.0f));
    } else {
        __m256 neg_d1 = _mm256_sub_ps(zero, d1);
        __m256 neg_d2 = _mm256_sub_ps(zero, d2);
        __m256 Nneg_d1 = num::simd::normal_cdf_ps(neg_d1);
        __m256 Nneg_d2 = num::simd::normal_cdf_ps(neg_d2);
        __m256 term2 = _mm256_mul_ps(_mm256_mul_ps(q, S), _mm256_mul_ps(dq_val, Nneg_d1));
        __m256 term3 = _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(zero, r), K), _mm256_mul_ps(dr_val, Nneg_d2));
        __m256 theta_val = _mm256_add_ps(term1, _mm256_add_ps(term2, term3));
        return _mm256_div_ps(theta_val, _mm256_set1_ps(365.0f));
    }
}

inline __m256 SimdOperationSingle::rho(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                                      __m256 T, bool isCall) {
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256 d2 = EuropeanD2(d1, vol, T);
    __m256 zero = _mm256_setzero_ps();
    __m256 neg_r_T = _mm256_mul_ps(_mm256_sub_ps(zero, r), T);
    __m256 dr_val = num::simd::exp_ps(neg_r_T);
    __m256 K_T = _mm256_mul_ps(K, T);
    __m256 base = _mm256_mul_ps(K_T, dr_val);
    __m256 scale = _mm256_set1_ps(0.01f);
    if (isCall) {
        __m256 Nd2 = num::simd::normal_cdf_ps(d2);
        return _mm256_mul_ps(_mm256_mul_ps(base, Nd2), scale);
    } else {
        __m256 neg_d2 = _mm256_sub_ps(zero, d2);
        __m256 Nneg_d2 = num::simd::normal_cdf_ps(neg_d2);
        return _mm256_mul_ps(_mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), base), Nneg_d2), scale);
    }
}


// American Option Approximations (Single Precision - AVX2 - 8 floats)
// Barone-Adesi-Whaley approximation
inline __m256 SimdOperationSingle::AmericanPut(__m256 S, __m256 K, __m256 r, __m256 q,
                                            __m256 vol, __m256 T) {
    __m256 euro_price = EuropeanPut(S, K, r, q, vol, T);
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 eps = _mm256_set1_ps(1e-7f); // Epsilon for float comparisons

    // Check if early exercise is valuable: r > q
    __m256 r_gt_q_mask = _mm256_cmp_ps(r, q, _CMP_GT_OQ);
    
    // For cases where r <= q, American = European
    __m256 american_price = euro_price; 

    // Calculate b = r - q
    __m256 b_val = _mm256_sub_ps(r, q);

    // M = 2r / vol^2
    __m256 vol_sq = _mm256_mul_ps(vol, vol);
    __m256 M_val = _mm256_div_ps(_mm256_mul_ps(two, r), _mm256_add_ps(vol_sq, eps)); // Add epsilon

    // N_param = 2b / vol^2
    __m256 N_param = _mm256_div_ps(_mm256_mul_ps(two, b_val), _mm256_add_ps(vol_sq, eps)); // Add epsilon

    // q2_param = (-(N-1) + sqrt((N-1)^2 + 4M)) / 2
    __m256 N_minus_1 = _mm256_sub_ps(N_param, one);
    __m256 term_sqrt = num::simd::sqrt_ps(
        _mm256_add_ps(_mm256_mul_ps(N_minus_1, N_minus_1), _mm256_mul_ps(_mm256_set1_ps(4.0f), M_val))
    );
    __m256 q2_param = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_add_ps(_mm256_sub_ps(one, N_param), term_sqrt));


    // S_star (critical price) = K / (1 - 1/q2_param) = K * q2_param / (q2_param - 1)
    // Handle q2_param close to 1 or 0 carefully
    __m256 q2_minus_1 = _mm256_sub_ps(q2_param, one);
    __m256 S_star_denom_safe = _mm256_blendv_ps(q2_minus_1, _mm256_set1_ps(1.0f), _mm256_cmp_ps(_mm256_add_ps(q2_minus_1, eps), zero, _CMP_LE_OQ)); // Avoid div by zero
    __m256 S_star = _mm256_div_ps(_mm256_mul_ps(K, q2_param), S_star_denom_safe);

    // Mask for S <= S_star
    __m256 S_le_S_star_mask = _mm256_cmp_ps(S, S_star, _CMP_LE_OQ);

    // Intrinsic value: max(0, K-S)
    __m256 intrinsic = _mm256_max_ps(zero, _mm256_sub_ps(K, S));

    // Calculate A2 (premium term component)
    __m256 S_div_S_star_safe = _mm256_div_ps(S, _mm256_add_ps(S_star, eps)); // Avoid div by zero
    __m256 neg_q2 = _mm256_sub_ps(zero, q2_param);
    __m256 ratio_pow_neg_q2 = num::simd::exp_ps(_mm256_mul_ps(neg_q2, num::simd::log_ps(S_div_S_star_safe)));

    __m256 d1 = EuropeanD1(S, S_star, r, q, vol, T); // Note: Using S_star as strike here for A2
    __m256 N_neg_d1_S_star = num::simd::normal_cdf_ps(_mm256_sub_ps(zero, d1));
    __m256 exp_b_minus_r_T = num::simd::exp_ps(_mm256_mul_ps(_mm256_sub_ps(b_val,r), T));
    __m256 A2_factor = _mm256_sub_ps(one, _mm256_mul_ps(exp_b_minus_r_T, N_neg_d1_S_star));
    __m256 A2_term = _mm256_mul_ps( _mm256_div_ps(S_star, _mm256_add_ps(q2_param, eps)), A2_factor); // Add eps for safety
    
    __m256 premium = _mm256_mul_ps(A2_term, ratio_pow_neg_q2);

    // American price candidate
    __m256 candidate_price = _mm256_add_ps(euro_price, premium);
    candidate_price = _mm256_max_ps(candidate_price, intrinsic); // Cannot be less than intrinsic

    // Combine: if r > q and S <= S_star, price is intrinsic, else candidate_price
    __m256 final_mask = _mm256_and_ps(r_gt_q_mask, S_le_S_star_mask);
    american_price = _mm256_blendv_ps(candidate_price, intrinsic, final_mask);
    
    // Ensure it's not less than European price if r_gt_q_mask is false
    american_price = _mm256_blendv_ps(american_price, euro_price, _mm256_xor_ps(r_gt_q_mask, _mm256_castsi256_ps(_mm256_set1_epi32(-1)) ) );


    return american_price;
}


inline __m256 SimdOperationSingle::AmericanCall(__m256 S, __m256 K, __m256 r, __m256 q,
                                             __m256 vol, __m256 T) {
    __m256 euro_price = EuropeanCall(S, K, r, q, vol, T);
    __m256 zero = _mm256_setzero_ps();
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 eps = _mm256_set1_ps(1e-7f);

    // Mask for q > 0 (early exercise only valuable if q > 0)
    __m256 q_gt_zero_mask = _mm256_cmp_ps(q, eps, _CMP_GT_OQ);
    
    __m256 american_price = euro_price; // Default to European

    // Calculate b = r - q
    __m256 b_val = _mm256_sub_ps(r, q);

    // M = 2r / vol^2
    __m256 vol_sq = _mm256_mul_ps(vol, vol);
    __m256 M_val = _mm256_div_ps(_mm256_mul_ps(two, r), _mm256_add_ps(vol_sq, eps));

    // N_param = 2b / vol^2
    __m256 N_param = _mm256_div_ps(_mm256_mul_ps(two, b_val), _mm256_add_ps(vol_sq, eps));
    
    // q1_param = (-(N-1) - sqrt((N-1)^2 + 4M)) / 2  -- Note: Formula for q1 has minus before sqrt for calls
    __m256 N_minus_1 = _mm256_sub_ps(N_param, one);
    __m256 term_sqrt = num::simd::sqrt_ps(
        _mm256_add_ps(_mm256_mul_ps(N_minus_1, N_minus_1), _mm256_mul_ps(_mm256_set1_ps(4.0f), M_val))
    );
    __m256 q1_param = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_sub_ps(_mm256_sub_ps(one, N_param), term_sqrt));


    // S_star (critical price) = K / (1 - 1/q1_param) = K * q1_param / (q1_param - 1)
    // Handle q1_param close to 1 or 0 carefully
    __m256 q1_minus_1 = _mm256_sub_ps(q1_param, one);
    __m256 S_star_denom_safe = _mm256_blendv_ps(q1_minus_1, _mm256_set1_ps(1.0f), _mm256_cmp_ps(_mm256_add_ps(q1_minus_1, eps), zero, _CMP_LE_OQ));
    __m256 S_star = _mm256_div_ps(_mm256_mul_ps(K, q1_param), S_star_denom_safe);
    
    // Mask for S >= S_star
    __m256 S_ge_S_star_mask = _mm256_cmp_ps(S, S_star, _CMP_GE_OQ);

    // Intrinsic value: max(0, S-K)
    __m256 intrinsic = _mm256_max_ps(zero, _mm256_sub_ps(S, K));

    // Calculate A1 (premium term component)
    __m256 S_div_S_star_safe = _mm256_div_ps(S, _mm256_add_ps(S_star, eps));
    __m256 ratio_pow_q1 = num::simd::exp_ps(_mm256_mul_ps(q1_param, num::simd::log_ps(S_div_S_star_safe)));
    
    __m256 d1 = EuropeanD1(S, S_star, r, q, vol, T); // Note: Using S_star as strike here for A1
    __m256 N_d1_S_star = num::simd::normal_cdf_ps(d1);
    __m256 exp_b_minus_r_T = num::simd::exp_ps(_mm256_mul_ps(_mm256_sub_ps(b_val,r), T));
    __m256 A1_factor = _mm256_sub_ps(one, _mm256_mul_ps(exp_b_minus_r_T, N_d1_S_star));
    __m256 A1_term = _mm256_mul_ps( _mm256_div_ps(S_star, _mm256_add_ps(q1_param,eps)), A1_factor); // Add eps

    __m256 premium = _mm256_mul_ps(A1_term, ratio_pow_q1);

    // American price candidate
    __m256 candidate_price = _mm256_add_ps(euro_price, premium);
    candidate_price = _mm256_max_ps(candidate_price, intrinsic);

    // Combine: if q > 0 and S >= S_star, price is intrinsic, else candidate_price
    __m256 final_mask = _mm256_and_ps(q_gt_zero_mask, S_ge_S_star_mask);
    american_price = _mm256_blendv_ps(candidate_price, intrinsic, final_mask);
    
    // If q <= 0, ensure it's European
    american_price = _mm256_blendv_ps(american_price, euro_price, _mm256_xor_ps(q_gt_zero_mask, _mm256_castsi256_ps(_mm256_set1_epi32(-1)) ) );

    return american_price;
}


} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_SIMD_H
