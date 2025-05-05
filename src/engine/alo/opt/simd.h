#ifndef ENGINE_ALO_OPT_SIMD_H
#define ENGINE_ALO_OPT_SIMD_H

#include "../num/float.h"
#include <immintrin.h>

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
  static inline __m256d load(const std::array<double, 4> &arr) {
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
  static inline void store(std::array<double, 4> &arr, __m256d vec) {
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
   * @brief Calculate European option d1 for 4 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing d1 values
   */
  static inline __m256d EuropeanD1(__m256d S, __m256d K, __m256d r, __m256d q,
                                   __m256d vol, __m256d T) {
    // Prepare intermediate values
    __m256d sqrt_T = _mm256_sqrt_pd(T);
    __m256d vol_sqrt_T = _mm256_mul_pd(vol, sqrt_T);

    // Calculate log(S/K) using SLEEF
    __m256d S_div_K = _mm256_div_pd(S, K);
    __m256d log_S_div_K = Sleef_logd4_u10avx2(S_div_K);

    // Calculate drift term
    __m256d vol_squared = _mm256_mul_pd(vol, vol);
    __m256d half_vol_squared = _mm256_mul_pd(_mm256_set1_pd(0.5), vol_squared);
    __m256d r_minus_q = _mm256_sub_pd(r, q);
    __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
    __m256d drift_T = _mm256_mul_pd(drift, T);

    // Combine terms and calculate d1
    __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
    return _mm256_div_pd(numerator, vol_sqrt_T);
  }

  /**
   * @brief Calculate European option d2 from d1
   *
   * @param d1 d1 values
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing d2 values
   */
  static inline __m256d EuropeanD2(__m256d d1, __m256d vol, __m256d T) {
    // d2 = d1 - vol * sqrt(T)
    __m256d sqrt_T = _mm256_sqrt_pd(T);
    __m256d vol_sqrt_T = _mm256_mul_pd(vol, sqrt_T);
    return _mm256_sub_pd(d1, vol_sqrt_T);
  }

  /**
   * @brief Calculate European put prices for 4 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing put option prices
   */
  static inline __m256d EuropeanPut(__m256d S, __m256d K, __m256d r, __m256d q,
                                    __m256d vol, __m256d T) {
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
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d d2 = EuropeanD2(d1, vol, T);

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
   * @brief Calculate European call prices for 4 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing call option prices
   */
  static inline __m256d EuropeanCall(__m256d S, __m256d K, __m256d r, __m256d q,
                                     __m256d vol, __m256d T) {
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
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d d2 = EuropeanD2(d1, vol, T);

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

    // Create masks for extreme values
    __m256d large_neg_mask = _mm256_cmp_pd(x, _mm256_set1_pd(-8.0), _CMP_LT_OS);
    __m256d large_pos_mask = _mm256_cmp_pd(x, _mm256_set1_pd(8.0), _CMP_GT_OS);

    // Calculate erf(x/sqrt(2))
    __m256d scaled_x = _mm256_mul_pd(x, SQRT2_INV);
    __m256d erf_result = Sleef_erfd4_u10avx2(scaled_x);

    // Normal range: 0.5 * (1 + erf(x/sqrt(2)))
    __m256d normal_result = _mm256_mul_pd(HALF, _mm256_add_pd(ONE, erf_result));

    // Handle extreme values
    __m256d result =
        _mm256_blendv_pd(normal_result,
                         _mm256_setzero_pd(), // Return 0 for large negative x
                         large_neg_mask);

    result = _mm256_blendv_pd(result,
                              ONE, // Return 1 for large positive x
                              large_pos_mask);

    return result;
  }

  /**
   * @brief Normal PDF calculation
   *
   * @param x Input vector
   * @return Vector containing normalPDF(x)
   */
  static inline __m256d normalPDF(__m256d x) {
    const __m256d NEG_HALF = _mm256_set1_pd(-0.5);
    const __m256d INV_SQRT_2PI =
        _mm256_set1_pd(0.3989422804014327); // 1/sqrt(2π)

    // normalPDF(x) = exp(-0.5 * x^2) / sqrt(2*PI)
    __m256d x_squared = _mm256_mul_pd(x, x);
    __m256d exponent = _mm256_mul_pd(NEG_HALF, x_squared);

    // Use SLEEF's exponential function
    __m256d exp_term = Sleef_expd4_u10avx2(exponent);

    return _mm256_mul_pd(exp_term, INV_SQRT_2PI);
  }

  /**
   * @brief Calculate option delta for 4 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @param isCall true for call options, false for put options
   * @return Vector containing delta values
   */
  static inline __m256d delta(__m256d S, __m256d K, __m256d r, __m256d q,
                              __m256d vol, __m256d T, bool isCall) {
    // Calculate d1
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);

    // Calculate discount factor e^(-q*T)
    __m256d neg_q_T = _mm256_mul_pd(_mm256_mul_pd(q, T), _mm256_set1_pd(-1.0));
    __m256d dq = Sleef_expd4_u10avx2(neg_q_T);

    if (isCall) {
      // Call delta = e^(-q*T) * N(d1)
      __m256d Nd1 = normalCDF(d1);
      return _mm256_mul_pd(dq, Nd1);
    } else {
      // Put delta = -e^(-q*T) * N(-d1)
      __m256d neg_d1 = _mm256_sub_pd(_mm256_setzero_pd(), d1);
      __m256d Nneg_d1 = normalCDF(neg_d1);
      return _mm256_mul_pd(_mm256_set1_pd(-1.0), _mm256_mul_pd(dq, Nneg_d1));
    }
  }

  /**
   * @brief Calculate option gamma for 4 options at once
   *
   * Gamma is the same for both put and call
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing gamma values
   */
  static inline __m256d gamma(__m256d S, __m256d K, __m256d r, __m256d q,
                              __m256d vol, __m256d T) {
    // Calculate d1
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);

    // Calculate n(d1) - normal density function
    __m256d pdf_d1 = normalPDF(d1);

    // Calculate discount factor e^(-q*T)
    __m256d neg_q_T = _mm256_mul_pd(_mm256_mul_pd(q, T), _mm256_set1_pd(-1.0));
    __m256d dq = Sleef_expd4_u10avx2(neg_q_T);

    // Calculate S * vol * sqrt(T)
    __m256d sqrt_T = _mm256_sqrt_pd(T);
    __m256d vol_sqrt_T = _mm256_mul_pd(vol, sqrt_T);
    __m256d denominator = _mm256_mul_pd(S, vol_sqrt_T);

    // gamma = e^(-q*T) * n(d1) / (S * vol * sqrt(T))
    __m256d numerator = _mm256_mul_pd(dq, pdf_d1);
    return _mm256_div_pd(numerator, denominator);
  }

  /**
   * @brief Calculate option vega for 4 options at once
   *
   * Vega is the same for both put and call
   * Vega is defined as dV/dσ * 0.01 for a 1% change in volatility
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing vega values
   */
  static inline __m256d vega(__m256d S, __m256d K, __m256d r, __m256d q,
                             __m256d vol, __m256d T) {
    // Calculate d1
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);

    // Calculate n(d1) - normal density function
    __m256d pdf_d1 = normalPDF(d1);

    // Calculate discount factor e^(-q*T)
    __m256d neg_q_T = _mm256_mul_pd(_mm256_mul_pd(q, T), _mm256_set1_pd(-1.0));
    __m256d dq = Sleef_expd4_u10avx2(neg_q_T);

    // Calculate sqrt(T)
    __m256d sqrt_T = _mm256_sqrt_pd(T);

    // vega = S * e^(-q*T) * n(d1) * sqrt(T) * 0.01
    __m256d result = _mm256_mul_pd(S, dq);
    result = _mm256_mul_pd(result, pdf_d1);
    result = _mm256_mul_pd(result, sqrt_T);
    return _mm256_mul_pd(result, _mm256_set1_pd(0.01)); // Scale for 1% change
  }

  /**
   * @brief Calculate option theta for 4 options at once
   *
   * Theta is defined as dV/dT * (1/365) for a 1-day change in time
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @param isCall true for call options, false for put options
   * @return Vector containing theta values
   */
  static inline __m256d theta(__m256d S, __m256d K, __m256d r, __m256d q,
                              __m256d vol, __m256d T, bool isCall) {
    // Calculate d1 and d2
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d d2 = EuropeanD2(d1, vol, T);

    // Calculate n(d1) - normal density function
    __m256d pdf_d1 = normalPDF(d1);

    // Calculate discount factors
    __m256d zero = _mm256_setzero_pd();
    __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
    __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q), T);
    __m256d dr = Sleef_expd4_u10avx2(neg_r_T);
    __m256d dq = Sleef_expd4_u10avx2(neg_q_T);

    // Calculate sqrt(T)
    __m256d sqrt_T = _mm256_sqrt_pd(T);

    // First term: -S * e^(-q*T) * n(d1) * vol / (2*sqrt(T))
    __m256d term1_factor =
        _mm256_div_pd(vol, _mm256_mul_pd(_mm256_set1_pd(2.0), sqrt_T));
    __m256d term1 = _mm256_mul_pd(S, dq);
    term1 = _mm256_mul_pd(term1, pdf_d1);
    term1 = _mm256_mul_pd(term1, term1_factor);
    term1 = _mm256_sub_pd(zero, term1); // Negate

    // For call and put, the first term is the same but remaining terms differ
    if (isCall) {
      // N(d1) and N(d2)
      __m256d Nd1 = normalCDF(d1);
      __m256d Nd2 = normalCDF(d2);

      // Second term: -q*S*e^(-q*T)*N(d1)
      __m256d term2 =
          _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(zero, q), S), dq);
      term2 = _mm256_mul_pd(term2, Nd1);

      // Third term: r*K*e^(-r*T)*N(d2)
      __m256d term3 = _mm256_mul_pd(_mm256_mul_pd(r, K), dr);
      term3 = _mm256_mul_pd(term3, Nd2);

      // theta = (term1 + term2 + term3) / 365
      __m256d theta_value = _mm256_add_pd(term1, _mm256_add_pd(term2, term3));
      return _mm256_div_pd(theta_value, _mm256_set1_pd(365.0));
    } else {
      // N(-d1) and N(-d2)
      __m256d neg_d1 = _mm256_sub_pd(zero, d1);
      __m256d neg_d2 = _mm256_sub_pd(zero, d2);
      __m256d Nneg_d1 = normalCDF(neg_d1);
      __m256d Nneg_d2 = normalCDF(neg_d2);

      // Second term: q*S*e^(-q*T)*N(-d1)
      __m256d term2 = _mm256_mul_pd(_mm256_mul_pd(q, S), dq);
      term2 = _mm256_mul_pd(term2, Nneg_d1);

      // Third term: -r*K*e^(-r*T)*N(-d2)
      __m256d term3 =
          _mm256_mul_pd(_mm256_mul_pd(_mm256_sub_pd(zero, r), K), dr);
      term3 = _mm256_mul_pd(term3, Nneg_d2);

      // theta = (term1 + term2 + term3) / 365
      __m256d theta_value = _mm256_add_pd(term1, _mm256_add_pd(term2, term3));
      return _mm256_div_pd(theta_value, _mm256_set1_pd(365.0));
    }
  }

  /**
   * @brief Calculate option rho for 4 options at once
   *
   * Rho is defined as dV/dr * 0.01 for a 1% change in interest rate
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @param isCall true for call options, false for put options
   * @return Vector containing rho values
   */
  static inline __m256d rho(__m256d S, __m256d K, __m256d r, __m256d q,
                            __m256d vol, __m256d T, bool isCall) {
    // Calculate d2
    __m256d d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256d d2 = EuropeanD2(d1, vol, T);

    // Calculate discount factor e^(-r*T)
    __m256d zero = _mm256_setzero_pd();
    __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
    __m256d dr = Sleef_expd4_u10avx2(neg_r_T);

    // Calculate K*T*e^(-r*T) * 0.01
    __m256d K_T = _mm256_mul_pd(K, T);
    __m256d base = _mm256_mul_pd(K_T, dr);
    __m256d scale = _mm256_set1_pd(0.01); // Scale for 1% change

    if (isCall) {
      // For call: rho = K*T*e^(-r*T)*N(d2) * 0.01
      __m256d Nd2 = normalCDF(d2);
      return _mm256_mul_pd(_mm256_mul_pd(base, Nd2), scale);
    } else {
      // For put: rho = -K*T*e^(-r*T)*N(-d2) * 0.01
      __m256d neg_d2 = _mm256_sub_pd(zero, d2);
      __m256d Nneg_d2 = normalCDF(neg_d2);
      return _mm256_mul_pd(
          _mm256_mul_pd(_mm256_mul_pd(_mm256_set1_pd(-1.0), base), Nneg_d2),
          scale);
    }
  }
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
   *
   * @param ptr Pointer to array of 8 floats
   * @return 256-bit vector containing 8 floats
   */
  static inline __m256 load(const float *ptr) { return _mm256_loadu_ps(ptr); }

  /**
   * @brief Store a vector into 8 floats
   *
   * @param ptr Pointer to array of 8 floats
   * @param vec 256-bit vector containing 8 floats
   */
  static inline void store(float *ptr, __m256 vec) {
    _mm256_storeu_ps(ptr, vec);
  }

  /**
   * @brief Set all elements to a scalar value
   *
   * @param value Scalar value
   * @return 256-bit vector with all elements set to value
   */
  static inline __m256 set1(float value) { return _mm256_set1_ps(value); }

  /**
   * @brief Calculate European option d1 for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing d1 values
   */
  static inline __m256 EuropeanD1(__m256 S, __m256 K, __m256 r, __m256 q,
                                  __m256 vol, __m256 T) {
    // Prepare intermediate values
    __m256 sqrt_T = _mm256_sqrt_ps(T);
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);

    // Calculate log(S/K) using our optimized log function
    __m256 S_div_K = _mm256_div_ps(S, K);
    __m256 log_S_div_K = num::simd::log_ps(S_div_K);

    // Calculate drift term: (r-q) + 0.5*vol^2
    __m256 r_minus_q = _mm256_sub_ps(r, q);
    __m256 vol_squared = _mm256_mul_ps(vol, vol);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 half_vol_squared = _mm256_mul_ps(half, vol_squared);
    __m256 drift = _mm256_add_ps(r_minus_q, half_vol_squared);

    // Calculate drift * T
    __m256 drift_T = _mm256_mul_ps(drift, T);

    // Calculate d1 = (log(S/K) + (r-q+0.5*vol^2)*T) / (vol*sqrt(T))
    __m256 numerator = _mm256_add_ps(log_S_div_K, drift_T);
    return _mm256_div_ps(numerator, vol_sqrt_T);
  }

  /**
   * @brief Calculate European option d2 from d1
   *
   * @param d1 d1 values
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing d2 values
   */
  static inline __m256 EuropeanD2(__m256 d1, __m256 vol, __m256 T) {
    // d2 = d1 - vol * sqrt(T)
    __m256 sqrt_T = _mm256_sqrt_ps(T);
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);
    return _mm256_sub_ps(d1, vol_sqrt_T);
  }

  /**
   * @brief Calculate European put prices for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing put option prices
   */
  static inline __m256 EuropeanPut(__m256 S, __m256 K, __m256 r, __m256 q,
                                   __m256 vol, __m256 T) {
    // Check for degenerate cases
    __m256 zero = _mm256_setzero_ps();
    __m256 eps = _mm256_set1_ps(1e-6f);

    // Create mask for vol <= 0 or T <= 0
    __m256 vol_mask = _mm256_cmp_ps(vol, eps, _CMP_LE_OQ);
    __m256 t_mask = _mm256_cmp_ps(T, eps, _CMP_LE_OQ);
    __m256 degenerate_mask = _mm256_or_ps(vol_mask, t_mask);

    // Calculate K-S for degenerate cases
    __m256 K_minus_S = _mm256_sub_ps(K, S);
    __m256 degenerate_value = _mm256_max_ps(zero, K_minus_S);

    // For non-degenerate cases, calculate d1
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);

    // Calculate d2 = d1 - vol*sqrt(T)
    __m256 sqrt_T = _mm256_sqrt_ps(T);
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);
    __m256 d2 = _mm256_sub_ps(d1, vol_sqrt_T);

    // Calculate -d1 and -d2 for put formula
    __m256 neg_d1 = _mm256_sub_ps(zero, d1);
    __m256 neg_d2 = _mm256_sub_ps(zero, d2);

    // Calculate N(-d1) and N(-d2) using our optimized normal CDF
    __m256 Nd1 = num::simd::normal_cdf_ps(neg_d1);
    __m256 Nd2 = num::simd::normal_cdf_ps(neg_d2);

    // Calculate discount factors
    __m256 neg_r_T = _mm256_mul_ps(_mm256_mul_ps(r, T), _mm256_set1_ps(-1.0f));
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dr = num::simd::exp_ps(neg_r_T);
    __m256 dq = num::simd::exp_ps(neg_q_T);

    // Calculate put price components
    __m256 term1 = _mm256_mul_ps(K, dr);
    term1 = _mm256_mul_ps(term1, Nd2);

    __m256 term2 = _mm256_mul_ps(S, dq);
    term2 = _mm256_mul_ps(term2, Nd1);

    // Calculate put = K*e^(-r*T)*N(-d2) - S*e^(-q*T)*N(-d1)
    __m256 put_value = _mm256_sub_ps(term1, term2);

    // Blend degenerate and computed values
    return _mm256_blendv_ps(put_value, degenerate_value, degenerate_mask);
  }

  /**
   * @brief Calculate European call prices for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing call option prices
   */
  static inline __m256 EuropeanCall(__m256 S, __m256 K, __m256 r, __m256 q,
                                    __m256 vol, __m256 T) {
    // Check for degenerate cases
    __m256 zero = _mm256_setzero_ps();
    __m256 eps = _mm256_set1_ps(1e-6f);

    // Create mask for vol <= 0 or T <= 0
    __m256 vol_mask = _mm256_cmp_ps(vol, eps, _CMP_LE_OQ);
    __m256 t_mask = _mm256_cmp_ps(T, eps, _CMP_LE_OQ);
    __m256 degenerate_mask = _mm256_or_ps(vol_mask, t_mask);

    // Calculate S-K for degenerate cases
    __m256 S_minus_K = _mm256_sub_ps(S, K);
    __m256 degenerate_value = _mm256_max_ps(zero, S_minus_K);

    // For non-degenerate cases, calculate d1
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);

    // Calculate d2 = d1 - vol*sqrt(T)
    __m256 sqrt_T = _mm256_sqrt_ps(T);
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);
    __m256 d2 = _mm256_sub_ps(d1, vol_sqrt_T);

    // Calculate N(d1) and N(d2) using our optimized normal CDF
    __m256 Nd1 = num::simd::normal_cdf_ps(d1);
    __m256 Nd2 = num::simd::normal_cdf_ps(d2);

    // Calculate discount factors
    __m256 neg_r_T = _mm256_mul_ps(_mm256_mul_ps(r, T), _mm256_set1_ps(-1.0f));
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dr = num::simd::exp_ps(neg_r_T);
    __m256 dq = num::simd::exp_ps(neg_q_T);

    // Calculate call price components
    __m256 term1 = _mm256_mul_ps(S, dq);
    term1 = _mm256_mul_ps(term1, Nd1);

    __m256 term2 = _mm256_mul_ps(K, dr);
    term2 = _mm256_mul_ps(term2, Nd2);

    // Calculate call = S*e^(-q*T)*N(d1) - K*e^(-r*T)*N(d2)
    __m256 call_value = _mm256_sub_ps(term1, term2);

    // Blend degenerate and computed values
    return _mm256_blendv_ps(call_value, degenerate_value, degenerate_mask);
  }

  /**
   * @brief Calculate option delta for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @param isCall true for call options, false for put options
   * @return Vector containing delta values
   */
  static inline __m256 delta(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                             __m256 T, bool isCall) {
    // Calculate d1
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);

    // Calculate discount factor e^(-q*T)
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dq = num::simd::exp_ps(neg_q_T);

    // For call: delta = e^(-q*T) * N(d1)
    // For put:  delta = e^(-q*T) * (N(d1) - 1) = -e^(-q*T) * N(-d1)
    if (isCall) {
      __m256 Nd1 = num::simd::normal_cdf_ps(d1);
      return _mm256_mul_ps(dq, Nd1);
    } else {
      __m256 neg_d1 = _mm256_sub_ps(_mm256_setzero_ps(), d1);
      __m256 Nneg_d1 = num::simd::normal_cdf_ps(neg_d1);
      return _mm256_mul_ps(dq, _mm256_sub_ps(_mm256_set1_ps(0.0f), Nneg_d1));
    }
  }

  /**
   * @brief Calculate option gamma for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing gamma values
   */
  static inline __m256 gamma(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                             __m256 T) {
    // Calculate d1
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);

    // Calculate norma pdf of d1
    __m256 d1_squared = _mm256_mul_ps(d1, d1);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 neg_half_d1_squared =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), half), d1_squared);
    __m256 exp_term = num::simd::exp_ps(neg_half_d1_squared);
    __m256 inv_sqrt_2pi = _mm256_set1_ps(0.3989422804f); // 1/sqrt(2π)
    __m256 pdf_d1 = _mm256_mul_ps(inv_sqrt_2pi, exp_term);

    // Calculate discount factor e^(-q*T)
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dq = num::simd::exp_ps(neg_q_T);

    // Calculate vol*sqrt(T)
    __m256 sqrt_T = _mm256_sqrt_ps(T);
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);

    // Calculate S * vol * sqrt(T)
    __m256 S_vol_sqrt_T = _mm256_mul_ps(S, vol_sqrt_T);

    // gamma = e^(-q*T) * pdf(d1) / (S * vol * sqrt(T))
    __m256 numerator = _mm256_mul_ps(dq, pdf_d1);
    return _mm256_div_ps(numerator, S_vol_sqrt_T);
  }

  /**
   * @brief Calculate option vega for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing vega values
   */
  static inline __m256 vega(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                            __m256 T) {
    // Calculate d1
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);

    // Calculate normal pdf of d1
    __m256 d1_squared = _mm256_mul_ps(d1, d1);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 neg_half_d1_squared =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), half), d1_squared);
    __m256 exp_term = num::simd::exp_ps(neg_half_d1_squared);
    __m256 inv_sqrt_2pi = _mm256_set1_ps(0.3989422804f); // 1/sqrt(2π)
    __m256 pdf_d1 = _mm256_mul_ps(inv_sqrt_2pi, exp_term);

    // Calculate discount factor e^(-q*T)
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dq = num::simd::exp_ps(neg_q_T);

    // Calculate sqrt(T)
    __m256 sqrt_T = _mm256_sqrt_ps(T);

    // Calculate S * e^(-q*T) * pdf(d1) * sqrt(T) * 0.01
    __m256 S_dq = _mm256_mul_ps(S, dq);
    __m256 pdf_sqrt_T = _mm256_mul_ps(pdf_d1, sqrt_T);
    __m256 hundredth = _mm256_set1_ps(0.01f); // For percentage points

    return _mm256_mul_ps(_mm256_mul_ps(S_dq, pdf_sqrt_T), hundredth);
  }

  /**
   * @brief Calculate option theta for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @param isCall true for call options, false for put options
   * @return Vector containing theta values
   */
  static inline __m256 theta(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                             __m256 T, bool isCall) {
    // Calculate d1 and d2
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256 sqrt_T = _mm256_sqrt_ps(T);
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);
    __m256 d2 = _mm256_sub_ps(d1, vol_sqrt_T);

    // Calculate normal pdf of d1
    __m256 d1_squared = _mm256_mul_ps(d1, d1);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 neg_half_d1_squared =
        _mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), half), d1_squared);
    __m256 exp_term = num::simd::exp_ps(neg_half_d1_squared);
    __m256 inv_sqrt_2pi = _mm256_set1_ps(0.3989422804f); // 1/sqrt(2π)
    __m256 pdf_d1 = _mm256_mul_ps(inv_sqrt_2pi, exp_term);

    // Calculate discount factors
    __m256 neg_r_T = _mm256_mul_ps(_mm256_mul_ps(r, T), _mm256_set1_ps(-1.0f));
    __m256 neg_q_T = _mm256_mul_ps(_mm256_mul_ps(q, T), _mm256_set1_ps(-1.0f));
    __m256 dr = num::simd::exp_ps(neg_r_T);
    __m256 dq = num::simd::exp_ps(neg_q_T);

    // Calculate first term (same for both call and put):
    // -S*e^(-q*T)*pdf(d1)*vol/(2*sqrt(T))
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 two_sqrt_T = _mm256_mul_ps(two, sqrt_T);
    __m256 vol_by_two_sqrt_T = _mm256_div_ps(vol, two_sqrt_T);
    __m256 S_dq_pdf_d1 = _mm256_mul_ps(_mm256_mul_ps(S, dq), pdf_d1);
    __m256 term1 = _mm256_mul_ps(
        _mm256_sub_ps(_mm256_setzero_ps(), S_dq_pdf_d1), vol_by_two_sqrt_T);

    // Calculate remaining terms
    if (isCall) {
      // Call: -q*S*e^(-q*T)*N(d1) + r*K*e^(-r*T)*N(d2)
      __m256 Nd1 = num::simd::normal_cdf_ps(d1);
      __m256 Nd2 = num::simd::normal_cdf_ps(d2);

      __m256 term2 =
          _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), q), S),
                        _mm256_mul_ps(dq, Nd1));
      __m256 term3 = _mm256_mul_ps(_mm256_mul_ps(r, K), _mm256_mul_ps(dr, Nd2));

      // Combine terms and divide by 365 for daily theta
      __m256 raw_theta = _mm256_add_ps(term1, _mm256_add_ps(term2, term3));
      return _mm256_div_ps(raw_theta, _mm256_set1_ps(365.0f));
    } else {
      // Put: -q*S*e^(-q*T)*N(-d1) + r*K*e^(-r*T)*N(-d2)
      __m256 Nneg_d1 =
          num::simd::normal_cdf_ps(_mm256_sub_ps(_mm256_setzero_ps(), d1));
      __m256 Nneg_d2 =
          num::simd::normal_cdf_ps(_mm256_sub_ps(_mm256_setzero_ps(), d2));

      __m256 term2 =
          _mm256_mul_ps(_mm256_mul_ps(_mm256_sub_ps(_mm256_setzero_ps(), q), S),
                        _mm256_mul_ps(dq, Nneg_d1));
      __m256 term3 =
          _mm256_mul_ps(_mm256_mul_ps(r, K), _mm256_mul_ps(dr, Nneg_d2));

      // Combine terms and divide by 365 for daily theta
      __m256 raw_theta = _mm256_add_ps(term1, _mm256_add_ps(term2, term3));
      return _mm256_div_ps(raw_theta, _mm256_set1_ps(365.0f));
    }
  }

  /**
   * @brief Calculate option rho for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @param isCall true for call options, false for put options
   * @return Vector containing rho values
   */
  static inline __m256 rho(__m256 S, __m256 K, __m256 r, __m256 q, __m256 vol,
                           __m256 T, bool isCall) {
    // Calculate d2
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256 sqrt_T = _mm256_sqrt_ps(T);
    __m256 vol_sqrt_T = _mm256_mul_ps(vol, sqrt_T);
    __m256 d2 = _mm256_sub_ps(d1, vol_sqrt_T);

    // Calculate discount factor e^(-r*T)
    __m256 neg_r_T = _mm256_mul_ps(_mm256_mul_ps(r, T), _mm256_set1_ps(-1.0f));
    __m256 dr = num::simd::exp_ps(neg_r_T);

    // Calculate K*T*e^(-r*T)
    __m256 K_T = _mm256_mul_ps(K, T);
    __m256 K_T_dr = _mm256_mul_ps(K_T, dr);
    __m256 hundredth = _mm256_set1_ps(0.01f); // For percentage points

    if (isCall) {
      // For call: rho = K*T*e^(-r*T)*N(d2)*0.01
      __m256 Nd2 = num::simd::normal_cdf_ps(d2);
      return _mm256_mul_ps(_mm256_mul_ps(K_T_dr, Nd2), hundredth);
    } else {
      // For put: rho = -K*T*e^(-r*T)*N(-d2)*0.01
      __m256 Nneg_d2 =
          num::simd::normal_cdf_ps(_mm256_sub_ps(_mm256_setzero_ps(), d2));
      return _mm256_mul_ps(
          _mm256_mul_ps(_mm256_mul_ps(_mm256_set1_ps(-1.0f), K_T_dr), Nneg_d2),
          hundredth);
    }
  }

  /**
   * @brief Calculate American put prices for 8 options at once
   *
   * Uses Barone-Adesi-Whaley approximation
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing American put prices
   */
  static inline __m256 AmericanPut(__m256 S, __m256 K, __m256 r, __m256 q,
                                   __m256 vol, __m256 T) {
    // First calculate European price
    __m256 euro_price = EuropeanPut(S, K, r, q, vol, T);

    // Check for early exercise premium (r > q)
    __m256 r_gt_q = _mm256_cmp_ps(r, q, _CMP_GT_OQ);

    // For Barone-Adesi-Whaley approximation parameters
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 two = _mm256_set1_ps(2.0f);
    __m256 zero = _mm256_setzero_ps();

    // Calculate b = r - q
    __m256 b = _mm256_sub_ps(r, q);

    // Calculate M = 2r / vol^2
    __m256 vol_squared = _mm256_mul_ps(vol, vol);
    __m256 two_r = _mm256_mul_ps(two, r);
    __m256 M = _mm256_div_ps(two_r, vol_squared);

    // Calculate N = 2b / vol^2
    __m256 two_b = _mm256_mul_ps(two, b);
    __m256 N = _mm256_div_ps(two_b, vol_squared);

    // Calculate q2 for put approximation
    __m256 N_minus_one = _mm256_sub_ps(N, one);
    __m256 N_minus_one_squared = _mm256_mul_ps(N_minus_one, N_minus_one);
    __m256 four_M = _mm256_mul_ps(_mm256_set1_ps(4.0f), M);
    __m256 sqrt_term =
        _mm256_sqrt_ps(_mm256_add_ps(N_minus_one_squared, four_M));
    __m256 q2 = _mm256_mul_ps(_mm256_set1_ps(0.5f),
                              _mm256_add_ps(_mm256_sub_ps(one, N), sqrt_term));

    // Calculate critical price S*
    __m256 S_star = _mm256_div_ps(K, _mm256_add_ps(one, q2));

    // If S <= S*, exercise is optimal
    __m256 S_le_S_star = _mm256_cmp_ps(S, S_star, _CMP_LE_OQ);

    // Calculate intrinsic value K - S
    __m256 intrinsic = _mm256_max_ps(zero, _mm256_sub_ps(K, S));

    // For S > S*, add early exercise premium
    __m256 S_div_S_star = _mm256_div_ps(S, S_star);

    // Calculate (S/S*)^(-q2)
    // Use: (S/S*)^(-q2) = exp(-q2 * log(S/S*))
    __m256 log_ratio = num::simd::log_ps(S_div_S_star);
    __m256 neg_q2 = _mm256_sub_ps(zero, q2);
    __m256 log_pow = _mm256_mul_ps(neg_q2, log_ratio);
    __m256 ratio_pow = num::simd::exp_ps(log_pow);

    // Calculate alpha term: (K/q2) * (1 - exp((b-r)*T) * N(-d1))
    __m256 d1 = EuropeanD1(S, K, r, q, vol, T);
    __m256 neg_d1 = _mm256_sub_ps(zero, d1);
    __m256 Nd1 = num::simd::normal_cdf_ps(neg_d1);

    __m256 b_minus_r = _mm256_sub_ps(b, r);
    __m256 b_minus_r_T = _mm256_mul_ps(b_minus_r, T);
    __m256 exp_term = num::simd::exp_ps(b_minus_r_T);
    __m256 one_minus_term = _mm256_sub_ps(one, _mm256_mul_ps(exp_term, Nd1));

    __m256 K_div_q2 = _mm256_div_ps(K, q2);
    __m256 alpha = _mm256_mul_ps(K_div_q2, one_minus_term);

    // Calculate early exercise premium: alpha * (S/S*)^(-q2)
    __m256 premium = _mm256_mul_ps(alpha, ratio_pow);

    // Select between intrinsic and euro_price + premium
    __m256 american_price = _mm256_add_ps(euro_price, premium);
    american_price = _mm256_max_ps(american_price, intrinsic);

    // Select based on conditions: (r > q) && (S <= S_star)
    __m256 exercise_mask = _mm256_and_ps(r_gt_q, S_le_S_star);
    american_price = _mm256_blendv_ps(american_price, intrinsic, exercise_mask);

    return american_price;
  }

  /**
   * @brief Calculate American call prices for 8 options at once
   *
   * @param S Spot prices
   * @param K Strike prices
   * @param r Risk-free rates
   * @param q Dividend yields
   * @param vol Volatilities
   * @param T Times to maturity
   * @return Vector containing American call prices
   */
  static inline __m256 AmericanCall(__m256 S, __m256 K, __m256 r, __m256 q,
                                    __m256 vol, __m256 T) {
    // For zero dividend, American call = European call
    __m256 zero = _mm256_setzero_ps();
    __m256 eps = _mm256_set1_ps(1e-6f);
    __m256 q_le_zero = _mm256_cmp_ps(q, eps, _CMP_LE_OQ);

    // Calculate European call
    __m256 euro_price = EuropeanCall(S, K, r, q, vol, T);

    // Check if q <= 0, if so, use European price
    __m256 needs_american = _mm256_cmp_ps(q, eps, _CMP_GT_OQ);

    // For Barone-Adesi-Whaley parameters
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 two = _mm256_set1_ps(2.0f);

    // Calculate b = r - q
    __m256 b = _mm256_sub_ps(r, q);

    // Calculate M = 2r / vol^2
    __m256 vol_squared = _mm256_mul_ps(vol, vol);
    __m256 two_r = _mm256_mul_ps(two, r);
    __m256 M = _mm256_div_ps(two_r, vol_squared);

    // Calculate N = 2b / vol^2
    __m256 two_b = _mm256_mul_ps(two, b);
    __m256 N = _mm256_div_ps(two_b, vol_squared);

    // Calculate q1 for call approximation
    __m256 N_minus_one = _mm256_sub_ps(N, one);
    __m256 N_minus_one_squared = _mm256_mul_ps(N_minus_one, N_minus_one);
    __m256 four_M = _mm256_mul_ps(_mm256_set1_ps(4.0f), M);
    __m256 sqrt_term =
        _mm256_sqrt_ps(_mm256_add_ps(N_minus_one_squared, four_M));
    __m256 q1 = _mm256_mul_ps(_mm256_set1_ps(0.5f),
                              _mm256_sub_ps(_mm256_add_ps(one, N), sqrt_term));

    // Calculate critical price S*
    __m256 S_star = _mm256_div_ps(K, _mm256_sub_ps(one, q1));

    // If S >= S*, exercise is optimal
    __m256 S_ge_S_star = _mm256_cmp_ps(S, S_star, _CMP_GE_OQ);

    // Calculate intrinsic value S - K
    __m256 intrinsic = _mm256_max_ps(zero, _mm256_sub_ps(S, K));

    // For S < S*, add early exercise premium
    __m256 S_div_S_star = _mm256_div_ps(S, S_star);

    // Calculate (S/S*)^q1
    __m256 log_ratio = num::simd::log_ps(S_div_S_star);
    __m256 log_pow = _mm256_mul_ps(q1, log_ratio);
    __m256 ratio_pow = num::simd::exp_ps(log_pow);

    // Calculate alpha term: (S_star - K) * (1 - (S/S*)^q1)
    __m256 S_star_minus_K = _mm256_sub_ps(S_star, K);
    __m256 one_minus_ratio_pow = _mm256_sub_ps(one, ratio_pow);
    __m256 premium = _mm256_mul_ps(S_star_minus_K, one_minus_ratio_pow);

    // Select between intrinsic and euro_price + premium
    __m256 american_price = _mm256_add_ps(euro_price, premium);
    american_price = _mm256_max_ps(american_price, intrinsic);

    // Select based on conditions: q > 0 && S >= S_star
    __m256 exercise_mask = _mm256_and_ps(needs_american, S_ge_S_star);
    american_price = _mm256_blendv_ps(american_price, intrinsic, exercise_mask);

    // If q <= 0, use European price
    american_price = _mm256_blendv_ps(american_price, euro_price, q_le_zero);

    return american_price;
  }
};

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_SIMD_H