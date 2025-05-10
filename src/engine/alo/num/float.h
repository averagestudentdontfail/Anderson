#ifndef ENGINE_ALO_NUM_FLOAT_H
#define ENGINE_ALO_NUM_FLOAT_H

#include <cmath>
#include <immintrin.h>
#include <sleef.h>    

namespace engine {
namespace alo {
namespace num {

/**
 * @brief Fast approximation of error function for single-precision
 *
 * Uses the Abramowitz and Stegun approximation for the error function.
 * This provides a significant speedup over std::erf with minimal precision
 * loss.
 *
 * @param x Input value
 * @return Approximated erf(x)
 */
inline float fast_erf(float x) {
  // Abramowitz and Stegun approximation
  float t = 1.0f / (1.0f + 0.3275911f * std::abs(x));
  float result =
      1.0f - (((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t -
                0.284496736f) *
                   t +
               0.254829592f) *
                  t * std::exp(-x * x));
  return (x < 0.0f) ? -result : result;
}

/**
 * @brief Fast normal CDF approximation for single-precision
 *
 * Uses the relation: Φ(x) = 0.5 * (1 + erf(x/√2))
 *
 * @param x Input value
 * @return Approximated normal CDF value
 */
inline float fast_normal_cdf(float x) {
  return 0.5f * (1.0f + fast_erf(x * 0.70710678118f)); // 0.707... = 1/sqrt(2)
}

/**
 * @brief Fast normal PDF calculation for single-precision
 *
 * @param x Input value
 * @return Normal PDF value
 */
inline float fast_normal_pdf(float x) {
  static const float INV_SQRT_2PI = 0.3989422804f; // 1/sqrt(2*PI)
  return INV_SQRT_2PI * std::exp(-0.5f * x * x);
}

/**
 * @brief Interface for SIMD vectorized operations (single-precision)
 */
namespace simd {
/**
 * @brief Optimized 8-wide error function (AVX2) using SLEEF
 * @param x Input vector
 * @return Vector of erf values
 */
inline __m256 erf_ps(__m256 x) {
    return Sleef_erff8_u10avx2(x); // Using SLEEF's AVX2 erf for 8 floats
}

/**
 * @brief Optimized 8-wide normal CDF (AVX2)
 * @param x Input vector
 * @return Vector of normal CDF values
 */
inline __m256 normal_cdf_ps(__m256 x) {
    const __m256 HALF = _mm256_set1_ps(0.5f);
    const __m256 ONE = _mm256_set1_ps(1.0f);
    const __m256 SQRT2_INV_PS = _mm256_set1_ps(0.7071067811865475f); // 1/sqrt(2)
    __m256 scaled_x = _mm256_mul_ps(x, SQRT2_INV_PS);
    __m256 erf_result = erf_ps(scaled_x); // Uses Sleef_erff8_u10avx2
    return _mm256_mul_ps(HALF, _mm256_add_ps(ONE, erf_result));
}

/**
 * @brief Optimized 8-wide normal PDF (AVX2)
 * @param x Input vector
 * @return Vector of normal PDF values
 */
inline __m256 normal_pdf_ps(__m256 x) {
    const __m256 NEG_HALF = _mm256_set1_ps(-0.5f);
    const __m256 INV_SQRT_2PI_PS = _mm256_set1_ps(0.3989422804f); // 1/sqrt(2*PI)
    __m256 x_squared = _mm256_mul_ps(x, x);
    __m256 exponent = _mm256_mul_ps(NEG_HALF, x_squared);
    __m256 exp_term = Sleef_expf8_u10avx2(exponent); 
    return _mm256_mul_ps(exp_term, INV_SQRT_2PI_PS);
}

/**
 * @brief Optimized 8-wide exponential function (AVX2) using SLEEF
 * @param x Input vector
 * @return Vector of exp values
 */
inline __m256 exp_ps(__m256 x) {
    return Sleef_expf8_u10avx2(x); // Using SLEEF's AVX2 exp for 8 floats
}

/**
 * @brief Optimized 8-wide logarithm (AVX2) using SLEEF
 * @param x Input vector
 * @return Vector of log values
 */
inline __m256 log_ps(__m256 x) {
    return Sleef_logf8_u10avx2(x); // Using SLEEF's AVX2 log for 8 floats
}

/**
 * @brief Optimized 8-wide square root (AVX2)
 * @param x Input vector
 * @return Vector of sqrt values
 */
inline __m256 sqrt_ps(__m256 x) { 
    return _mm256_sqrt_ps(x); 
}

} // namespace simd
} // namespace num
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_NUM_FLOAT_H