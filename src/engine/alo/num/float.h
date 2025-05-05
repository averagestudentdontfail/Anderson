#ifndef ENGINE_ALO_NUM_FLOAT_H
#define ENGINE_ALO_NUM_FLOAT_H

#include <cmath>
#include <immintrin.h>

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
      1.0f - ((((1.061405429f * t + -1.453152027f) * t + 1.421413741f) * t +
               -0.284496736f) *
                  t +
              0.254829592f) *
                 t * std::exp(-x * x);
  return (x < 0) ? -result : result;
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
  return 0.5f * (1.0f + fast_erf(x / 1.414213562f));
}

/**
 * @brief Fast normal PDF calculation for single-precision
 *
 * @param x Input value
 * @return Normal PDF value
 */
inline float fast_normal_pdf(float x) {
  static const float INV_SQRT_2PI = 0.3989422804f;
  return INV_SQRT_2PI * std::exp(-0.5f * x * x);
}

/**
 * @brief Interface for SIMD vectorized operations
 */
namespace simd {
/**
 * @brief Optimized 8-wide error function (AVX2)
 * @param x Input vector
 * @return Vector of erf values
 */
__m256 erf_ps(__m256 x);

/**
 * @brief Optimized 8-wide normal CDF
 * @param x Input vector
 * @return Vector of normal CDF values
 */
__m256 normal_cdf_ps(__m256 x);

/**
 * @brief Optimized 8-wide normal PDF
 * @param x Input vector
 * @return Vector of normal PDF values
 */
__m256 normal_pdf_ps(__m256 x);

/**
 * @brief Optimized 8-wide exponential function
 * @param x Input vector
 * @return Vector of exp values
 */
__m256 exp_ps(__m256 x);

/**
 * @brief Optimized 8-wide logarithm
 * @param x Input vector
 * @return Vector of log values
 */
__m256 log_ps(__m256 x);

/**
 * @brief Optimized 8-wide square root
 * @param x Input vector
 * @return Vector of sqrt values
 */
inline __m256 sqrt_ps(__m256 x) { return _mm256_sqrt_ps(x); }
} // namespace simd

} // namespace num
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_NUM_FLOAT_H