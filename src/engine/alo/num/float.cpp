#include "float.h"

namespace engine {
namespace alo {
namespace num {
namespace simd {

/**
 * @brief Fast AVX2 implementation of the error function
 */
__m256 erf_ps(__m256 x) {
  // Extract sign for later reconstruction
  __m256 sign_bit = _mm256_and_ps(x, _mm256_set1_ps(-0.0f));

  // Take absolute value of x
  __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);

  // Constants for Abramowitz & Stegun approximation
  const __m256 a1 = _mm256_set1_ps(0.254829592f);
  const __m256 a2 = _mm256_set1_ps(-0.284496736f);
  const __m256 a3 = _mm256_set1_ps(1.421413741f);
  const __m256 a4 = _mm256_set1_ps(-1.453152027f);
  const __m256 a5 = _mm256_set1_ps(1.061405429f);
  const __m256 p = _mm256_set1_ps(0.3275911f);

  // Calculate t = 1/(1 + p*|x|)
  __m256 p_abs_x = _mm256_mul_ps(p, abs_x);
  __m256 denom = _mm256_add_ps(_mm256_set1_ps(1.0f), p_abs_x);
  __m256 t = _mm256_div_ps(_mm256_set1_ps(1.0f), denom);

  // Calculate polynomial using optimized FMA operations
  __m256 polynomial = a5;
  polynomial = _mm256_fmadd_ps(polynomial, t, a4);
  polynomial = _mm256_fmadd_ps(polynomial, t, a3);
  polynomial = _mm256_fmadd_ps(polynomial, t, a2);
  polynomial = _mm256_fmadd_ps(polynomial, t, a1);
  polynomial = _mm256_mul_ps(polynomial, t);

  // Calculate e^(-x^2)
  __m256 x_squared = _mm256_mul_ps(abs_x, abs_x);
  __m256 neg_x_squared = _mm256_mul_ps(x_squared, _mm256_set1_ps(-1.0f));
  __m256 exp_term = exp_ps(neg_x_squared);

  // Calculate final result: 1 - polynomial * exp(-x^2)
  __m256 result =
      _mm256_sub_ps(_mm256_set1_ps(1.0f), _mm256_mul_ps(polynomial, exp_term));

  // Restore sign
  return _mm256_xor_ps(result, sign_bit);
}

/**
 * @brief Fast AVX2 implementation of the normal CDF
 */
__m256 normal_cdf_ps(__m256 x) {
  // Scale factor for erf
  const __m256 SQRT2_INV = _mm256_set1_ps(0.7071067811865475f); // 1/sqrt(2)

  // Calculate erf(x/sqrt(2))
  __m256 scaled_x = _mm256_mul_ps(x, SQRT2_INV);
  __m256 erf_result = erf_ps(scaled_x);

  // Calculate 0.5 * (1 + erf(x/sqrt(2)))
  return _mm256_mul_ps(_mm256_set1_ps(0.5f),
                       _mm256_add_ps(_mm256_set1_ps(1.0f), erf_result));
}

/**
 * @brief Fast AVX2 implementation of the normal PDF
 */
__m256 normal_pdf_ps(__m256 x) {
  // Constants
  const __m256 INV_SQRT_2PI = _mm256_set1_ps(0.3989422804f);

  // Calculate -0.5 * x^2
  __m256 x_squared = _mm256_mul_ps(x, x);
  __m256 x_squared_halved = _mm256_mul_ps(x_squared, _mm256_set1_ps(-0.5f));

  // Calculate exp(-0.5 * x^2)
  __m256 exp_term = exp_ps(x_squared_halved);

  // Return INV_SQRT_2PI * exp_term
  return _mm256_mul_ps(INV_SQRT_2PI, exp_term);
}

/**
 * @brief Optimized exponential function for AVX2
 *
 * This is an optimized implementation based on a polynomial approximation,
 * providing a balance between speed and precision.
 */
__m256 exp_ps(__m256 x) {
  // Constants
  const __m256 LOG2E = _mm256_set1_ps(1.44269504088896341f);
  const __m256 C1 = _mm256_set1_ps(0.693147180559945f);
  const __m256 C2 = _mm256_set1_ps(0.240226506959101f);
  const __m256 C3 = _mm256_set1_ps(0.0555041086648216f);
  const __m256 C4 = _mm256_set1_ps(0.00961812905951724f);
  const __m256 C5 = _mm256_set1_ps(0.00133335582175501f);
  const __m256 C6 = _mm256_set1_ps(1.54097299613915e-4f);
  const __m256 C7 = _mm256_set1_ps(1.52587890625e-5f);

  // Range reduction: exp(x) = 2^integer_part * exp(fractional_part)
  __m256 tx = _mm256_mul_ps(x, LOG2E);
  __m256 txs = _mm256_round_ps(tx, _MM_FROUND_TO_NEAREST_INT);
  __m256 fx = _mm256_sub_ps(x, _mm256_mul_ps(txs, C1));

  // Polynomial approximation for exp(fx) in reduced range
  __m256 result = C7;
  result = _mm256_fmadd_ps(result, fx, C6);
  result = _mm256_fmadd_ps(result, fx, C5);
  result = _mm256_fmadd_ps(result, fx, C4);
  result = _mm256_fmadd_ps(result, fx, C3);
  result = _mm256_fmadd_ps(result, fx, C2);
  result = _mm256_fmadd_ps(result, fx, _mm256_set1_ps(1.0f));

  // Scale by 2^integer_part
  // Convert integer part to an integer vector
  __m256i emm0 = _mm256_cvtps_epi32(txs);
  // Shift to get 2^n in floating point
  emm0 = _mm256_add_epi32(emm0, _mm256_set1_epi32(127));
  emm0 = _mm256_slli_epi32(emm0, 23);
  __m256 pow2n = _mm256_castsi256_ps(emm0);

  // Final result: 2^integer_part * exp(fractional_part)
  return _mm256_mul_ps(result, pow2n);
}

/**
 * @brief Optimized natural logarithm function for AVX2
 *
 * This is an implementation based on the standard approach:
 * log(x) = log(2) * log2(x)
 */
__m256 log_ps(__m256 x) {
  // Constants
  const __m256 LN2 = _mm256_set1_ps(0.693147180559945f);
  const __m256 ONE = _mm256_set1_ps(1.0f);
  const __m256 C1 = _mm256_set1_ps(0.5f);
  const __m256 C2 = _mm256_set1_ps(0.333333333333f);
  const __m256 C3 = _mm256_set1_ps(0.25f);
  const __m256 C4 = _mm256_set1_ps(0.2f);

  // Get the exponent
  __m256i emm0 = _mm256_castps_si256(x);
  __m256i exponent = _mm256_srli_epi32(emm0, 23);
  exponent = _mm256_sub_epi32(exponent, _mm256_set1_epi32(127));
  __m256 e = _mm256_cvtepi32_ps(exponent);

  // Get the mantissa
  __m256i mmm0 = _mm256_and_si256(emm0, _mm256_set1_epi32(0x007FFFFF));
  __m256 mantissa =
      _mm256_castsi256_ps(_mm256_or_si256(mmm0, _mm256_castps_si256(ONE)));

  // Range reduction: log(x) = log(2)*exponent + log(mantissa)
  // where 1 <= mantissa < 2

  // Apply polynomial approximation for log(1+y) where y = mantissa-1
  __m256 y = _mm256_sub_ps(mantissa, ONE);
  __m256 y2 = _mm256_mul_ps(y, y);

  // Use the approximation log(1+y) ≈ y - y^2/2 + y^3/3 - y^4/4 + y^5/5
  __m256 p1 = _mm256_mul_ps(y, _mm256_set1_ps(1.0f));
  __m256 p2 = _mm256_mul_ps(y2, C1);
  __m256 p3 = _mm256_mul_ps(y2, y);
  p3 = _mm256_mul_ps(p3, C2);
  __m256 p4 = _mm256_mul_ps(y2, y2);
  p4 = _mm256_mul_ps(p4, C3);
  __m256 p5 = _mm256_mul_ps(p4, y);
  p5 = _mm256_mul_ps(p5, C4);

  __m256 result = p1;
  result = _mm256_sub_ps(result, p2);
  result = _mm256_add_ps(result, p3);
  result = _mm256_sub_ps(result, p4);
  result = _mm256_add_ps(result, p5);

  // Add the exponent part: log(2)*e
  result = _mm256_add_ps(result, _mm256_mul_ps(e, LN2));

  return result;
}

} // namespace simd
} // namespace num
} // namespace alo
} // namespace engine