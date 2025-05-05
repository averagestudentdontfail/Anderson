#include "simd.h"
#include "../num/float.h"
#include <cmath>

namespace engine {
namespace alo {
namespace num {
namespace simd {

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

  // Calculate polynomial using optimized operations
  __m256 polynomial = _mm256_fmadd_ps(
      _mm256_fmadd_ps(
          _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_fmadd_ps(a5, t, a4), t, a3), t,
                          a2),
          t, a1),
      t, _mm256_setzero_ps());

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

// Implementation of normal_cdf_ps
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

// Implementation of normal_pdf_ps
__m256 normal_pdf_ps(__m256 x) {
  const __m256 NEG_HALF = _mm256_set1_ps(-0.5f);
  const __m256 INV_SQRT_2PI = _mm256_set1_ps(0.3989422804f); // 1/sqrt(2π)

  // Calculate x^2
  __m256 x_squared = _mm256_mul_ps(x, x);

  // Calculate -0.5 * x^2
  __m256 exponent = _mm256_mul_ps(NEG_HALF, x_squared);

  // Calculate exp(-0.5 * x^2)
  __m256 exp_term = exp_ps(exponent);

  // Calculate exp(-0.5 * x^2) / sqrt(2π)
  return _mm256_mul_ps(exp_term, INV_SQRT_2PI);
}

// Implementation of exp_ps using polynomial approximation
__m256 exp_ps(__m256 x) {
  // Clamp input to avoid overflow/underflow
  __m256 max_input = _mm256_set1_ps(88.3762626647949f);  // log(FLT_MAX)
  __m256 min_input = _mm256_set1_ps(-88.3762626647949f); // log(FLT_MIN)

  x = _mm256_max_ps(_mm256_min_ps(x, max_input), min_input);

  // Express e^x as 2^(x/ln(2))
  const __m256 ln2 = _mm256_set1_ps(0.693147180559945f);
  __m256 x_over_ln2 = _mm256_div_ps(x, ln2);

  // Split into integer and fractional parts
  __m256 x_int = _mm256_round_ps(x_over_ln2, _MM_FROUND_TO_NEAREST_INT);
  __m256 x_frac = _mm256_sub_ps(x_over_ln2, x_int);

  // Compute 2^frac using polynomial approximation
  // Coefficients for a degree-5 minimax approximation
  const __m256 c0 = _mm256_set1_ps(1.0f);
  const __m256 c1 = _mm256_set1_ps(0.693359375f);
  const __m256 c2 = _mm256_set1_ps(0.2400844f);
  const __m256 c3 = _mm256_set1_ps(0.0551904f);
  const __m256 c4 = _mm256_set1_ps(0.0098892f);
  const __m256 c5 = _mm256_set1_ps(0.0012398f);

  // Evaluate polynomial using Horner scheme
  __m256 polynomial = _mm256_add_ps(
      c0,
      _mm256_mul_ps(
          x_frac,
          _mm256_add_ps(
              c1, _mm256_mul_ps(
                      x_frac,
                      _mm256_add_ps(
                          c2, _mm256_mul_ps(
                                  x_frac,
                                  _mm256_add_ps(
                                      c3, _mm256_mul_ps(
                                              x_frac,
                                              _mm256_add_ps(
                                                  c4, _mm256_mul_ps(
                                                          x_frac, c5))))))))));

  // Convert int to float bits for 2^int (bit trick)
  __m256i exp_int = _mm256_cvttps_epi32(x_int);
  exp_int = _mm256_add_epi32(exp_int, _mm256_set1_epi32(127));
  exp_int = _mm256_slli_epi32(exp_int, 23);
  __m256 pow2_int = _mm256_castsi256_ps(exp_int);

  // Combine 2^int * 2^frac = 2^(int+frac) = e^x
  return _mm256_mul_ps(pow2_int, polynomial);
}

// Implementation of log_ps using polynomial approximation
__m256 log_ps(__m256 x) {
  // Handle special cases: x <= 0
  __m256 zero = _mm256_setzero_ps();
  __m256 mask_zero_or_neg = _mm256_cmp_ps(x, zero, _CMP_LE_OQ);
  __m256 nan_value = _mm256_set1_ps(std::numeric_limits<float>::quiet_NaN());

  // Extract exponent and mantissa using bit manipulation
  __m256i x_bits = _mm256_castps_si256(x);
  __m256i exp_bits = _mm256_srli_epi32(
      _mm256_and_si256(x_bits, _mm256_set1_epi32(0x7F800000)), 23);
  __m256 exponent =
      _mm256_cvtepi32_ps(_mm256_sub_epi32(exp_bits, _mm256_set1_epi32(127)));

  // Calculate mantissa in [1, 2) range
  __m256 mantissa = _mm256_or_ps(
      _mm256_and_ps(x, _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF))),
      _mm256_set1_ps(1.0f));

  // Polynomial approximation for log(1+x) where x is mantissa-1
  __m256 z = _mm256_sub_ps(mantissa, _mm256_set1_ps(1.0f));

  // Coefficients for log(1+z) minimax approximation
  // log(1+z) ≈ z - z^2/2 + z^3/3 - z^4/4 + z^5/5 - z^6/6 + z^7/7
  const __m256 c1 = _mm256_set1_ps(1.0f);
  const __m256 c2 = _mm256_set1_ps(-0.5f);
  const __m256 c3 = _mm256_set1_ps(0.333333333f);
  const __m256 c4 = _mm256_set1_ps(-0.25f);
  const __m256 c5 = _mm256_set1_ps(0.2f);
  const __m256 c6 = _mm256_set1_ps(-0.166666667f);
  const __m256 c7 = _mm256_set1_ps(0.142857143f);

  // z, z^2, z^3, ...
  __m256 z2 = _mm256_mul_ps(z, z);
  __m256 z3 = _mm256_mul_ps(z2, z);
  __m256 z4 = _mm256_mul_ps(z3, z);
  __m256 z5 = _mm256_mul_ps(z4, z);
  __m256 z6 = _mm256_mul_ps(z5, z);
  __m256 z7 = _mm256_mul_ps(z6, z);

  // Evaluate polynomial
  __m256 log_mantissa = _mm256_add_ps(
      _mm256_mul_ps(c1, z),
      _mm256_add_ps(
          _mm256_mul_ps(c2, z2),
          _mm256_add_ps(
              _mm256_mul_ps(c3, z3),
              _mm256_add_ps(
                  _mm256_mul_ps(c4, z4),
                  _mm256_add_ps(_mm256_mul_ps(c5, z5),
                                _mm256_add_ps(_mm256_mul_ps(c6, z6),
                                              _mm256_mul_ps(c7, z7)))))));

  // log(x) = log(2^exp * mantissa) = exp*log(2) + log(mantissa)
  const __m256 ln2 = _mm256_set1_ps(0.693147180559945f);
  __m256 result = _mm256_add_ps(_mm256_mul_ps(exponent, ln2), log_mantissa);

  // Apply mask for zero or negative input
  return _mm256_blendv_ps(result, nan_value, mask_zero_or_neg);
}

} // namespace simd
} // namespace num

namespace opt {
} // namespace opt
} // namespace alo
} // namespace engine