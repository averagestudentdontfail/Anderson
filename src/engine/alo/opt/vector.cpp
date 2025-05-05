#include "vector.h"
#include "simd.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include <sleef.h>
#include <vector>

// Platform-specific CPU detection
#ifdef _WIN32
#include <intrin.h>
inline void get_cpuid(int level, int output[4]) { __cpuid(output, level); }
#else
#include <cpuid.h>
inline void get_cpuid(int level, int output[4]) {
  unsigned int *regs = reinterpret_cast<unsigned int *>(output);
  __get_cpuid(level, &regs[0], &regs[1], &regs[2], &regs[3]);
}
#endif

namespace engine {
namespace alo {
namespace opt {

SIMDSupport detectSIMDSupport() {
  int info[4];

  // Check for SSE2
  get_cpuid(1, info);
  if (!(info[3] & (1 << 26)))
    return NONE;

  // Check for AVX
  if (!(info[2] & (1 << 28)))
    return SSE2;

  // Check for AVX2
  get_cpuid(7, info);
  if (!(info[1] & (1 << 5)))
    return AVX;

  // Check for AVX512F
  if (!(info[1] & (1 << 16)))
    return AVX2;

  return AVX512;
}

// Data conversion utilities
std::vector<float>
VectorDouble::convertToSingle(const std::vector<double> &input) {
  std::vector<float> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = static_cast<float>(input[i]);
  }
  return result;
}

std::vector<double>
VectorSingle::convertToDouble(const std::vector<float> &input) {
  std::vector<double> result(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    result[i] = static_cast<double>(input[i]);
  }
  return result;
}

inline bool shouldUseSimd(size_t size) {
  return size >= 32; // Only use SIMD for operations with size >= 32
}

void VectorDouble::EuropeanPut(const double *S, const double *K,
                               const double *r, const double *q,
                               const double *vol, const double *T,
                               double *result, size_t size) {
  // Process in steps of 4 (AVX2 double precision)
  size_t i = 0;
  for (; i + 3 < size; i += 4) {
    // Load vectors
    __m256d S_vec = SimdOperationDouble::load(S + i);
    __m256d K_vec = SimdOperationDouble::load(K + i);
    __m256d r_vec = SimdOperationDouble::load(r + i);
    __m256d q_vec = SimdOperationDouble::load(q + i);
    __m256d vol_vec = SimdOperationDouble::load(vol + i);
    __m256d T_vec = SimdOperationDouble::load(T + i);

    // Calculate option prices
    __m256d prices = SimdOperationDouble::EuropeanPut(S_vec, K_vec, r_vec,
                                                      q_vec, vol_vec, T_vec);

    // Store results
    SimdOperationDouble::store(result + i, prices);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    // Use scalar calculation
    double d1 =
        (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) /
        (vol[i] * std::sqrt(T[i]));
    double d2 = d1 - vol[i] * std::sqrt(T[i]);

    double Nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
    double Nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));

    result[i] = K[i] * std::exp(-r[i] * T[i]) * Nd2 -
                S[i] * std::exp(-q[i] * T[i]) * Nd1;
  }
}

void VectorDouble::EuropeanCall(const double *S, const double *K,
                                const double *r, const double *q,
                                const double *vol, const double *T,
                                double *result, size_t size) {
  // Process in steps of 4 (AVX2 double precision)
  size_t i = 0;
  for (; i + 3 < size; i += 4) {
    // Load vectors
    __m256d S_vec = SimdOperationDouble::load(S + i);
    __m256d K_vec = SimdOperationDouble::load(K + i);
    __m256d r_vec = SimdOperationDouble::load(r + i);
    __m256d q_vec = SimdOperationDouble::load(q + i);
    __m256d vol_vec = SimdOperationDouble::load(vol + i);
    __m256d T_vec = SimdOperationDouble::load(T + i);

    // Calculate option prices
    __m256d prices = SimdOperationDouble::EuropeanCall(S_vec, K_vec, r_vec,
                                                       q_vec, vol_vec, T_vec);

    // Store results
    SimdOperationDouble::store(result + i, prices);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    // Use scalar calculation
    double d1 =
        (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) /
        (vol[i] * std::sqrt(T[i]));
    double d2 = d1 - vol[i] * std::sqrt(T[i]);

    double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double Nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));

    result[i] = S[i] * std::exp(-q[i] * T[i]) * Nd1 -
                K[i] * std::exp(-r[i] * T[i]) * Nd2;
  }
}

void VectorDouble::EuropeanPutGreek(const double *S, const double *K,
                                    const double *r, const double *q,
                                    const double *vol, const double *T,
                                    GreekDouble *results, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    // Handle degenerate cases
    if (vol[i] <= 0.0 || T[i] <= 0.0) {
      results[i].price = std::max(0.0, K[i] - S[i]);
      results[i].delta = (S[i] < K[i]) ? -1.0 : 0.0;
      results[i].gamma = 0.0;
      results[i].vega = 0.0;
      results[i].theta = 0.0;
      results[i].rho = 0.0;
      continue;
    }

    // Calculate d1 and d2
    double sqrt_T = std::sqrt(T[i]);
    double d1 =
        (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) /
        (vol[i] * sqrt_T);
    double d2 = d1 - vol[i] * sqrt_T;

    // Calculate discount factors
    double dr = std::exp(-r[i] * T[i]);
    double dq = std::exp(-q[i] * T[i]);

    // Calculate N(-d1) and N(-d2)
    double Nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
    double Nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));

    // Calculate option price
    results[i].price = K[i] * dr * Nd2 - S[i] * dq * Nd1;

    // Calculate PDF for Greeks
    double pdf_d1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);

    // Delta = -e^(-q*T) * N(-d1)
    results[i].delta = -dq * Nd1;

    // Gamma = e^(-q*T) * PDF(d1) / (S * vol * sqrt(T))
    results[i].gamma = dq * pdf_d1 / (S[i] * vol[i] * sqrt_T);

    // Vega = S * e^(-q*T) * PDF(d1) * sqrt(T) / 100
    results[i].vega = 0.01 * S[i] * dq * pdf_d1 * sqrt_T;

    // Theta calculation (daily)
    double term1 = -S[i] * dq * pdf_d1 * vol[i] / (2.0 * sqrt_T);
    double term2 = q[i] * S[i] * dq * Nd1;
    double term3 = -r[i] * K[i] * dr * Nd2;
    results[i].theta = (term1 + term2 + term3) / 365.0;

    // Rho = -K * T * e^(-r*T) * N(-d2) / 100
    results[i].rho = -0.01 * K[i] * T[i] * dr * Nd2;
  }
}

void VectorDouble::EuropeanCallGreek(const double *S, const double *K,
                                     const double *r, const double *q,
                                     const double *vol, const double *T,
                                     GreekDouble *results, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    // Handle degenerate cases
    if (vol[i] <= 0.0 || T[i] <= 0.0) {
      results[i].price = std::max(0.0, S[i] - K[i]);
      results[i].delta = (S[i] > K[i]) ? 1.0 : 0.0;
      results[i].gamma = 0.0;
      results[i].vega = 0.0;
      results[i].theta = 0.0;
      results[i].rho = 0.0;
      continue;
    }

    // Calculate d1 and d2
    double sqrt_T = std::sqrt(T[i]);
    double d1 =
        (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) /
        (vol[i] * sqrt_T);
    double d2 = d1 - vol[i] * sqrt_T;

    // Calculate discount factors
    double dr = std::exp(-r[i] * T[i]);
    double dq = std::exp(-q[i] * T[i]);

    // Calculate N(d1) and N(d2)
    double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double Nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));

    // Calculate option price
    results[i].price = S[i] * dq * Nd1 - K[i] * dr * Nd2;

    // Calculate PDF for Greeks
    double pdf_d1 = std::exp(-0.5 * d1 * d1) / std::sqrt(2.0 * M_PI);

    // Delta = e^(-q*T) * N(d1)
    results[i].delta = dq * Nd1;

    // Gamma = e^(-q*T) * PDF(d1) / (S * vol * sqrt(T))
    results[i].gamma = dq * pdf_d1 / (S[i] * vol[i] * sqrt_T);

    // Vega = S * e^(-q*T) * PDF(d1) * sqrt(T) / 100
    results[i].vega = 0.01 * S[i] * dq * pdf_d1 * sqrt_T;

    // Theta calculation (daily)
    double term1 = -S[i] * dq * pdf_d1 * vol[i] / (2.0 * sqrt_T);
    double term2 = -q[i] * S[i] * dq * Nd1;
    double term3 = r[i] * K[i] * dr * Nd2;
    results[i].theta = (term1 + term2 + term3) / 365.0;

    // Rho = K * T * e^(-r*T) * N(d2) / 100
    results[i].rho = 0.01 * K[i] * T[i] * dr * Nd2;
  }
}

void VectorDouble::AmericanPut(const double *S, const double *K,
                               const double *r, const double *q,
                               const double *vol, const double *T,
                               double *results, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    // First calculate European price
    double bs_price = 0.0;

    // Use Black-Scholes formula directly
    if (vol[i] <= 0.0 || T[i] <= 0.0) {
      bs_price = std::max(0.0, K[i] - S[i]);
    } else {
      double d1 = (std::log(S[i] / K[i]) +
                   (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) /
                  (vol[i] * std::sqrt(T[i]));
      double d2 = d1 - vol[i] * std::sqrt(T[i]);

      double Nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
      double Nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));

      bs_price = K[i] * std::exp(-r[i] * T[i]) * Nd2 -
                 S[i] * std::exp(-q[i] * T[i]) * Nd1;
    }

    // Check if early exercise might be valuable
    double premium = 0.0;
    if (r[i] > q[i]) {
      // Calculate approximate critical price
      double num = 1.0 - std::exp(-r[i] * T[i]);
      double denom = 1.0 - std::exp(-q[i] * T[i]);
      if (denom == 0.0)
        denom = 1.0;

      double b_star = K[i] * num / denom;

      // If S <= b*, calculate approximate early exercise premium
      if (S[i] <= b_star) {
        double power = 2.0 * r[i] / (vol[i] * vol[i]);
        double ratio_pow = std::pow(S[i] / b_star, power);
        premium = std::max(0.0, K[i] - S[i] - (K[i] - b_star) * ratio_pow);
      }
    }

    results[i] = bs_price + premium;
  }
}

void VectorDouble::AmericanCall(const double *S, const double *K,
                                const double *r, const double *q,
                                const double *vol, const double *T,
                                double *results, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    // For zero dividend, American call = European call
    if (q[i] <= 0.0) {
      // European call formula
      if (vol[i] <= 0.0 || T[i] <= 0.0) {
        results[i] = std::max(0.0, S[i] - K[i]);
        continue;
      }

      double d1 = (std::log(S[i] / K[i]) +
                   (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) /
                  (vol[i] * std::sqrt(T[i]));
      double d2 = d1 - vol[i] * std::sqrt(T[i]);

      double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
      double Nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));

      results[i] = S[i] * std::exp(-q[i] * T[i]) * Nd1 -
                   K[i] * std::exp(-r[i] * T[i]) * Nd2;
    } else {
      // Calculate the critical price
      double q1 =
          0.5 * (-(r[i] - q[i]) / (vol[i] * vol[i]) +
                 std::sqrt(std::pow((r[i] - q[i]) / (vol[i] * vol[i]), 2.0) +
                           8.0 * r[i] / (vol[i] * vol[i])));

      double critical_price = K[i] / (1.0 - 1.0 / q1);

      // Check if early exercise is optimal
      if (S[i] >= critical_price) {
        results[i] = S[i] - K[i]; // Intrinsic value
      } else {
        // Calculate European price
        double d1 = (std::log(S[i] / K[i]) +
                     (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) /
                    (vol[i] * std::sqrt(T[i]));
        double d2 = d1 - vol[i] * std::sqrt(T[i]);

        double Nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
        double Nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));

        double euro_price = S[i] * std::exp(-q[i] * T[i]) * Nd1 -
                            K[i] * std::exp(-r[i] * T[i]) * Nd2;

        // Add early exercise premium
        double ratio = std::pow(S[i] / critical_price, q1);
        double premium = (critical_price - K[i]) * (1.0 - ratio);

        results[i] = euro_price + premium;
      }
    }
  }
}

void VectorSingle::EuropeanPut(const float *S, const float *K, const float *r,
                               const float *q, const float *vol, const float *T,
                               float *result, size_t size) {
  // Process in steps of 8 (AVX2 single precision)
  size_t i = 0;
  for (; i + 7 < size; i += 8) {
    // Load vectors
    __m256 S_vec = SimdOperationSingle::load(S + i);
    __m256 K_vec = SimdOperationSingle::load(K + i);
    __m256 r_vec = SimdOperationSingle::load(r + i);
    __m256 q_vec = SimdOperationSingle::load(q + i);
    __m256 vol_vec = SimdOperationSingle::load(vol + i);
    __m256 T_vec = SimdOperationSingle::load(T + i);

    // Calculate option prices
    __m256 prices = SimdOperationSingle::EuropeanPut(S_vec, K_vec, r_vec, q_vec,
                                                     vol_vec, T_vec);

    // Store results
    SimdOperationSingle::store(result + i, prices);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    // Use scalar calculation
    float d1 = (std::log(S[i] / K[i]) +
                (r[i] - q[i] + 0.5f * vol[i] * vol[i]) * T[i]) /
               (vol[i] * std::sqrt(T[i]));
    float d2 = d1 - vol[i] * std::sqrt(T[i]);

    float Nd1 = 0.5f * (1.0f + num::fast_erf(-d1 / 1.414213562f));
    float Nd2 = 0.5f * (1.0f + num::fast_erf(-d2 / 1.414213562f));

    result[i] = K[i] * std::exp(-r[i] * T[i]) * Nd2 -
                S[i] * std::exp(-q[i] * T[i]) * Nd1;
  }
}

void VectorSingle::EuropeanCall(const float *S, const float *K, const float *r,
                                const float *q, const float *vol,
                                const float *T, float *result, size_t size) {
  // Process in steps of 8 (AVX2 single precision)
  size_t i = 0;
  for (; i + 7 < size; i += 8) {
    // Load vectors
    __m256 S_vec = SimdOperationSingle::load(S + i);
    __m256 K_vec = SimdOperationSingle::load(K + i);
    __m256 r_vec = SimdOperationSingle::load(r + i);
    __m256 q_vec = SimdOperationSingle::load(q + i);
    __m256 vol_vec = SimdOperationSingle::load(vol + i);
    __m256 T_vec = SimdOperationSingle::load(T + i);

    // Calculate option prices
    __m256 prices = SimdOperationSingle::EuropeanCall(S_vec, K_vec, r_vec,
                                                      q_vec, vol_vec, T_vec);

    // Store results
    SimdOperationSingle::store(result + i, prices);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    // Use scalar calculation
    float d1 = (std::log(S[i] / K[i]) +
                (r[i] - q[i] + 0.5f * vol[i] * vol[i]) * T[i]) /
               (vol[i] * std::sqrt(T[i]));
    float d2 = d1 - vol[i] * std::sqrt(T[i]);

    float Nd1 = 0.5f * (1.0f + num::fast_erf(d1 / 1.414213562f));
    float Nd2 = 0.5f * (1.0f + num::fast_erf(d2 / 1.414213562f));

    result[i] = S[i] * std::exp(-q[i] * T[i]) * Nd1 -
                K[i] * std::exp(-r[i] * T[i]) * Nd2;
  }
}

void VectorSingle::EuropeanPutGreek(const float *S, const float *K,
                                    const float *r, const float *q,
                                    const float *vol, const float *T,
                                    GreekSingle *results, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    // Handle degenerate cases
    if (vol[i] <= 0.0f || T[i] <= 0.0f) {
      results[i].price = std::max(0.0f, K[i] - S[i]);
      results[i].delta = (S[i] < K[i]) ? -1.0f : 0.0f;
      results[i].gamma = 0.0f;
      results[i].vega = 0.0f;
      results[i].theta = 0.0f;
      results[i].rho = 0.0f;
      continue;
    }

    // Calculate d1 and d2
    float sqrt_T = std::sqrt(T[i]);
    float d1 = (std::log(S[i] / K[i]) +
                (r[i] - q[i] + 0.5f * vol[i] * vol[i]) * T[i]) /
               (vol[i] * sqrt_T);
    float d2 = d1 - vol[i] * sqrt_T;

    // Calculate discount factors
    float dr = std::exp(-r[i] * T[i]);
    float dq = std::exp(-q[i] * T[i]);

    // Calculate N(-d1) and N(-d2)
    float Nd1 = 0.5f * (1.0f + num::fast_erf(-d1 / 1.414213562f));
    float Nd2 = 0.5f * (1.0f + num::fast_erf(-d2 / 1.414213562f));

    // Calculate option price
    results[i].price = K[i] * dr * Nd2 - S[i] * dq * Nd1;

    // Calculate PDF for Greeks
    float pdf_d1 = std::exp(-0.5f * d1 * d1) / 2.506628275f; // 1/sqrt(2π)

    // Delta = -e^(-q*T) * N(-d1)
    results[i].delta = -dq * Nd1;

    // Gamma = e^(-q*T) * PDF(d1) / (S * vol * sqrt(T))
    results[i].gamma = dq * pdf_d1 / (S[i] * vol[i] * sqrt_T);

    // Vega = S * e^(-q*T) * PDF(d1) * sqrt(T) / 100
    results[i].vega = 0.01f * S[i] * dq * pdf_d1 * sqrt_T;

    // Theta calculation (daily)
    float term1 = -S[i] * dq * pdf_d1 * vol[i] / (2.0f * sqrt_T);
    float term2 = q[i] * S[i] * dq * Nd1;
    float term3 = -r[i] * K[i] * dr * Nd2;
    results[i].theta = (term1 + term2 + term3) / 365.0f;

    // Rho = -K * T * e^(-r*T) * N(-d2) / 100
    results[i].rho = -0.01f * K[i] * T[i] * dr * Nd2;
  }
}

void VectorSingle::EuropeanCallGreek(const float *S, const float *K,
                                     const float *r, const float *q,
                                     const float *vol, const float *T,
                                     GreekSingle *results, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    // Handle degenerate cases
    if (vol[i] <= 0.0f || T[i] <= 0.0f) {
      results[i].price = std::max(0.0f, S[i] - K[i]);
      results[i].delta = (S[i] > K[i]) ? 1.0f : 0.0f;
      results[i].gamma = 0.0f;
      results[i].vega = 0.0f;
      results[i].theta = 0.0f;
      results[i].rho = 0.0f;
      continue;
    }

    // Calculate d1 and d2
    float sqrt_T = std::sqrt(T[i]);
    float d1 = (std::log(S[i] / K[i]) +
                (r[i] - q[i] + 0.5f * vol[i] * vol[i]) * T[i]) /
               (vol[i] * sqrt_T);
    float d2 = d1 - vol[i] * sqrt_T;

    // Calculate discount factors
    float dr = std::exp(-r[i] * T[i]);
    float dq = std::exp(-q[i] * T[i]);

    // Calculate N(d1) and N(d2)
    float Nd1 = 0.5f * (1.0f + num::fast_erf(d1 / 1.414213562f));
    float Nd2 = 0.5f * (1.0f + num::fast_erf(d2 / 1.414213562f));

    // Calculate option price
    results[i].price = S[i] * dq * Nd1 - K[i] * dr * Nd2;

    // Calculate PDF for Greeks
    float pdf_d1 = std::exp(-0.5f * d1 * d1) / 2.506628275f; // 1/sqrt(2π)

    // Delta = e^(-q*T) * N(d1)
    results[i].delta = dq * Nd1;

    // Gamma = e^(-q*T) * PDF(d1) / (S * vol * sqrt(T))
    results[i].gamma = dq * pdf_d1 / (S[i] * vol[i] * sqrt_T);

    // Vega = S * e^(-q*T) * PDF(d1) * sqrt(T) / 100
    results[i].vega = 0.01f * S[i] * dq * pdf_d1 * sqrt_T;

    // Theta calculation (daily)
    float term1 = -S[i] * dq * pdf_d1 * vol[i] / (2.0f * sqrt_T);
    float term2 = -q[i] * S[i] * dq * Nd1;
    float term3 = r[i] * K[i] * dr * Nd2;
    results[i].theta = (term1 + term2 + term3) / 365.0f;

    // Rho = K * T * e^(-r*T) * N(d2) / 100
    results[i].rho = 0.01f * K[i] * T[i] * dr * Nd2;
  }
}

void VectorSingle::AmericanPut(const float *S, const float *K, const float *r,
                               const float *q, const float *vol, const float *T,
                               float *results, size_t size) {
  // Process in steps of 8 (AVX2 single precision)
  size_t i = 0;
  for (; i + 7 < size; i += 8) {
    // Load vectors
    __m256 S_vec = SimdOperationSingle::load(S + i);
    __m256 K_vec = SimdOperationSingle::load(K + i);
    __m256 r_vec = SimdOperationSingle::load(r + i);
    __m256 q_vec = SimdOperationSingle::load(q + i);
    __m256 vol_vec = SimdOperationSingle::load(vol + i);
    __m256 T_vec = SimdOperationSingle::load(T + i);

    // Calculate option prices
    __m256 prices = SimdOperationSingle::AmericanPut(S_vec, K_vec, r_vec, q_vec,
                                                     vol_vec, T_vec);

    // Store results
    SimdOperationSingle::store(results + i, prices);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    // First calculate European price
    float bs_price = 0.0f;

    // Use Black-Scholes formula directly
    if (vol[i] <= 0.0f || T[i] <= 0.0f) {
      bs_price = std::max(0.0f, K[i] - S[i]);
    } else {
      float d1 = (std::log(S[i] / K[i]) +
                  (r[i] - q[i] + 0.5f * vol[i] * vol[i]) * T[i]) /
                 (vol[i] * std::sqrt(T[i]));
      float d2 = d1 - vol[i] * std::sqrt(T[i]);

      float Nd1 = 0.5f * (1.0f + num::fast_erf(-d1 / 1.414213562f));
      float Nd2 = 0.5f * (1.0f + num::fast_erf(-d2 / 1.414213562f));

      bs_price = K[i] * std::exp(-r[i] * T[i]) * Nd2 -
                 S[i] * std::exp(-q[i] * T[i]) * Nd1;
    }

    // Check if early exercise is potentially valuable
    float premium = 0.0f;
    if (r[i] > q[i]) {
      // Calculate critical price using the quadratic approximation
      float b = K[i] * (1.0f - std::exp(-r[i] * T[i]));
      if (q[i] > 0.0f) {
        b /= (1.0f - std::exp(-q[i] * T[i]));
      }

      // If S is below critical price, apply approximation
      if (S[i] <= b) {
        float power = 2.0f * r[i] / (vol[i] * vol[i]);
        float ratio = std::pow(S[i] / b, power);
        premium = std::max(0.0f, K[i] - S[i] - (K[i] - b) * ratio);
      }
    }

    results[i] = bs_price + premium;
  }
}

void VectorSingle::AmericanCall(const float *S, const float *K, const float *r,
                                const float *q, const float *vol,
                                const float *T, float *results, size_t size) {
  // Process in steps of 8 (AVX2 single precision)
  size_t i = 0;
  for (; i + 7 < size; i += 8) {
    // Load vectors
    __m256 S_vec = SimdOperationSingle::load(S + i);
    __m256 K_vec = SimdOperationSingle::load(K + i);
    __m256 r_vec = SimdOperationSingle::load(r + i);
    __m256 q_vec = SimdOperationSingle::load(q + i);
    __m256 vol_vec = SimdOperationSingle::load(vol + i);
    __m256 T_vec = SimdOperationSingle::load(T + i);

    // Calculate option prices
    __m256 prices = SimdOperationSingle::AmericanCall(S_vec, K_vec, r_vec,
                                                      q_vec, vol_vec, T_vec);

    // Store results
    SimdOperationSingle::store(results + i, prices);
  }

  // Handle remaining elements with scalar calculation
  for (; i < size; ++i) {
    // For zero dividend, American call = European call
    if (q[i] <= 0.0f) {
      // European call formula
      if (vol[i] <= 0.0f || T[i] <= 0.0f) {
        results[i] = std::max(0.0f, S[i] - K[i]);
        continue;
      }

      float d1 = (std::log(S[i] / K[i]) +
                  (r[i] - q[i] + 0.5f * vol[i] * vol[i]) * T[i]) /
                 (vol[i] * std::sqrt(T[i]));
      float d2 = d1 - vol[i] * std::sqrt(T[i]);

      float Nd1 = 0.5f * (1.0f + num::fast_erf(d1 / 1.414213562f));
      float Nd2 = 0.5f * (1.0f + num::fast_erf(d2 / 1.414213562f));

      results[i] = S[i] * std::exp(-q[i] * T[i]) * Nd1 -
                   K[i] * std::exp(-r[i] * T[i]) * Nd2;
    } else {
      // Calculate the critical price
      float q1 =
          0.5f * (-(r[i] - q[i]) / (vol[i] * vol[i]) +
                  std::sqrt(std::pow((r[i] - q[i]) / (vol[i] * vol[i]), 2.0f) +
                            8.0f * r[i] / (vol[i] * vol[i])));

      float critical_price = K[i] / (1.0f - 1.0f / q1);

      // Check if early exercise is optimal
      if (S[i] >= critical_price) {
        results[i] = S[i] - K[i]; // Intrinsic value
      } else {
        // Calculate European price
        float d1 = (std::log(S[i] / K[i]) +
                    (r[i] - q[i] + 0.5f * vol[i] * vol[i]) * T[i]) /
                   (vol[i] * std::sqrt(T[i]));
        float d2 = d1 - vol[i] * std::sqrt(T[i]);

        float Nd1 = 0.5f * (1.0f + num::fast_erf(d1 / 1.414213562f));
        float Nd2 = 0.5f * (1.0f + num::fast_erf(d2 / 1.414213562f));

        float euro_price = S[i] * std::exp(-q[i] * T[i]) * Nd1 -
                           K[i] * std::exp(-r[i] * T[i]) * Nd2;

        // Add early exercise premium
        float ratio = std::pow(S[i] / critical_price, q1);
        float premium = (critical_price - K[i]) * (1.0f - ratio);

        results[i] = euro_price + premium;
      }
    }
  }
}

void VectorMath::exp(const double *x, double *result, size_t size) {
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
    __m256d res = Sleef_expd4_u10avx2(vec);
    _mm256_storeu_pd(result + i, res);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    result[i] = std::exp(x[i]);
  }
}

void VectorMath::log(const double *x, double *result, size_t size) {
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
    __m256d res = Sleef_logd4_u10avx2(vec);
    _mm256_storeu_pd(result + i, res);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    result[i] = std::log(x[i]);
  }
}

void VectorMath::sqrt(const double *x, double *result, size_t size) {
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

void VectorMath::erf(const double *x, double *result, size_t size) {
  // Use scalar operations for small data sizes
  if (!shouldUseSimd(size)) {
    for (size_t i = 0; i < size; ++i) {
      result[i] = std::erf(x[i]);
    }
    return;
  }

  // Process in chunks of 4 using SLEEF's native erf function
  size_t i = 0;
  for (; i + 3 < size; i += 4) {
    __m256d vec = _mm256_loadu_pd(x + i);
    __m256d res = Sleef_erfd4_u10avx2(vec);
    _mm256_storeu_pd(result + i, res);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    result[i] = std::erf(x[i]);
  }
}

void VectorMath::normalCDF(const double *x, double *result, size_t size) {
  // Use scalar operations for small data sizes
  if (!shouldUseSimd(size)) {
    for (size_t i = 0; i < size; ++i) {
      // Choose the most numerically stable formula based on the value of x
      if (x[i] < -8.0) {
        // For large negative x, use erfc-based formula
        result[i] = 0.5 * std::erfc(-x[i] / std::sqrt(2.0));
      } else if (x[i] > 8.0) {
        // For large positive x, result is very close to 1
        result[i] = 1.0 - 0.5 * std::erfc(x[i] / std::sqrt(2.0));
      } else {
        // For moderate values, use erf-based formula
        result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
      }
    }
    return;
  }

  // Process in chunks of 4 using SLEEF with appropriate formula
  size_t i = 0;
  for (; i + 3 < size; i += 4) {
    __m256d x_vec = _mm256_loadu_pd(x + i);

    // Create masks for extreme values
    __m256d large_neg_mask =
        _mm256_cmp_pd(x_vec, _mm256_set1_pd(-8.0), _CMP_LT_OS);
    __m256d large_pos_mask =
        _mm256_cmp_pd(x_vec, _mm256_set1_pd(8.0), _CMP_GT_OS);

    // Scale for normal CDF: x/sqrt(2)
    __m256d scaled_x = _mm256_div_pd(x_vec, _mm256_set1_pd(M_SQRT2));

    // For normal range values (-8 to 8), use erf
    __m256d erf_scaled = Sleef_erfd4_u10avx2(scaled_x);
    __m256d one = _mm256_set1_pd(1.0);
    __m256d half = _mm256_set1_pd(0.5);
    __m256d normal_result = _mm256_mul_pd(_mm256_add_pd(one, erf_scaled), half);

    // For extreme values, calculate using erfc for better numerical stability
    __m256d neg_scaled_x = _mm256_sub_pd(_mm256_setzero_pd(), scaled_x);

    // For negative large x: N(x) = 0.5*erfc(-x/sqrt(2))
    __m256d large_neg_result =
        _mm256_mul_pd(half, Sleef_erfcd4_u15avx2(neg_scaled_x));

    // For positive large x: N(x) = 1 - 0.5*erfc(x/sqrt(2))
    __m256d large_pos_result =
        _mm256_sub_pd(one, _mm256_mul_pd(half, Sleef_erfcd4_u15avx2(scaled_x)));

    // Blend results based on extreme value masks
    __m256d result_vec = normal_result;
    result_vec = _mm256_blendv_pd(result_vec, large_neg_result, large_neg_mask);
    result_vec = _mm256_blendv_pd(result_vec, large_pos_result, large_pos_mask);

    // Store result
    _mm256_storeu_pd(result + i, result_vec);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    if (x[i] < -8.0) {
      result[i] = 0.5 * std::erfc(-x[i] / std::sqrt(2.0));
    } else if (x[i] > 8.0) {
      result[i] = 1.0 - 0.5 * std::erfc(x[i] / std::sqrt(2.0));
    } else {
      result[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
    }
  }
}

void VectorMath::normalPDF(const double *x, double *result, size_t size) {
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
    __m256d exp_term = Sleef_expd4_u10avx2(scaled);

    // Multiply by 1/sqrt(2π)
    __m256d pdf = _mm256_mul_pd(exp_term, _mm256_set1_pd(INV_SQRT_2PI));

    _mm256_storeu_pd(result + i, pdf);
  }

  // Handle remaining elements
  for (; i < size; ++i) {
    result[i] = INV_SQRT_2PI * std::exp(-0.5 * x[i] * x[i]);
  }
}

void VectorMath::multiply(const double *a, const double *b, double *result,
                          size_t size) {
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

void VectorMath::add(const double *a, const double *b, double *result,
                     size_t size) {
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

void VectorMath::subtract(const double *a, const double *b, double *result,
                          size_t size) {
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

void VectorMath::divide(const double *a, const double *b, double *result,
                        size_t size) {
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

} // namespace opt
} // namespace alo
} // namespace engine