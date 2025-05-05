#include "european.h"
#include "../opt/simd.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace engine {
namespace alo {
namespace mod {

double EuropeanOption::d1(double S, double K, double r, double q, double vol,
                          double T) {
  if (vol <= 0.0 || T <= 0.0) {
    throw std::domain_error("Invalid parameters: vol and T must be positive");
  }

  return (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) /
         (vol * std::sqrt(T));
}

double EuropeanOption::d2(double d1, double vol, double T) {
  return d1 - vol * std::sqrt(T);
}

double EuropeanOption::normalCDF(double x) {
  return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double EuropeanOption::normalPDF(double x) {
  return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

double EuropeanPut::calculatePrice(double S, double K, double r, double q,
                                   double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return std::max(0.0, K - S);
  }

  // Calculate d1 and d2
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);

  // Calculate option price
  double Nd1 = normalCDF(-d1_val);
  double Nd2 = normalCDF(-d2_val);

  return K * std::exp(-r * T) * Nd2 - S * std::exp(-q * T) * Nd1;
}

double EuropeanPut::calculateDelta(double S, double K, double r, double q,
                                   double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return (S < K) ? -1.0 : 0.0;
  }

  // Calculate d1
  double d1_val = d1(S, K, r, q, vol, T);

  // Calculate delta
  return std::exp(-q * T) * (normalCDF(d1_val) - 1.0);
}

double EuropeanPut::calculateGamma(double S, double K, double r, double q,
                                   double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return 0.0;
  }

  // Calculate d1
  double d1_val = d1(S, K, r, q, vol, T);

  // Calculate gamma
  return std::exp(-q * T) * normalPDF(d1_val) / (S * vol * std::sqrt(T));
}

double EuropeanPut::calculateVega(double S, double K, double r, double q,
                                  double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return 0.0;
  }

  // Calculate d1
  double d1_val = d1(S, K, r, q, vol, T);

  // Calculate vega (as percentage of spot price)
  return 0.01 * S * std::exp(-q * T) * normalPDF(d1_val) * std::sqrt(T);
}

double EuropeanPut::calculateTheta(double S, double K, double r, double q,
                                   double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return 0.0;
  }

  // Calculate d1 and d2
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);

  // Calculate theta (per calendar day)
  double term1 =
      -S * std::exp(-q * T) * normalPDF(d1_val) * vol / (2.0 * std::sqrt(T));
  double term2 = q * S * std::exp(-q * T) * normalCDF(-d1_val);
  double term3 = -r * K * std::exp(-r * T) * normalCDF(-d2_val);

  return (term1 + term2 + term3) / 365.0;
}

double EuropeanPut::calculateRho(double S, double K, double r, double q,
                                 double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return (S < K) ? -K * T : 0.0;
  }

  // Calculate d2
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);

  // Calculate rho (as percentage)
  return -0.01 * K * T * std::exp(-r * T) * normalCDF(-d2_val);
}

std::vector<double>
EuropeanPut::batchCalculatePrice(double S, const std::vector<double> &strikes,
                                 double r, double q, double vol,
                                 double T) const {
  // Return empty vector for empty input
  if (strikes.empty()) {
    return {};
  }

  std::vector<double> results(strikes.size());
  const size_t n = strikes.size();

  // For small batches, use scalar computation
  if (n <= 8) {
    for (size_t i = 0; i < n; ++i) {
      results[i] = calculatePrice(S, strikes[i], r, q, vol, T);
    }
    return results;
  }

  // For larger batches, use SIMD with 4 options at a time
  size_t i = 0;

  // Process in groups of 4 using SIMD
  for (; i + 3 < n; i += 4) {
    std::array<double, 4> strike_array = {strikes[i], strikes[i + 1],
                                          strikes[i + 2], strikes[i + 3]};

    auto prices =
        calculatePrice4({S, S, S, S}, strike_array, {r, r, r, r}, {q, q, q, q},
                        {vol, vol, vol, vol}, {T, T, T, T});

    for (size_t j = 0; j < 4; ++j) {
      results[i + j] = prices[j];
    }
  }

  // Handle remaining options with scalar computation
  for (; i < n; ++i) {
    results[i] = calculatePrice(S, strikes[i], r, q, vol, T);
  }

  return results;
}

std::array<double, 4> EuropeanPut::calculatePrice4(
    const std::array<double, 4> &spots, const std::array<double, 4> &strikes,
    const std::array<double, 4> &rs, const std::array<double, 4> &qs,
    const std::array<double, 4> &vols, const std::array<double, 4> &Ts) const {

  // Initialize result array
  std::array<double, 4> results = {0.0, 0.0, 0.0, 0.0};

  // Check for degenerate cases where SIMD calculation would fail
  bool has_degenerate = false;
  for (size_t i = 0; i < 4; ++i) {
    if (vols[i] <= 0.0 || Ts[i] <= 0.0) {
      has_degenerate = true;
      break;
    }
  }

  // If we have degenerate cases, fall back to scalar calculation
  if (has_degenerate) {
    for (size_t i = 0; i < 4; ++i) {
      results[i] =
          calculatePrice(spots[i], strikes[i], rs[i], qs[i], vols[i], Ts[i]);
    }
    return results;
  }

  // Use SIMD operations for batch calculation
  __m256d S_vec = opt::SimdOps::load(spots);
  __m256d K_vec = opt::SimdOps::load(strikes);
  __m256d r_vec = opt::SimdOps::load(rs);
  __m256d q_vec = opt::SimdOps::load(qs);
  __m256d vol_vec = opt::SimdOps::load(vols);
  __m256d T_vec = opt::SimdOps::load(Ts);

  // Calculate Black-Scholes put prices
  __m256d prices =
      opt::SimdOps::bsPut(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);

  // Store results
  opt::SimdOps::store(results, prices);

  return results;
}

double EuropeanCall::calculatePrice(double S, double K, double r, double q,
                                    double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return std::max(0.0, S - K);
  }

  // Calculate d1 and d2
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);

  // Calculate option price
  double Nd1 = normalCDF(d1_val);
  double Nd2 = normalCDF(d2_val);

  return S * std::exp(-q * T) * Nd1 - K * std::exp(-r * T) * Nd2;
}

double EuropeanCall::calculateDelta(double S, double K, double r, double q,
                                    double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return (S > K) ? 1.0 : 0.0;
  }

  // Calculate d1
  double d1_val = d1(S, K, r, q, vol, T);

  // Calculate delta
  return std::exp(-q * T) * normalCDF(d1_val);
}

double EuropeanCall::calculateGamma(double S, double K, double r, double q,
                                    double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return 0.0;
  }

  // Calculate d1
  double d1_val = d1(S, K, r, q, vol, T);

  // Calculate gamma
  return std::exp(-q * T) * normalPDF(d1_val) / (S * vol * std::sqrt(T));
}

double EuropeanCall::calculateVega(double S, double K, double r, double q,
                                   double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return 0.0;
  }

  // Calculate d1
  double d1_val = d1(S, K, r, q, vol, T);

  // Calculate vega (as percentage of spot price)
  return 0.01 * S * std::exp(-q * T) * normalPDF(d1_val) * std::sqrt(T);
}

double EuropeanCall::calculateTheta(double S, double K, double r, double q,
                                    double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return 0.0;
  }

  // Calculate d1 and d2
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);

  // Calculate theta (per calendar day)
  double term1 =
      -S * std::exp(-q * T) * normalPDF(d1_val) * vol / (2.0 * std::sqrt(T));
  double term2 = -q * S * std::exp(-q * T) * normalCDF(d1_val);
  double term3 = r * K * std::exp(-r * T) * normalCDF(d2_val);

  return (term1 + term2 + term3) / 365.0;
}

double EuropeanCall::calculateRho(double S, double K, double r, double q,
                                  double vol, double T) const {
  // Handle special cases
  if (vol <= 0.0 || T <= 0.0) {
    return (S > K) ? K * T : 0.0;
  }

  // Calculate d2
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);

  // Calculate rho (as percentage)
  return 0.01 * K * T * std::exp(-r * T) * normalCDF(d2_val);
}

std::vector<double>
EuropeanCall::batchCalculatePrice(double S, const std::vector<double> &strikes,
                                  double r, double q, double vol,
                                  double T) const {
  // Return empty vector for empty input
  if (strikes.empty()) {
    return {};
  }

  std::vector<double> results(strikes.size());
  const size_t n = strikes.size();

  // For small batches, use scalar computation
  if (n <= 8) {
    for (size_t i = 0; i < n; ++i) {
      results[i] = calculatePrice(S, strikes[i], r, q, vol, T);
    }
    return results;
  }

  // For larger batches, use SIMD with 4 options at a time
  size_t i = 0;

  // Process in groups of 4 using SIMD
  for (; i + 3 < n; i += 4) {
    std::array<double, 4> strike_array = {strikes[i], strikes[i + 1],
                                          strikes[i + 2], strikes[i + 3]};

    auto prices =
        calculatePrice4({S, S, S, S}, strike_array, {r, r, r, r}, {q, q, q, q},
                        {vol, vol, vol, vol}, {T, T, T, T});

    for (size_t j = 0; j < 4; ++j) {
      results[i + j] = prices[j];
    }
  }

  // Handle remaining options with scalar computation
  for (; i < n; ++i) {
    results[i] = calculatePrice(S, strikes[i], r, q, vol, T);
  }

  return results;
}

std::array<double, 4> EuropeanCall::calculatePrice4(
    const std::array<double, 4> &spots, const std::array<double, 4> &strikes,
    const std::array<double, 4> &rs, const std::array<double, 4> &qs,
    const std::array<double, 4> &vols, const std::array<double, 4> &Ts) const {

  // Initialize result array
  std::array<double, 4> results = {0.0, 0.0, 0.0, 0.0};

  // Check for degenerate cases where SIMD calculation would fail
  bool has_degenerate = false;
  for (size_t i = 0; i < 4; ++i) {
    if (vols[i] <= 0.0 || Ts[i] <= 0.0) {
      has_degenerate = true;
      break;
    }
  }

  // If we have degenerate cases, fall back to scalar calculation
  if (has_degenerate) {
    for (size_t i = 0; i < 4; ++i) {
      results[i] =
          calculatePrice(spots[i], strikes[i], rs[i], qs[i], vols[i], Ts[i]);
    }
    return results;
  }

  // Use SIMD operations for batch calculation
  __m256d S_vec = opt::SimdOps::load(spots);
  __m256d K_vec = opt::SimdOps::load(strikes);
  __m256d r_vec = opt::SimdOps::load(rs);
  __m256d q_vec = opt::SimdOps::load(qs);
  __m256d vol_vec = opt::SimdOps::load(vols);
  __m256d T_vec = opt::SimdOps::load(Ts);

  // Calculate Black-Scholes call prices
  __m256d prices =
      opt::SimdOps::bsCall(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);

  // Store results
  opt::SimdOps::store(results, prices);

  return results;
}

double putCallParity(bool isPut, double price, double S, double K, double r,
                     double q, double T) {
  double pv_K = K * std::exp(-r * T); // Present value of strike
  double pv_S = S * std::exp(-q * T); // Present value of spot with dividends

  if (isPut) {
    // Convert call to put: P = C - S*exp(-q*T) + K*exp(-r*T)
    return price - pv_S + pv_K;
  } else {
    // Convert put to call: C = P + S*exp(-q*T) - K*exp(-r*T)
    return price + pv_S - pv_K;
  }
}

float EuropeanOptionFloat::d1(float S, float K, float r, float q, float vol,
                              float T) {
  if (vol <= 0.0f || T <= 0.0f) {
    throw std::domain_error("Invalid parameters: vol and T must be positive");
  }

  return (std::log(S / K) + (r - q + 0.5f * vol * vol) * T) /
         (vol * std::sqrt(T));
}

float EuropeanOptionFloat::d2(float d1, float vol, float T) {
  return d1 - vol * std::sqrt(T);
}

float EuropeanOptionFloat::normalCDF(float x) {
  return num::fast_normal_cdf(x);
}

float EuropeanOptionFloat::normalPDF(float x) {
  return num::fast_normal_pdf(x);
}

// EuropeanPutFloat implementation
float EuropeanPutFloat::calculatePrice(float S, float K, float r, float q,
                                       float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return std::max(0.0f, K - S);
  }

  // Calculate d1 and d2
  float d1_val = d1(S, K, r, q, vol, T);
  float d2_val = d2(d1_val, vol, T);

  // Calculate option price using fast approximation
  float Nd1 = normalCDF(-d1_val);
  float Nd2 = normalCDF(-d2_val);

  return K * std::exp(-r * T) * Nd2 - S * std::exp(-q * T) * Nd1;
}

float EuropeanPutFloat::calculateDelta(float S, float K, float r, float q,
                                       float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return (S < K) ? -1.0f : 0.0f;
  }

  // Calculate d1
  float d1_val = d1(S, K, r, q, vol, T);

  // Calculate delta
  return std::exp(-q * T) * (normalCDF(d1_val) - 1.0f);
}

float EuropeanPutFloat::calculateGamma(float S, float K, float r, float q,
                                       float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return 0.0f;
  }

  // Calculate d1
  float d1_val = d1(S, K, r, q, vol, T);

  // Calculate gamma
  return std::exp(-q * T) * normalPDF(d1_val) / (S * vol * std::sqrt(T));
}

float EuropeanPutFloat::calculateVega(float S, float K, float r, float q,
                                      float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return 0.0f;
  }

  // Calculate d1
  float d1_val = d1(S, K, r, q, vol, T);

  // Calculate vega (as percentage of spot price)
  return 0.01f * S * std::exp(-q * T) * normalPDF(d1_val) * std::sqrt(T);
}

float EuropeanPutFloat::calculateTheta(float S, float K, float r, float q,
                                       float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return 0.0f;
  }

  // Calculate d1 and d2
  float d1_val = d1(S, K, r, q, vol, T);
  float d2_val = d2(d1_val, vol, T);

  // Calculate theta (per calendar day)
  float term1 =
      -S * std::exp(-q * T) * normalPDF(d1_val) * vol / (2.0f * std::sqrt(T));
  float term2 = q * S * std::exp(-q * T) * normalCDF(-d1_val);
  float term3 = -r * K * std::exp(-r * T) * normalCDF(-d2_val);

  return (term1 + term2 + term3) / 365.0f;
}

float EuropeanPutFloat::calculateRho(float S, float K, float r, float q,
                                     float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return (S < K) ? -K * T : 0.0f;
  }

  // Calculate d2
  float d1_val = d1(S, K, r, q, vol, T);
  float d2_val = d2(d1_val, vol, T);

  // Calculate rho (as percentage)
  return -0.01f * K * T * std::exp(-r * T) * normalCDF(-d2_val);
}

std::vector<float> EuropeanPutFloat::batchCalculatePrice(
    float S, const std::vector<float> &strikes, float r, float q, float vol,
    float T) const {
  // Return empty vector for empty input
  if (strikes.empty()) {
    return {};
  }

  std::vector<float> results(strikes.size());
  const size_t n = strikes.size();

  // For small batches, use scalar computation
  if (n < 8) {
    for (size_t i = 0; i < n; ++i) {
      results[i] = calculatePrice(S, strikes[i], r, q, vol, T);
    }
    return results;
  }

  // Process in groups of 8 using AVX2 SIMD
  size_t i = 0;

  // Process complete groups of 8
  for (; i + 7 < n; i += 8) {
    std::array<float, 8> strike_array = {
        strikes[i],     strikes[i + 1], strikes[i + 2], strikes[i + 3],
        strikes[i + 4], strikes[i + 5], strikes[i + 6], strikes[i + 7]};

    auto prices = calculatePrice8(
        {S, S, S, S, S, S, S, S}, strike_array, {r, r, r, r, r, r, r, r},
        {q, q, q, q, q, q, q, q}, {vol, vol, vol, vol, vol, vol, vol, vol},
        {T, T, T, T, T, T, T, T});

    for (size_t j = 0; j < 8; ++j) {
      results[i + j] = prices[j];
    }
  }

  // Handle remaining options with scalar computation
  for (; i < n; ++i) {
    results[i] = calculatePrice(S, strikes[i], r, q, vol, T);
  }

  return results;
}

std::array<float, 8> EuropeanPutFloat::calculatePrice8(
    const std::array<float, 8> &spots, const std::array<float, 8> &strikes,
    const std::array<float, 8> &rs, const std::array<float, 8> &qs,
    const std::array<float, 8> &vols, const std::array<float, 8> &Ts) const {

  // Initialize result array
  std::array<float, 8> results = {0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f};

  // Check for degenerate cases where SIMD calculation would fail
  bool has_degenerate = false;
  for (size_t i = 0; i < 8; ++i) {
    if (vols[i] <= 0.0f || Ts[i] <= 0.0f) {
      has_degenerate = true;
      break;
    }
  }

  // If we have degenerate cases, fall back to scalar calculation
  if (has_degenerate) {
    for (size_t i = 0; i < 8; ++i) {
      results[i] =
          calculatePrice(spots[i], strikes[i], rs[i], qs[i], vols[i], Ts[i]);
    }
    return results;
  }

  // Use AVX2 SIMD operations for batch calculation
  __m256 S_vec = _mm256_loadu_ps(spots.data());
  __m256 K_vec = _mm256_loadu_ps(strikes.data());
  __m256 r_vec = _mm256_loadu_ps(rs.data());
  __m256 q_vec = _mm256_loadu_ps(qs.data());
  __m256 vol_vec = _mm256_loadu_ps(vols.data());
  __m256 T_vec = _mm256_loadu_ps(Ts.data());

  // Calculate Black-Scholes put prices using opt::SimdOpsFloat
  __m256 prices =
      opt::SimdOpsFloat::bsPut(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);

  // Store results
  _mm256_storeu_ps(results.data(), prices);

  return results;
}

// EuropeanCallFloat implementation
float EuropeanCallFloat::calculatePrice(float S, float K, float r, float q,
                                        float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return std::max(0.0f, S - K);
  }

  // Calculate d1 and d2
  float d1_val = d1(S, K, r, q, vol, T);
  float d2_val = d2(d1_val, vol, T);

  // Calculate option price using fast approximation
  float Nd1 = normalCDF(d1_val);
  float Nd2 = normalCDF(d2_val);

  return S * std::exp(-q * T) * Nd1 - K * std::exp(-r * T) * Nd2;
}

float EuropeanCallFloat::calculateDelta(float S, float K, float r, float q,
                                        float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return (S > K) ? 1.0f : 0.0f;
  }

  // Calculate d1
  float d1_val = d1(S, K, r, q, vol, T);

  // Calculate delta
  return std::exp(-q * T) * normalCDF(d1_val);
}

float EuropeanCallFloat::calculateGamma(float S, float K, float r, float q,
                                        float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return 0.0f;
  }

  // Calculate d1
  float d1_val = d1(S, K, r, q, vol, T);

  // Calculate gamma
  return std::exp(-q * T) * normalPDF(d1_val) / (S * vol * std::sqrt(T));
}

float EuropeanCallFloat::calculateVega(float S, float K, float r, float q,
                                       float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return 0.0f;
  }

  // Calculate d1
  float d1_val = d1(S, K, r, q, vol, T);

  // Calculate vega (as percentage of spot price)
  return 0.01f * S * std::exp(-q * T) * normalPDF(d1_val) * std::sqrt(T);
}

float EuropeanCallFloat::calculateTheta(float S, float K, float r, float q,
                                        float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return 0.0f;
  }

  // Calculate d1 and d2
  float d1_val = d1(S, K, r, q, vol, T);
  float d2_val = d2(d1_val, vol, T);

  // Calculate theta (per calendar day)
  float term1 =
      -S * std::exp(-q * T) * normalPDF(d1_val) * vol / (2.0f * std::sqrt(T));
  float term2 = -q * S * std::exp(-q * T) * normalCDF(d1_val);
  float term3 = r * K * std::exp(-r * T) * normalCDF(d2_val);

  return (term1 + term2 + term3) / 365.0f;
}

float EuropeanCallFloat::calculateRho(float S, float K, float r, float q,
                                      float vol, float T) const {
  // Handle special cases
  if (vol <= 0.0f || T <= 0.0f) {
    return (S > K) ? K * T : 0.0f;
  }

  // Calculate d2
  float d1_val = d1(S, K, r, q, vol, T);
  float d2_val = d2(d1_val, vol, T);

  // Calculate rho (as percentage)
  return 0.01f * K * T * std::exp(-r * T) * normalCDF(d2_val);
}

std::vector<float> EuropeanCallFloat::batchCalculatePrice(
    float S, const std::vector<float> &strikes, float r, float q, float vol,
    float T) const {
  // Return empty vector for empty input
  if (strikes.empty()) {
    return {};
  }

  std::vector<float> results(strikes.size());
  const size_t n = strikes.size();

  // For small batches, use scalar computation
  if (n < 8) {
    for (size_t i = 0; i < n; ++i) {
      results[i] = calculatePrice(S, strikes[i], r, q, vol, T);
    }
    return results;
  }

  // Process in groups of 8 using AVX2 SIMD
  size_t i = 0;

  // Process complete groups of 8
  for (; i + 7 < n; i += 8) {
    std::array<float, 8> strike_array = {
        strikes[i],     strikes[i + 1], strikes[i + 2], strikes[i + 3],
        strikes[i + 4], strikes[i + 5], strikes[i + 6], strikes[i + 7]};

    auto prices = calculatePrice8(
        {S, S, S, S, S, S, S, S}, strike_array, {r, r, r, r, r, r, r, r},
        {q, q, q, q, q, q, q, q}, {vol, vol, vol, vol, vol, vol, vol, vol},
        {T, T, T, T, T, T, T, T});

    for (size_t j = 0; j < 8; ++j) {
      results[i + j] = prices[j];
    }
  }

  // Handle remaining options with scalar computation
  for (; i < n; ++i) {
    results[i] = calculatePrice(S, strikes[i], r, q, vol, T);
  }

  return results;
}

std::array<float, 8> EuropeanCallFloat::calculatePrice8(
    const std::array<float, 8> &spots, const std::array<float, 8> &strikes,
    const std::array<float, 8> &rs, const std::array<float, 8> &qs,
    const std::array<float, 8> &vols, const std::array<float, 8> &Ts) const {

  // Initialize result array
  std::array<float, 8> results = {0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f};

  // Check for degenerate cases where SIMD calculation would fail
  bool has_degenerate = false;
  for (size_t i = 0; i < 8; ++i) {
    if (vols[i] <= 0.0f || Ts[i] <= 0.0f) {
      has_degenerate = true;
      break;
    }
  }

  // If we have degenerate cases, fall back to scalar calculation
  if (has_degenerate) {
    for (size_t i = 0; i < 8; ++i) {
      results[i] =
          calculatePrice(spots[i], strikes[i], rs[i], qs[i], vols[i], Ts[i]);
    }
    return results;
  }

  // Use AVX2 SIMD operations for batch calculation
  __m256 S_vec = _mm256_loadu_ps(spots.data());
  __m256 K_vec = _mm256_loadu_ps(strikes.data());
  __m256 r_vec = _mm256_loadu_ps(rs.data());
  __m256 q_vec = _mm256_loadu_ps(qs.data());
  __m256 vol_vec = _mm256_loadu_ps(vols.data());
  __m256 T_vec = _mm256_loadu_ps(Ts.data());

  // Calculate Black-Scholes call prices using opt::SimdOpsFloat
  __m256 prices =
      opt::SimdOpsFloat::bsCall(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);

  // Store results
  _mm256_storeu_ps(results.data(), prices);

  return results;
}

float putCallParityFloat(bool isPut, float price, float S, float K, float r,
                         float q, float T) {
  float pv_K = K * std::exp(-r * T); // Present value of strike
  float pv_S = S * std::exp(-q * T); // Present value of spot with dividends

  if (isPut) {
    // Convert call to put: P = C - S*exp(-q*T) + K*exp(-r*T)
    return price - pv_S + pv_K;
  } else {
    // Convert put to call: C = P + S*exp(-q*T) - K*exp(-r*T)
    return price + pv_S - pv_K;
  }
}

} // namespace mod
} // namespace alo
} // namespace engine