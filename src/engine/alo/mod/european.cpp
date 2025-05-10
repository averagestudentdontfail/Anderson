#include "european.h"
#include "../opt/simd.h"   
#include "../num/float.h"    

#include <algorithm>   
#include <cmath>        
#include <stdexcept>   
#include <limits>       

// Ensure M_PI and M_SQRT2 are defined
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
// Constant for 1/sqrt(2*PI)
const double INV_SQRT_2PI_DBL_EUR = 0.39894228040143267794;
const float  INV_SQRT_2PI_SGL_EUR = 0.3989422804f;


namespace engine {
namespace alo {
namespace mod {

// --- EuropeanOptionDouble ---
double EuropeanOptionDouble::d1(double S, double K, double r, double q, double vol,
                                double T) {
  // Robust handling for vol or T being extremely small or zero
  if (vol <= 1e-12 || T <= 1e-12) {
    double S_eff = S > 0 ? S : 1e-12;
    double K_eff = K > 0 ? K : 1e-12;
    if (std::abs(S_eff - K_eff) < 1e-9 * K_eff) { // Effectively at-the-money
        // d1 tends to (r-q)/vol * sqrt(T) + 0.5 * vol * sqrt(T)
        // If vol is also tiny, this can be large.
        // A very small perturbation can swing d1 from -inf to +inf.
        // Let's return based on drift to avoid NaN from 0/0.
        double drift_term = (r - q + 0.0 * vol * vol) * (T < 1e-12 ? 1e-12 : T); // Use 0 for vol^2 term
        return drift_term > 0 ? std::numeric_limits<double>::infinity() : 
               (drift_term < 0 ? -std::numeric_limits<double>::infinity() : 0.0);
    }
    return (S_eff > K_eff) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
  }
  return (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
}

double EuropeanOptionDouble::d2(double d1_val, double vol, double T) {
  if (T <= 1e-12) return d1_val; // Avoid issues with sqrt(0) if T is effectively zero
  return d1_val - vol * std::sqrt(T);
}

double EuropeanOptionDouble::normalCDF(double x) {
  return 0.5 * (1.0 + std::erf(x / M_SQRT2));
}

double EuropeanOptionDouble::normalPDF(double x) {
  return INV_SQRT_2PI_DBL_EUR * std::exp(-0.5 * x * x);
}

double EuropeanPutDouble::calculatePrice(double S, double K, double r, double q,
                                   double vol, double T) const {
  if (T <= 1e-9) { return std::max(0.0, K - S); } // Handle time expiry separately for clarity
  if (vol <= 1e-9) { return std::max(0.0, K * std::exp(-r * T) - S * std::exp(-q * T)); }


  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);

  return K * std::exp(-r * T) * normalCDF(-d2_val) - S * std::exp(-q * T) * normalCDF(-d1_val);
}

double EuropeanPutDouble::calculateDelta(double S, double K, double r, double q,
                                   double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9) { return (S < K) ? -1.0 : 0.0; }
  double d1_val = d1(S, K, r, q, vol, T);
  return std::exp(-q * T) * (normalCDF(d1_val) - 1.0);
}

double EuropeanPutDouble::calculateGamma(double S, double K, double r, double q,
                                   double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9 || S <= 1e-9) { return 0.0; }
  double d1_val = d1(S, K, r, q, vol, T);
  return std::exp(-q * T) * normalPDF(d1_val) / (S * vol * std::sqrt(T));
}

double EuropeanPutDouble::calculateVega(double S, double K, double r, double q,
                                  double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9 || S <= 1e-9) { return 0.0; }
  double d1_val = d1(S, K, r, q, vol, T);
  return 0.01 * S * std::exp(-q * T) * normalPDF(d1_val) * std::sqrt(T);
}

double EuropeanPutDouble::calculateTheta(double S, double K, double r, double q,
                                   double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9) { 
      // Simplified theta at expiry or zero vol (rate of change of discounted intrinsic)
      return (r * K * std::exp(-r * T) - q * S * std::exp(-q * T)) / 365.0; 
  }
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);
  double term1 = -S * std::exp(-q * T) * normalPDF(d1_val) * vol / (2.0 * std::sqrt(T));
  double term2 =  q * S * std::exp(-q * T) * normalCDF(-d1_val); // +qS...N(-d1) for put
  double term3 = -r * K * std::exp(-r * T) * normalCDF(-d2_val); // -rK...N(-d2) for put
  return (term1 + term2 + term3) / 365.0;
}

double EuropeanPutDouble::calculateRho(double S, double K, double r, double q,
                                 double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9) { 
      return ( (K * std::exp(-r*T) - S * std::exp(-q*T) > 0) ? -0.01 * K * T * std::exp(-r*T) : 0.0 );
  }
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);
  return -0.01 * K * T * std::exp(-r * T) * normalCDF(-d2_val);
}

std::vector<double>
EuropeanPutDouble::batchCalculatePrice(double S, const std::vector<double> &strikes,
                                 double r, double q, double vol, double T) const {
  if (strikes.empty()) return {};
  std::vector<double> results(strikes.size());
  opt::VectorDouble::EuropeanPut(&S, strikes.data(), &r, &q, &vol, &T, results.data(), strikes.size());
  // Note: This passes pointers to single doubles for S,r,q,vol,T.
  // VectorDouble::EuropeanPut needs to handle this (e.g. by broadcasting S,r,q,vol,T in its SIMD part)
  // Or, more simply, this batch method can create full vectors for all params.
  // For now, assuming VectorDouble::EuropeanPut handles scalar broadcast internally or loops.
  // A cleaner way for this specific function:
  std::vector<double> S_vec(strikes.size(), S);
  std::vector<double> r_vec(strikes.size(), r);
  std::vector<double> q_vec(strikes.size(), q);
  std::vector<double> vol_vec(strikes.size(), vol);
  std::vector<double> T_vec(strikes.size(), T);
  opt::VectorDouble::EuropeanPut(S_vec.data(), strikes.data(), r_vec.data(), q_vec.data(), 
                                 vol_vec.data(), T_vec.data(), results.data(), strikes.size());
  return results;
}

std::array<double, 4> EuropeanPutDouble::calculatePrice4(
    const std::array<double, 4> &spots, const std::array<double, 4> &strikes,
    const std::array<double, 4> &rs, const std::array<double, 4> &qs,
    const std::array<double, 4> &vols, const std::array<double, 4> &Ts) const {
  std::array<double, 4> results;
  // Direct call to SIMD operation
  __m256d S_vec = opt::SimdOperationDouble::load(spots);
  __m256d K_vec = opt::SimdOperationDouble::load(strikes);
  __m256d r_vec = opt::SimdOperationDouble::load(rs);
  __m256d q_vec = opt::SimdOperationDouble::load(qs);
  __m256d vol_vec = opt::SimdOperationDouble::load(vols);
  __m256d T_vec = opt::SimdOperationDouble::load(Ts);
  __m256d prices = opt::SimdOperationDouble::EuropeanPut(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);
  opt::SimdOperationDouble::store(results, prices);
  return results;
}

// --- EuropeanCallDouble ---
// (Implementations mirror EuropeanPutDouble with call formulas)
double EuropeanCallDouble::calculatePrice(double S, double K, double r, double q,
                                    double vol, double T) const {
  if (T <= 1e-9) { return std::max(0.0, S - K); }
  if (vol <= 1e-9) { return std::max(0.0, S * std::exp(-q * T) - K * std::exp(-r * T)); }

  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);
  return S * std::exp(-q * T) * normalCDF(d1_val) - K * std::exp(-r * T) * normalCDF(d2_val);
}

double EuropeanCallDouble::calculateDelta(double S, double K, double r, double q,
                                    double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9) { return (S > K) ? 1.0 : 0.0; }
  double d1_val = d1(S, K, r, q, vol, T);
  return std::exp(-q * T) * normalCDF(d1_val);
}

double EuropeanCallDouble::calculateGamma(double S, double K, double r, double q,
                                    double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9 || S <= 1e-9) { return 0.0; }
  double d1_val = d1(S, K, r, q, vol, T);
  return std::exp(-q * T) * normalPDF(d1_val) / (S * vol * std::sqrt(T));
}

double EuropeanCallDouble::calculateVega(double S, double K, double r, double q,
                                   double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9 || S <= 1e-9) { return 0.0; }
  double d1_val = d1(S, K, r, q, vol, T);
  return 0.01 * S * std::exp(-q * T) * normalPDF(d1_val) * std::sqrt(T);
}

double EuropeanCallDouble::calculateTheta(double S, double K, double r, double q,
                                    double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9) { 
      return (-r * K * std::exp(-r * T) + q * S * std::exp(-q * T)) / 365.0;
  }
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);
  double term1 = -S * std::exp(-q * T) * normalPDF(d1_val) * vol / (2.0 * std::sqrt(T));
  double term2 = -q * S * std::exp(-q * T) * normalCDF(d1_val); // -qS...N(d1) for call
  double term3 =  r * K * std::exp(-r * T) * normalCDF(d2_val); // +rK...N(d2) for call
  return (term1 + term2 + term3) / 365.0;
}

double EuropeanCallDouble::calculateRho(double S, double K, double r, double q,
                                  double vol, double T) const {
  if (T <= 1e-9 || vol <= 1e-9) { 
      return ( (S * std::exp(-q*T) - K * std::exp(-r*T) > 0) ? 0.01 * K * T * std::exp(-r*T) : 0.0 );
  }
  double d1_val = d1(S, K, r, q, vol, T);
  double d2_val = d2(d1_val, vol, T);
  return 0.01 * K * T * std::exp(-r * T) * normalCDF(d2_val);
}

std::vector<double>
EuropeanCallDouble::batchCalculatePrice(double S, const std::vector<double> &strikes,
                                  double r, double q, double vol, double T) const {
  if (strikes.empty()) return {};
  std::vector<double> results(strikes.size());
  std::vector<double> S_vec(strikes.size(), S);
  std::vector<double> r_vec(strikes.size(), r);
  std::vector<double> q_vec(strikes.size(), q);
  std::vector<double> vol_vec(strikes.size(), vol);
  std::vector<double> T_vec(strikes.size(), T);
  opt::VectorDouble::EuropeanCall(S_vec.data(), strikes.data(), r_vec.data(), q_vec.data(), 
                                 vol_vec.data(), T_vec.data(), results.data(), strikes.size());
  return results;
}

std::array<double, 4> EuropeanCallDouble::calculatePrice4(
    const std::array<double, 4> &spots, const std::array<double, 4> &strikes,
    const std::array<double, 4> &rs, const std::array<double, 4> &qs,
    const std::array<double, 4> &vols, const std::array<double, 4> &Ts) const {
  std::array<double, 4> results;
  __m256d S_vec = opt::SimdOperationDouble::load(spots);
  __m256d K_vec = opt::SimdOperationDouble::load(strikes);
  __m256d r_vec = opt::SimdOperationDouble::load(rs);
  __m256d q_vec = opt::SimdOperationDouble::load(qs);
  __m256d vol_vec = opt::SimdOperationDouble::load(vols);
  __m256d T_vec = opt::SimdOperationDouble::load(Ts);
  __m256d prices = opt::SimdOperationDouble::EuropeanCall(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);
  opt::SimdOperationDouble::store(results, prices);
  return results;
}

double putCallParityDouble(bool isCall, double option_price, double S, double K, double r,
                     double q, double T) {
  double pv_K = K * std::exp(-r * T); 
  double pv_S = S * std::exp(-q * T); 
  if (isCall) { // Given price is a call, calculate put
    return option_price - pv_S + pv_K; // P = C - S0_adj + K_adj
  } else { // Given price is a put, calculate call
    return option_price + pv_S - pv_K; // C = P + S0_adj - K_adj
  }
}


// --- EuropeanOptionSingle ---
float EuropeanOptionSingle::d1(float S, float K, float r, float q, float vol,
                              float T) {
  if (vol <= 1e-7f || T <= 1e-7f) {
    float S_eff = S > 0 ? S : 1e-7f;
    float K_eff = K > 0 ? K : 1e-7f;
     if (std::abs(S_eff - K_eff) < 1e-7f * K_eff) {
        float drift_term = (r - q + 0.0f * vol * vol) * (T < 1e-7f ? 1e-7f : T);
        return drift_term > 0 ? std::numeric_limits<float>::infinity() : 
               (drift_term < 0 ? -std::numeric_limits<float>::infinity() : 0.0f);
    }
    return (S_eff > K_eff) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
  }
  return (std::log(S / K) + (r - q + 0.5f * vol * vol) * T) / (vol * std::sqrt(T));
}

float EuropeanOptionSingle::d2(float d1_val, float vol, float T) {
  if (T <= 1e-7f) return d1_val;
  return d1_val - vol * std::sqrt(T);
}

float EuropeanOptionSingle::normalCDF(float x) {
  return num::fast_normal_cdf(x);
}

float EuropeanOptionSingle::normalPDF(float x) {
  return num::fast_normal_pdf(x);
}

// --- EuropeanPutSingle ---
float EuropeanPutSingle::calculatePrice(float S, float K, float r, float q,
                                       float vol, float T) const {
  if (T <= 1e-7f) { return std::max(0.0f, K - S); }
  if (vol <= 1e-7f) { return std::max(0.0f, K * std::exp(-r * T) - S * std::exp(-q * T)); }
  
  float d1_val = d1(S, K, r, q, vol, T);
  float d2_val = d2(d1_val, vol, T);
  return K * std::exp(-r * T) * normalCDF(-d2_val) - S * std::exp(-q * T) * normalCDF(-d1_val);
}
// (Other EuropeanPutSingle Greeks and batch methods follow the same pattern as EuropeanPutDouble,
//  using float types and num::fast_normal_cdf/pdf. SIMD calls would use SimdOperationSingle)

float EuropeanPutSingle::calculateDelta(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f) { return (S < K) ? -1.0f : 0.0f; }
  float d1_val = d1(S,K,r,q,vol,T);
  return std::exp(-q*T) * (normalCDF(d1_val) - 1.0f);
}
float EuropeanPutSingle::calculateGamma(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f || S <= 1e-7f) { return 0.0f; }
  float d1_val = d1(S,K,r,q,vol,T);
  return std::exp(-q*T) * normalPDF(d1_val) / (S*vol*std::sqrt(T));
}
float EuropeanPutSingle::calculateVega(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f || S <= 1e-7f) { return 0.0f; }
  float d1_val = d1(S,K,r,q,vol,T);
  return 0.01f * S * std::exp(-q*T) * normalPDF(d1_val) * std::sqrt(T);
}
float EuropeanPutSingle::calculateTheta(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f) { return (r*K*std::exp(-r*T) - q*S*std::exp(-q*T)) / 365.0f; }
  float d1_val = d1(S,K,r,q,vol,T);
  float d2_val = d2(d1_val,vol,T);
  float term1 = -S*std::exp(-q*T)*normalPDF(d1_val)*vol / (2.0f * std::sqrt(T));
  float term2 =  q*S*std::exp(-q*T)*normalCDF(-d1_val);
  float term3 = -r*K*std::exp(-r*T)*normalCDF(-d2_val);
  return (term1+term2+term3)/365.0f;
}
float EuropeanPutSingle::calculateRho(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f) { return ((K*std::exp(-r*T) - S*std::exp(-q*T) > 0.0f) ? -0.01f*K*T*std::exp(-r*T) : 0.0f); }
  float d1_val = d1(S,K,r,q,vol,T);
  float d2_val = d2(d1_val,vol,T);
  return -0.01f * K * T * std::exp(-r*T) * normalCDF(-d2_val);
}

std::vector<float> EuropeanPutSingle::batchCalculatePrice(
    float S, const std::vector<float> &strikes, float r, float q, float vol,
    float T) const {
  if (strikes.empty()) return {};
  std::vector<float> results(strikes.size());
  std::vector<float> S_vec(strikes.size(), S);
  std::vector<float> r_vec(strikes.size(), r);
  std::vector<float> q_vec(strikes.size(), q);
  std::vector<float> vol_vec(strikes.size(), vol);
  std::vector<float> T_vec(strikes.size(), T);
  opt::VectorSingle::EuropeanPut(S_vec.data(), strikes.data(), r_vec.data(), q_vec.data(), 
                                 vol_vec.data(), T_vec.data(), results.data(), strikes.size());
  return results;
}

std::array<float, 8> EuropeanPutSingle::calculatePrice8(
    const std::array<float, 8> &spots, const std::array<float, 8> &strikes,
    const std::array<float, 8> &rs, const std::array<float, 8> &qs,
    const std::array<float, 8> &vols, const std::array<float, 8> &Ts) const {
  std::array<float, 8> results;
  __m256 S_vec = opt::SimdOperationSingle::load(spots);
  __m256 K_vec = opt::SimdOperationSingle::load(strikes);
  __m256 r_vec = opt::SimdOperationSingle::load(rs);
  __m256 q_vec = opt::SimdOperationSingle::load(qs);
  __m256 vol_vec = opt::SimdOperationSingle::load(vols);
  __m256 T_vec = opt::SimdOperationSingle::load(Ts);
  __m256 prices = opt::SimdOperationSingle::EuropeanPut(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);
  opt::SimdOperationSingle::store(results, prices);
  return results;
}


// --- EuropeanCallSingle ---
// (Implementations mirror EuropeanCallDouble with call formulas and float types)
float EuropeanCallSingle::calculatePrice(float S, float K, float r, float q,
                                        float vol, float T) const {
  if (T <= 1e-7f) { return std::max(0.0f, S - K); }
  if (vol <= 1e-7f) { return std::max(0.0f, S * std::exp(-q * T) - K * std::exp(-r * T)); }

  float d1_val = d1(S, K, r, q, vol, T);
  float d2_val = d2(d1_val, vol, T);
  return S * std::exp(-q * T) * normalCDF(d1_val) - K * std::exp(-r * T) * normalCDF(d2_val);
}
float EuropeanCallSingle::calculateDelta(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f) { return (S > K) ? 1.0f : 0.0f; }
  float d1_val = d1(S,K,r,q,vol,T);
  return std::exp(-q*T) * normalCDF(d1_val);
}
float EuropeanCallSingle::calculateGamma(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f || S <= 1e-7f) { return 0.0f; }
  float d1_val = d1(S,K,r,q,vol,T);
  return std::exp(-q*T) * normalPDF(d1_val) / (S*vol*std::sqrt(T));
}
float EuropeanCallSingle::calculateVega(float S, float K, float r, float q, float vol, float T) const {
 if (T <= 1e-7f || vol <= 1e-7f || S <= 1e-7f) { return 0.0f; }
  float d1_val = d1(S,K,r,q,vol,T);
  return 0.01f * S * std::exp(-q*T) * normalPDF(d1_val) * std::sqrt(T);
}
float EuropeanCallSingle::calculateTheta(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f) { return (-r*K*std::exp(-r*T) + q*S*std::exp(-q*T)) / 365.0f; }
  float d1_val = d1(S,K,r,q,vol,T);
  float d2_val = d2(d1_val,vol,T);
  float term1 = -S*std::exp(-q*T)*normalPDF(d1_val)*vol / (2.0f * std::sqrt(T));
  float term2 = -q*S*std::exp(-q*T)*normalCDF(d1_val);
  float term3 =  r*K*std::exp(-r*T)*normalCDF(d2_val);
  return (term1+term2+term3)/365.0f;
}
float EuropeanCallSingle::calculateRho(float S, float K, float r, float q, float vol, float T) const {
  if (T <= 1e-7f || vol <= 1e-7f) { return ((S*std::exp(-q*T) - K*std::exp(-r*T) > 0.0f) ? 0.01f*K*T*std::exp(-r*T) : 0.0f); }
  float d1_val = d1(S,K,r,q,vol,T);
  float d2_val = d2(d1_val,vol,T);
  return 0.01f * K * T * std::exp(-r*T) * normalCDF(d2_val);
}

std::vector<float> EuropeanCallSingle::batchCalculatePrice(
    float S, const std::vector<float> &strikes, float r, float q, float vol,
    float T) const {
  if (strikes.empty()) return {};
  std::vector<float> results(strikes.size());
  std::vector<float> S_vec(strikes.size(), S);
  std::vector<float> r_vec(strikes.size(), r);
  std::vector<float> q_vec(strikes.size(), q);
  std::vector<float> vol_vec(strikes.size(), vol);
  std::vector<float> T_vec(strikes.size(), T);
  opt::VectorSingle::EuropeanCall(S_vec.data(), strikes.data(), r_vec.data(), q_vec.data(), 
                                 vol_vec.data(), T_vec.data(), results.data(), strikes.size());
  return results;
}

std::array<float, 8> EuropeanCallSingle::calculatePrice8(
    const std::array<float, 8> &spots, const std::array<float, 8> &strikes,
    const std::array<float, 8> &rs, const std::array<float, 8> &qs,
    const std::array<float, 8> &vols, const std::array<float, 8> &Ts) const {
  std::array<float, 8> results;
  __m256 S_vec = opt::SimdOperationSingle::load(spots);
  __m256 K_vec = opt::SimdOperationSingle::load(strikes);
  __m256 r_vec = opt::SimdOperationSingle::load(rs);
  __m256 q_vec = opt::SimdOperationSingle::load(qs);
  __m256 vol_vec = opt::SimdOperationSingle::load(vols);
  __m256 T_vec = opt::SimdOperationSingle::load(Ts);
  __m256 prices = opt::SimdOperationSingle::EuropeanCall(S_vec, K_vec, r_vec, q_vec, vol_vec, T_vec);
  opt::SimdOperationSingle::store(results, prices);
  return results;
}

// --- Put-Call Parity ---
float putCallParitySingle(bool isCall, float option_price, float S, float K, float r,
                         float q, float T) {
  float pv_K = K * std::exp(-r * T); 
  float pv_S = S * std::exp(-q * T); 
  if (isCall) { // Given price is a call, calculate put
    return option_price - pv_S + pv_K;
  } else { // Given price is a put, calculate call
    return option_price + pv_S - pv_K;
  }
}


} // namespace mod
} // namespace alo
} // namespace engine