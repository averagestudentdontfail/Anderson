#include "vector.h"
#include "simd.h" 
#include <algorithm>
#include <cmath>
#include <cstdint> 
#include <cstring> 
#include <limits> 

// Ensure M_SQRT2 and M_PI are defined
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880 // sqrt(2)
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846 // pi
#endif

const double INV_SQRT_2PI = 0.39894228040143267794;
const float INV_SQRT_2PI_F = 0.3989422804f;

#ifdef _WIN32
#include <intrin.h>
inline void get_cpuidex(int output[4], int level, int sublevel) { __cpuidex(output, level, sublevel); }
inline void get_cpuid(int output[4], int level) { __cpuid(output, level); } 
#else
#include <cpuid.h>
inline void get_cpuidex(int output[4], int level, int sublevel) {
  __cpuid_count(level, sublevel, output[0], output[1], output[2], output[3]);
}
inline void get_cpuid(int output[4], int level) { 
  __get_cpuid(level, (unsigned int*)&output[0], (unsigned int*)&output[1], (unsigned int*)&output[2], (unsigned int*)&output[3]);
}
#endif

namespace engine {
namespace alo {
namespace opt {

SIMDSupport detectSIMDSupport() {
    int info[4];
    get_cpuid(info, 0x00000001);

    bool sse2 = (info[3] & (1 << 26)) != 0;
    bool avx  = (info[2] & (1 << 28)) != 0;

    bool avx2 = false;
    if (avx) { 
        get_cpuidex(info, 0x00000007, 0);
        avx2 = (info[1] & (1 << 5)) != 0;
    }

    if (avx2) return AVX2;
    if (avx) return AVX; 
    if (sse2) return SSE2;
    return NONE;
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

// --- Scalar Black-Scholes Helper Functions ---
namespace { 
    // Double precision scalar Black-Scholes
    double scalar_bs_d1(double S, double K, double r, double q, double vol, double T) {
        return (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
    }
    double scalar_bs_nd(double x) { // N(x)
        return 0.5 * (1.0 + std::erf(x / M_SQRT2));
    }
    double scalar_bs_pdf(double x) { // n(x)
        return INV_SQRT_2PI * std::exp(-0.5 * x * x);
    }

    // Single precision scalar Black-Scholes (using num::fast_*)
    float scalar_bs_d1_f(float S, float K, float r, float q, float vol, float T) {
        return (std::log(S / K) + (r - q + 0.5f * vol * vol) * T) / (vol * std::sqrt(T));
    }
    // For N(x) and n(x) in single precision, we use the num::fast_normal_cdf and num::fast_normal_pdf from float.h
}


// --- VectorDouble Implementations ---
void VectorDouble::EuropeanPut(const double *S_arr, const double *K_arr,
                               const double *r_arr, const double *q_arr,
                               const double *vol_arr, const double *T_arr,
                               double *result_arr, size_t size) {
  size_t i = 0;
  const size_t avx2_step = 4; 

  for (; i + (avx2_step - 1) < size; i += avx2_step) {
    __m256d S_vec = SimdOperationDouble::load(S_arr + i);
    __m256d K_vec = SimdOperationDouble::load(K_arr + i);
    __m256d r_vec = SimdOperationDouble::load(r_arr + i);
    __m256d q_vec = SimdOperationDouble::load(q_arr + i);
    __m256d vol_vec = SimdOperationDouble::load(vol_arr + i);
    __m256d T_vec = SimdOperationDouble::load(T_arr + i);

    __m256d prices = SimdOperationDouble::EuropeanPut(S_vec, K_vec, r_vec,
                                                      q_vec, vol_vec, T_vec);
    SimdOperationDouble::store(result_arr + i, prices);
  }

  for (; i < size; ++i) {
    if (vol_arr[i] <= 1e-12 || T_arr[i] <= 1e-12) {
        result_arr[i] = std::max(0.0, K_arr[i] - S_arr[i]);
        continue;
    }
    double d1 = scalar_bs_d1(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
    double d2 = d1 - vol_arr[i] * std::sqrt(T_arr[i]);
    result_arr[i] = K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * scalar_bs_nd(-d2) -
                    S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_nd(-d1);
  }
}

void VectorDouble::EuropeanCall(const double *S_arr, const double *K_arr,
                                const double *r_arr, const double *q_arr,
                                const double *vol_arr, const double *T_arr,
                                double *result_arr, size_t size) {
  size_t i = 0;
  const size_t avx2_step = 4; 

  for (; i + (avx2_step - 1) < size; i += avx2_step) {
    __m256d S_vec = SimdOperationDouble::load(S_arr + i);
    __m256d K_vec = SimdOperationDouble::load(K_arr + i);
    __m256d r_vec = SimdOperationDouble::load(r_arr + i);
    __m256d q_vec = SimdOperationDouble::load(q_arr + i);
    __m256d vol_vec = SimdOperationDouble::load(vol_arr + i);
    __m256d T_vec = SimdOperationDouble::load(T_arr + i);

    __m256d prices = SimdOperationDouble::EuropeanCall(S_vec, K_vec, r_vec,
                                                       q_vec, vol_vec, T_vec);
    SimdOperationDouble::store(result_arr + i, prices);
  }

  for (; i < size; ++i) {
    if (vol_arr[i] <= 1e-12 || T_arr[i] <= 1e-12) {
        result_arr[i] = std::max(0.0, S_arr[i] - K_arr[i]);
        continue;
    }
    double d1 = scalar_bs_d1(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
    double d2 = d1 - vol_arr[i] * std::sqrt(T_arr[i]);
    result_arr[i] = S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_nd(d1) -
                    K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * scalar_bs_nd(d2);
  }
}

void VectorDouble::EuropeanPutGreek(const double *S_arr, const double *K_arr,
                                    const double *r_arr, const double *q_arr,
                                    const double *vol_arr, const double *T_arr,
                                    GreekDouble *results_arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (vol_arr[i] <= 1e-12 || T_arr[i] <= 1e-12) {
            results_arr[i].price = std::max(0.0, K_arr[i] - S_arr[i]);
            results_arr[i].delta = (S_arr[i] < K_arr[i]) ? -1.0 : 0.0;
            results_arr[i].gamma = 0.0;
            results_arr[i].vega  = 0.0;
            results_arr[i].theta = r_arr[i] * K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) - q_arr[i] * S_arr[i] * std::exp(-q_arr[i] * T_arr[i]); // Simplified
             if (results_arr[i].price > 0) results_arr[i].theta /= 365.0; else results_arr[i].theta = 0;
            results_arr[i].rho   = -K_arr[i] * T_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * 0.01;
            if (results_arr[i].price == 0.0) results_arr[i].rho = 0.0; // if valueless, no rho
            continue;
        }

        double d1 = scalar_bs_d1(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
        double d2 = d1 - vol_arr[i] * std::sqrt(T_arr[i]);

        results_arr[i].price = K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * scalar_bs_nd(-d2) -
                               S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_nd(-d1);
        results_arr[i].delta = std::exp(-q_arr[i] * T_arr[i]) * (scalar_bs_nd(d1) - 1.0);
        results_arr[i].gamma = std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_pdf(d1) / (S_arr[i] * vol_arr[i] * std::sqrt(T_arr[i]));
        results_arr[i].vega  = S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_pdf(d1) * std::sqrt(T_arr[i]) * 0.01;
        
        double theta_term1 = - S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_pdf(d1) * vol_arr[i] / (2.0 * std::sqrt(T_arr[i]));
        double theta_term2 =   q_arr[i] * S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_nd(-d1);
        double theta_term3 = - r_arr[i] * K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * scalar_bs_nd(-d2);
        results_arr[i].theta = (theta_term1 + theta_term2 + theta_term3) / 365.0;
        results_arr[i].rho   = -K_arr[i] * T_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * scalar_bs_nd(-d2) * 0.01;
    }
}

void VectorDouble::EuropeanCallGreek(const double *S_arr, const double *K_arr,
                                     const double *r_arr, const double *q_arr,
                                     const double *vol_arr, const double *T_arr,
                                     GreekDouble *results_arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (vol_arr[i] <= 1e-12 || T_arr[i] <= 1e-12) {
            results_arr[i].price = std::max(0.0, S_arr[i] - K_arr[i]);
            results_arr[i].delta = (S_arr[i] > K_arr[i]) ? 1.0 : 0.0;
            results_arr[i].gamma = 0.0;
            results_arr[i].vega  = 0.0;
            results_arr[i].theta = -r_arr[i] * K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) + q_arr[i] * S_arr[i] * std::exp(-q_arr[i] * T_arr[i]); // Simplified
            if (results_arr[i].price > 0) results_arr[i].theta /= 365.0; else results_arr[i].theta = 0;
            results_arr[i].rho   = K_arr[i] * T_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * 0.01;
            if (results_arr[i].price == 0.0) results_arr[i].rho = 0.0;
            continue;
        }

        double d1 = scalar_bs_d1(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
        double d2 = d1 - vol_arr[i] * std::sqrt(T_arr[i]);

        results_arr[i].price = S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_nd(d1) -
                               K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * scalar_bs_nd(d2);
        results_arr[i].delta = std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_nd(d1);
        results_arr[i].gamma = std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_pdf(d1) / (S_arr[i] * vol_arr[i] * std::sqrt(T_arr[i]));
        results_arr[i].vega  = S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_pdf(d1) * std::sqrt(T_arr[i]) * 0.01;

        double theta_term1 = - S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_pdf(d1) * vol_arr[i] / (2.0 * std::sqrt(T_arr[i]));
        double theta_term2 = - q_arr[i] * S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * scalar_bs_nd(d1);
        double theta_term3 =   r_arr[i] * K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * scalar_bs_nd(d2);
        results_arr[i].theta = (theta_term1 + theta_term2 + theta_term3) / 365.0;
        results_arr[i].rho   = K_arr[i] * T_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * scalar_bs_nd(d2) * 0.01;
    }
}

// Robust Barone-Adesi-Whaley Approximation (Scalar Double)
double scalar_baw_put_approx(double S, double K, double r, double q, double vol, double T) {
    if (T <= 1e-9) return std::max(0.0, K - S); // Intrinsic for zero time
    if (vol <= 1e-9) return std::max(0.0, K * std::exp(-r * T) - S * std::exp(-q * T)); // Discounted intrinsic

    double euro_put = K * std::exp(-r * T) * scalar_bs_nd(-scalar_bs_d1(S,K,r,q,vol,T) + vol*std::sqrt(T)) - 
                      S * std::exp(-q * T) * scalar_bs_nd(-scalar_bs_d1(S,K,r,q,vol,T));

    if (r <= q || r <= 1e-9) { // Condition where early exercise is not expected or BAW is problematic
        return std::max(euro_put, K - S);
    }

    double b = r - q;
    double M = 2.0 * r / (vol * vol);
    double N_baw = 2.0 * b / (vol * vol);
    
    double q2_num = -(N_baw - 1.0) + std::sqrt((N_baw - 1.0) * (N_baw - 1.0) + 4.0 * M);
    double q2 = 0.5 * q2_num;

    if (q2 <= 1.0 + 1e-9) { // Critical price would be problematic or infinite
        return std::max(euro_put, K - S);
    }
    
    // Critical stock price (S_star)
    // S_star = K / (1 - 1/q2) = K * q2 / (q2 - 1)
    double S_star = K * q2 / (q2 - 1.0);

    if (S <= S_star) {
        return K - S;
    } else {
        double d1_S_star = scalar_bs_d1(S, S_star, r, q, vol, T); // d1 with S_star as strike for A2
        double A2 = (S_star / q2) * (1.0 - std::exp((b - r) * T) * scalar_bs_nd(-d1_S_star));
        double premium_val = euro_put + A2 * std::pow(S / S_star, -q2);
        return std::max(premium_val, K-S); // Ensure not less than intrinsic
    }
}

double scalar_baw_call_approx(double S, double K, double r, double q, double vol, double T) {
    if (q <= 1e-9) { // No dividend, American Call = European Call
        if (T <= 1e-9 || vol <= 1e-9) return std::max(0.0, S - K);
        double d1 = scalar_bs_d1(S,K,r,q,vol,T);
        double d2 = d1 - vol*std::sqrt(T);
        return S * std::exp(-q*T) * scalar_bs_nd(d1) - K * std::exp(-r*T) * scalar_bs_nd(d2);
    }
    if (T <= 1e-9) return std::max(0.0, S - K);
    if (vol <= 1e-9) return std::max(0.0, S * std::exp(-q * T) - K * std::exp(-r * T));


    double euro_call = S * std::exp(-q * T) * scalar_bs_nd(scalar_bs_d1(S,K,r,q,vol,T)) - 
                       K * std::exp(-r * T) * scalar_bs_nd(scalar_bs_d1(S,K,r,q,vol,T) - vol*std::sqrt(T));
    
    double b = r - q;
    double M = 2.0 * r / (vol*vol);
    double N_baw = 2.0 * b / (vol*vol);

    double q1_num = -(N_baw - 1.0) - std::sqrt( (N_baw - 1.0)*(N_baw - 1.0) + 4.0*M); // Note the minus before sqrt for call's q1
    double q1 = 0.5 * q1_num;

    if (q1 >= -1e-9 || std::abs(q1 - 1.0) < 1e-9 ) { // S_star would be problematic
         return std::max(euro_call, S - K);
    }

    // Critical stock price S_star = K / (1 - 1/q1) = K * q1 / (q1 - 1)
    double S_star = K * q1 / (q1 - 1.0);

    if (S >= S_star) {
        return S - K;
    } else {
        double d1_S_star = scalar_bs_d1(S, S_star, r, q, vol, T); // d1 with S_star as strike for A1
        double A1 = -(S_star / q1) * (1.0 - std::exp((b - r) * T) * scalar_bs_nd(d1_S_star));
        double premium_val = euro_call + A1 * std::pow(S / S_star, q1);
        return std::max(premium_val, S-K);
    }
}


void VectorDouble::AmericanPutApprox(const double *S_arr, const double *K_arr,
                                    const double *r_arr, const double *q_arr,
                                    const double *vol_arr, const double *T_arr,
                                    double *result_arr, size_t size) {
    // For now, we use the scalar robust BAW approximation.
    // SIMD version would require vectorizing the BAW logic carefully.
    for(size_t i=0; i<size; ++i) {
        result_arr[i] = scalar_baw_put_approx(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
    }
}

void VectorDouble::AmericanCallApprox(const double *S_arr, const double *K_arr,
                                     const double *r_arr, const double *q_arr,
                                     const double *vol_arr, const double *T_arr,
                                     double *result_arr, size_t size) {
    for(size_t i=0; i<size; ++i) {
        result_arr[i] = scalar_baw_call_approx(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
    }
}

// --- VectorSingle Implementations ---
void VectorSingle::EuropeanPut(const float *S_arr, const float *K_arr,
                               const float *r_arr, const float *q_arr,
                               const float *vol_arr, const float *T_arr,
                               float *result_arr, size_t size) {
  size_t i = 0;
  const size_t avx2_step = 8; 

  for (; i + (avx2_step - 1) < size; i += avx2_step) {
    __m256 S_vec = SimdOperationSingle::load(S_arr + i);
    __m256 K_vec = SimdOperationSingle::load(K_arr + i);
    __m256 r_vec = SimdOperationSingle::load(r_arr + i);
    __m256 q_vec = SimdOperationSingle::load(q_arr + i);
    __m256 vol_vec = SimdOperationSingle::load(vol_arr + i);
    __m256 T_vec = SimdOperationSingle::load(T_arr + i);

    __m256 prices = SimdOperationSingle::EuropeanPut(S_vec, K_vec, r_vec,
                                                     q_vec, vol_vec, T_vec);
    SimdOperationSingle::store(result_arr + i, prices);
  }

  for (; i < size; ++i) {
     if (vol_arr[i] <= 1e-7f || T_arr[i] <= 1e-7f) { 
        result_arr[i] = std::max(0.0f, K_arr[i] - S_arr[i]);
        continue;
    }
    float d1 = scalar_bs_d1_f(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
    float d2 = d1 - vol_arr[i] * std::sqrt(T_arr[i]);
    result_arr[i] = K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * num::fast_normal_cdf(-d2) -
                    S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * num::fast_normal_cdf(-d1);
  }
}

void VectorSingle::EuropeanCall(const float *S_arr, const float *K_arr,
                                const float *r_arr, const float *q_arr,
                                const float *vol_arr, const float *T_arr,
                                float *result_arr, size_t size) {
  size_t i = 0;
  const size_t avx2_step = 8;

  for (; i + (avx2_step-1) < size; i += avx2_step) {
    __m256 S_vec = SimdOperationSingle::load(S_arr + i);
    __m256 K_vec = SimdOperationSingle::load(K_arr + i);
    __m256 r_vec = SimdOperationSingle::load(r_arr + i);
    __m256 q_vec = SimdOperationSingle::load(q_arr + i);
    __m256 vol_vec = SimdOperationSingle::load(vol_arr + i);
    __m256 T_vec = SimdOperationSingle::load(T_arr + i);

    __m256 prices = SimdOperationSingle::EuropeanCall(S_vec, K_vec, r_vec,
                                                      q_vec, vol_vec, T_vec);
    SimdOperationSingle::store(result_arr + i, prices);
  }
  for (; i < size; ++i) {
    if (vol_arr[i] <= 1e-7f || T_arr[i] <= 1e-7f) {
        result_arr[i] = std::max(0.0f, S_arr[i] - K_arr[i]);
        continue;
    }
    float d1 = scalar_bs_d1_f(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
    float d2 = d1 - vol_arr[i] * std::sqrt(T_arr[i]);
    result_arr[i] = S_arr[i] * std::exp(-q_arr[i] * T_arr[i]) * num::fast_normal_cdf(d1) -
                    K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * num::fast_normal_cdf(d2);
  }
}

void VectorSingle::EuropeanPutGreek(const float *S_arr, const float *K_arr,
                                    const float *r_arr, const float *q_arr,
                                    const float *vol_arr, const float *T_arr,
                                    GreekSingle *results_arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (vol_arr[i] <= 1e-7f || T_arr[i] <= 1e-7f) {
            results_arr[i].price = std::max(0.0f, K_arr[i] - S_arr[i]);
            results_arr[i].delta = (S_arr[i] < K_arr[i]) ? -1.0f : 0.0f;
            results_arr[i].gamma = 0.0f;
            results_arr[i].vega  = 0.0f;
            results_arr[i].theta = r_arr[i] * K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) - q_arr[i] * S_arr[i] * std::exp(-q_arr[i] * T_arr[i]);
            if (results_arr[i].price > 0.0f) results_arr[i].theta /= 365.0f; else results_arr[i].theta = 0.0f;
            results_arr[i].rho   = -K_arr[i] * T_arr[i] * std::exp(-r_arr[i] * T_arr[i]) * 0.01f;
            if (results_arr[i].price == 0.0f) results_arr[i].rho = 0.0f;
            continue;
        }
        float d1 = scalar_bs_d1_f(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
        float d2 = d1 - vol_arr[i] * std::sqrt(T_arr[i]);

        results_arr[i].price = K_arr[i]*std::exp(-r_arr[i]*T_arr[i])*num::fast_normal_cdf(-d2) - S_arr[i]*std::exp(-q_arr[i]*T_arr[i])*num::fast_normal_cdf(-d1);
        results_arr[i].delta = std::exp(-q_arr[i]*T_arr[i]) * (num::fast_normal_cdf(d1) - 1.0f);
        results_arr[i].gamma = std::exp(-q_arr[i]*T_arr[i]) * num::fast_normal_pdf(d1) / (S_arr[i]*vol_arr[i]*std::sqrt(T_arr[i]));
        results_arr[i].vega  = S_arr[i]*std::exp(-q_arr[i]*T_arr[i]) * num::fast_normal_pdf(d1) * std::sqrt(T_arr[i]) * 0.01f;
        float theta_term1 = -S_arr[i]*std::exp(-q_arr[i]*T_arr[i])*num::fast_normal_pdf(d1)*vol_arr[i]/(2.0f*std::sqrt(T_arr[i]));
        float theta_term2 =  q_arr[i]*S_arr[i]*std::exp(-q_arr[i]*T_arr[i])*num::fast_normal_cdf(-d1);
        float theta_term3 = -r_arr[i]*K_arr[i]*std::exp(-r_arr[i]*T_arr[i])*num::fast_normal_cdf(-d2);
        results_arr[i].theta = (theta_term1 + theta_term2 + theta_term3) / 365.0f;
        results_arr[i].rho   = -K_arr[i]*T_arr[i]*std::exp(-r_arr[i]*T_arr[i])*num::fast_normal_cdf(-d2) * 0.01f;
    }
}

void VectorSingle::EuropeanCallGreek(const float *S_arr, const float *K_arr,
                                     const float *r_arr, const float *q_arr,
                                     const float *vol_arr, const float *T_arr,
                                     GreekSingle *results_arr, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        if (vol_arr[i] <= 1e-7f || T_arr[i] <= 1e-7f) {
            results_arr[i].price = std::max(0.0f, S_arr[i] - K_arr[i]);
            results_arr[i].delta = (S_arr[i] > K_arr[i]) ? 1.0f : 0.0f;
            results_arr[i].gamma = 0.0f;
            results_arr[i].vega  = 0.0f;
            results_arr[i].theta = -r_arr[i] * K_arr[i] * std::exp(-r_arr[i] * T_arr[i]) + q_arr[i] * S_arr[i] * std::exp(-q_arr[i] * T_arr[i]);
             if (results_arr[i].price > 0.0f) results_arr[i].theta /= 365.0f; else results_arr[i].theta = 0.0f;
            results_arr[i].rho   = K_arr[i]*T_arr[i]*std::exp(-r_arr[i]*T_arr[i])*0.01f;
            if (results_arr[i].price == 0.0f) results_arr[i].rho = 0.0f;
            continue;
        }
        float d1 = scalar_bs_d1_f(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
        float d2 = d1 - vol_arr[i]*std::sqrt(T_arr[i]);
        results_arr[i].price = S_arr[i]*std::exp(-q_arr[i]*T_arr[i])*num::fast_normal_cdf(d1) - K_arr[i]*std::exp(-r_arr[i]*T_arr[i])*num::fast_normal_cdf(d2);
        results_arr[i].delta = std::exp(-q_arr[i]*T_arr[i]) * num::fast_normal_cdf(d1);
        results_arr[i].gamma = std::exp(-q_arr[i]*T_arr[i]) * num::fast_normal_pdf(d1) / (S_arr[i]*vol_arr[i]*std::sqrt(T_arr[i]));
        results_arr[i].vega  = S_arr[i]*std::exp(-q_arr[i]*T_arr[i]) * num::fast_normal_pdf(d1) * std::sqrt(T_arr[i]) * 0.01f;
        float theta_term1 = -S_arr[i]*std::exp(-q_arr[i]*T_arr[i])*num::fast_normal_pdf(d1)*vol_arr[i]/(2.0f*std::sqrt(T_arr[i]));
        float theta_term2 = -q_arr[i]*S_arr[i]*std::exp(-q_arr[i]*T_arr[i])*num::fast_normal_cdf(d1);
        float theta_term3 =  r_arr[i]*K_arr[i]*std::exp(-r_arr[i]*T_arr[i])*num::fast_normal_cdf(d2);
        results_arr[i].theta = (theta_term1 + theta_term2 + theta_term3) / 365.0f;
        results_arr[i].rho   = K_arr[i]*T_arr[i]*std::exp(-r_arr[i]*T_arr[i])*num::fast_normal_cdf(d2) * 0.01f;
    }
}


// Robust Barone-Adesi-Whaley Approximation (Scalar Single Precision)
float scalar_baw_put_approx_f(float S, float K, float r, float q, float vol, float T) {
    if (T <= 1e-7f) return std::max(0.0f, K - S);
    if (vol <= 1e-7f) return std::max(0.0f, K * std::exp(-r * T) - S * std::exp(-q * T));

    float euro_put = K * std::exp(-r * T) * num::fast_normal_cdf(-scalar_bs_d1_f(S,K,r,q,vol,T) + vol*std::sqrt(T)) - 
                     S * std::exp(-q * T) * num::fast_normal_cdf(-scalar_bs_d1_f(S,K,r,q,vol,T));

    if (r <= q || r <= 1e-7f) {
        return std::max(euro_put, K - S);
    }

    float b = r - q;
    float M = 2.0f * r / (vol * vol);
    float N_baw = 2.0f * b / (vol * vol);
    
    float q2_num = -(N_baw - 1.0f) + std::sqrt((N_baw - 1.0f) * (N_baw - 1.0f) + 4.0f * M);
    float q2 = 0.5f * q2_num;

    if (q2 <= 1.0f + 1e-7f) {
        return std::max(euro_put, K-S);
    }
    
    float S_star = K * q2 / (q2 - 1.0f);

    if (S <= S_star) {
        return K - S;
    } else {
        float d1_S_star = scalar_bs_d1_f(S, S_star, r, q, vol, T);
        float A2 = (S_star / q2) * (1.0f - std::exp((b - r) * T) * num::fast_normal_cdf(-d1_S_star));
        float premium_val = euro_put + A2 * std::pow(S / S_star, -q2);
        return std::max(premium_val, K-S);
    }
}

float scalar_baw_call_approx_f(float S, float K, float r, float q, float vol, float T) {
    if (q <= 1e-7f) { // No dividend
        if (T <= 1e-7f || vol <= 1e-7f) return std::max(0.0f, S - K);
        float d1 = scalar_bs_d1_f(S,K,r,q,vol,T);
        float d2 = d1 - vol*std::sqrt(T);
        return S * std::exp(-q*T) * num::fast_normal_cdf(d1) - K * std::exp(-r*T) * num::fast_normal_cdf(d2);
    }
    if (T <= 1e-7f) return std::max(0.0f, S - K);
    if (vol <= 1e-7f) return std::max(0.0f, S * std::exp(-q * T) - K * std::exp(-r * T));

    float euro_call = S * std::exp(-q * T) * num::fast_normal_cdf(scalar_bs_d1_f(S,K,r,q,vol,T)) - 
                      K * std::exp(-r * T) * num::fast_normal_cdf(scalar_bs_d1_f(S,K,r,q,vol,T) - vol*std::sqrt(T));
    
    float b = r - q;
    float M = 2.0f * r / (vol*vol);
    float N_baw = 2.0f * b / (vol*vol);

    float q1_num = -(N_baw - 1.0f) - std::sqrt( (N_baw - 1.0f)*(N_baw - 1.0f) + 4.0f*M);
    float q1 = 0.5f * q1_num;

    if (q1 >= -1e-7f || std::abs(q1-1.0f) < 1e-7f) {
        return std::max(euro_call, S-K);
    }
    
    float S_star = K * q1 / (q1 - 1.0f);

    if (S >= S_star) {
        return S - K;
    } else {
        float d1_S_star = scalar_bs_d1_f(S, S_star, r, q, vol, T);
        float A1 = -(S_star / q1) * (1.0f - std::exp((b - r) * T) * num::fast_normal_cdf(d1_S_star));
        float premium_val = euro_call + A1 * std::pow(S / S_star, q1);
        return std::max(premium_val, S-K);
    }
}


void VectorSingle::AmericanPut(const float *S_arr, const float *K_arr,
                               const float *r_arr, const float *q_arr,
                               const float *vol_arr, const float *T_arr,
                               float *result_arr, size_t size) {
  size_t i = 0;
  const size_t avx2_step = 8;

  for (; i + (avx2_step - 1) < size; i += avx2_step) {
    __m256 S_vec = SimdOperationSingle::load(S_arr + i);
    __m256 K_vec = SimdOperationSingle::load(K_arr + i);
    __m256 r_vec = SimdOperationSingle::load(r_arr + i);
    __m256 q_vec = SimdOperationSingle::load(q_arr + i);
    __m256 vol_vec = SimdOperationSingle::load(vol_arr + i);
    __m256 T_vec = SimdOperationSingle::load(T_arr + i);

    __m256 prices = SimdOperationSingle::AmericanPut(S_vec, K_vec, r_vec,
                                                     q_vec, vol_vec, T_vec);
    SimdOperationSingle::store(result_arr + i, prices);
  }
  for (; i < size; ++i) {
      result_arr[i] = scalar_baw_put_approx_f(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
  }
}

void VectorSingle::AmericanCall(const float *S_arr, const float *K_arr,
                                const float *r_arr, const float *q_arr,
                                const float *vol_arr, const float *T_arr,
                                float *result_arr, size_t size) {
  size_t i = 0;
  const size_t avx2_step = 8;

  for (; i + (avx2_step - 1) < size; i += avx2_step) {
    __m256 S_vec = SimdOperationSingle::load(S_arr + i);
    __m256 K_vec = SimdOperationSingle::load(K_arr + i);
    __m256 r_vec = SimdOperationSingle::load(r_arr + i);
    __m256 q_vec = SimdOperationSingle::load(q_arr + i);
    __m256 vol_vec = SimdOperationSingle::load(vol_arr + i);
    __m256 T_vec = SimdOperationSingle::load(T_arr + i);

    __m256 prices = SimdOperationSingle::AmericanCall(S_vec, K_vec, r_vec,
                                                      q_vec, vol_vec, T_vec);
    SimdOperationSingle::store(result_arr + i, prices);
  }
  for (; i < size; ++i) {
      result_arr[i] = scalar_baw_call_approx_f(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
  }
}


} // namespace opt
} // namespace alo
} // namespace engine