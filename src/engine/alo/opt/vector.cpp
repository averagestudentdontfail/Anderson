#include "vector.h"
#include "simd.h" 
#include <algorithm>
#include <cmath>
#include <cstdint> 
#include <cstring> 
#include <limits> 

#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880 
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846 
#endif

// Using const double/float for these for type safety and to avoid macro redefinition issues
const double INV_SQRT_2PI_DBL = 0.39894228040143267794; // 1/sqrt(2*PI) double
const float  INV_SQRT_2PI_SGL = 0.3989422804f;       // 1/sqrt(2*PI) float


#ifdef _WIN32
#include <intrin.h>
// Helper for __cpuidex (level, sublevel)
inline void get_cpuidex_helper(int output[4], int level, int sublevel) { __cpuidex(output, level, sublevel); }
// Helper for __cpuid (level only)
inline void get_cpuid_helper(int output[4], int level) { __cpuid(output, level); }
#else
#include <cpuid.h>
inline void get_cpuidex_helper(int output[4], int level, int sublevel) {
  __cpuid_count(level, sublevel, output[0], output[1], output[2], output[3]);
}
inline void get_cpuid_helper(int output[4], int level) { 
  __get_cpuid(level, (unsigned int*)&output[0], (unsigned int*)&output[1], (unsigned int*)&output[2], (unsigned int*)&output[3]);
}
#endif


namespace engine {
namespace alo {
namespace opt {

SIMDSupport detectSIMDSupport() {
    int info[4] = {0,0,0,0}; 
    get_cpuid_helper(info, 0x00000001);

    bool sse2 = (info[3] & (1 << 26)) != 0; 
    bool avx  = (info[2] & (1 << 28)) != 0; 

    bool avx2 = false;
    if (avx) { 
        int info_leaf7[4] = {0,0,0,0}; 
        get_cpuidex_helper(info_leaf7, 0x00000007, 0);
        avx2 = (info_leaf7[1] & (1 << 5)) != 0; 
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

// --- Scalar Black-Scholes Helper Functions (anonymous namespace) ---
namespace { 
    // Double precision scalar Black-Scholes
    double scalar_bs_d1_dbl(double S, double K, double r, double q, double vol, double T) {
        if (vol <= 1e-12 || T <= 1e-12) { 
            double S_eff = S > 0 ? S : 1e-12; // Avoid log(0)
            double K_eff = K > 0 ? K : 1e-12;
            if (std::abs(S_eff - K_eff) < 1e-9 * K_eff) return (r - q) * std::sqrt(T < 1e-12 ? 1e-12 : T) / (vol < 1e-9 ? 1e-9 : vol);
            return (S_eff > K_eff) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
        }
        return (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
    }
    double scalar_normal_cdf_dbl(double x) {
        return 0.5 * (1.0 + std::erf(x / M_SQRT2));
    }
    double scalar_normal_pdf_dbl(double x) {
        return INV_SQRT_2PI_DBL * std::exp(-0.5 * x * x);
    }

    // Single precision scalar Black-Scholes
    float scalar_bs_d1_sgl(float S, float K, float r, float q, float vol, float T) {
        if (vol <= 1e-7f || T <= 1e-7f) {
            float S_eff = S > 0 ? S : 1e-7f;
            float K_eff = K > 0 ? K : 1e-7f;
            if (std::abs(S_eff - K_eff) < 1e-7f * K_eff) return (r - q) * std::sqrt(T < 1e-7f ? 1e-7f : T) / (vol < 1e-7f ? 1e-7f : vol);
            return (S_eff > K_eff) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
        }
        return (std::log(S / K) + (r - q + 0.5f * vol * vol) * T) / (vol * std::sqrt(T));
    }
    // For float N(x) and n(x), num::fast_normal_cdf and num::fast_normal_pdf are used from num/float.h

    // --- Robust Scalar Barone-Adesi-Whaley Put Approximation (double) ---
    double robust_scalar_baw_put_approx_dbl(double S, double K, double r, double q, double vol, double T) {
        if (T <= 1e-9) return std::max(0.0, K - S);
        if (vol <= 1e-9) return std::max(0.0, K * std::exp(-r * T) - S * std::exp(-q * T));

        double d1_euro = scalar_bs_d1_dbl(S, K, r, q, vol, T);
        double d2_euro = d1_euro - vol * std::sqrt(T);
        double euro_put_price = K * std::exp(-r * T) * scalar_normal_cdf_dbl(-d2_euro) -
                                S * std::exp(-q * T) * scalar_normal_cdf_dbl(-d1_euro);

        if (r <= q || r <= 1e-9) {
            return std::max(euro_put_price, K - S);
        }

        double b_baw = r - q; 
        double M_baw = 2.0 * r / (vol * vol);
        double N_baw = 2.0 * b_baw / (vol * vol);
        
        double q2_discriminant = (N_baw - 1.0) * (N_baw - 1.0) + 4.0 * M_baw;
        if (q2_discriminant < 0.0) return std::max(euro_put_price, K - S); 

        double q2 = 0.5 * (-(N_baw - 1.0) + std::sqrt(q2_discriminant));

        if (q2 <= 1.0 + 1e-9) { 
            return std::max(euro_put_price, K - S);
        }
        
        double S_star = K * q2 / (q2 - 1.0);

        if (S <= S_star) {
            return std::max(0.0, K-S); 
        } else {
            double d1_S_star_baw = scalar_bs_d1_dbl(S, S_star, r, b_baw, vol, T); // Using b_baw for q in d1 for A2
            double A2 = (S_star / q2) * (1.0 - std::exp((b_baw - r) * T) * scalar_normal_cdf_dbl(-d1_S_star_baw));
            
            double american_price = euro_put_price + A2 * std::pow(S / S_star, -q2);
            return std::max(american_price, K - S); 
        }
    }

    // --- Robust Scalar Barone-Adesi-Whaley Call Approximation (double) ---
    double robust_scalar_baw_call_approx_dbl(double S, double K, double r, double q, double vol, double T) {
        if (q <= 1e-9) { 
            if (T <= 1e-9 || vol <= 1e-9) return std::max(0.0, S - K);
            double d1 = scalar_bs_d1_dbl(S,K,r,q,vol,T);
            double d2 = d1 - vol*std::sqrt(T);
            return S * std::exp(-q*T) * scalar_normal_cdf_dbl(d1) - K * std::exp(-r*T) * scalar_normal_cdf_dbl(d2);
        }
        if (T <= 1e-9) return std::max(0.0, S - K);
        if (vol <= 1e-9) return std::max(0.0, S * std::exp(-q * T) - K * std::exp(-r * T));

        double d1_euro = scalar_bs_d1_dbl(S, K, r, q, vol, T);
        double d2_euro = d1_euro - vol * std::sqrt(T);
        double euro_call_price = S * std::exp(-q * T) * scalar_normal_cdf_dbl(d1_euro) - 
                                 K * std::exp(-r * T) * scalar_normal_cdf_dbl(d2_euro);
        
        double b_baw = r - q;
        double M_baw = 2.0 * r / (vol*vol);
        double N_baw = 2.0 * b_baw / (vol*vol);

        double q1_discriminant = (N_baw - 1.0) * (N_baw - 1.0) + 4.0 * M_baw;
         if (q1_discriminant < 0.0) return std::max(euro_call_price, S - K);

        double q1 = 0.5 * (-(N_baw - 1.0) - std::sqrt(q1_discriminant)); 

        if (q1 >= -1e-9 || std::abs(q1 - 1.0) < 1e-9 ) { 
             return std::max(euro_call_price, S - K);
        }

        double S_star = K * q1 / (q1 - 1.0);
        
        if (S >= S_star && S_star > 0) { // Ensure S_star is reasonable
            return std::max(0.0, S-K);
        } else {
            double d1_S_star_baw = scalar_bs_d1_dbl(S, S_star, r, b_baw, vol, T); // Using b_baw for q in d1 for A1
            double A1 = -(S_star / q1) * (1.0 - std::exp((b_baw - r) * T) * scalar_normal_cdf_dbl(d1_S_star_baw));
            
            double american_price = euro_call_price + A1 * std::pow(S / S_star, q1);
            return std::max(american_price, S - K);
        }
    }

    // --- Robust Scalar Barone-Adesi-Whaley Put Approximation (float) ---
    float robust_scalar_baw_put_approx_sgl(float S, float K, float r, float q, float vol, float T) {
        if (T <= 1e-7f) return std::max(0.0f, K - S);
        if (vol <= 1e-7f) return std::max(0.0f, K * std::exp(-r * T) - S * std::exp(-q * T));

        float d1_euro = scalar_bs_d1_sgl(S, K, r, q, vol, T);
        float d2_euro = d1_euro - vol * std::sqrt(T);
        float euro_put_price = K * std::exp(-r * T) * num::fast_normal_cdf(-d2_euro) -
                               S * std::exp(-q * T) * num::fast_normal_cdf(-d1_euro);

        if (r <= q || r <= 1e-7f) {
            return std::max(euro_put_price, K - S);
        }

        float b_baw = r - q;
        float M_baw = 2.0f * r / (vol * vol);
        float N_baw = 2.0f * b_baw / (vol * vol);
        
        float q2_discriminant = (N_baw - 1.0f) * (N_baw - 1.0f) + 4.0f * M_baw;
        if (q2_discriminant < 0.0f) return std::max(euro_put_price, K - S);

        float q2 = 0.5f * (-(N_baw - 1.0f) + std::sqrt(q2_discriminant));

        if (q2 <= 1.0f + 1e-7f) {
            return std::max(euro_put_price, K-S);
        }
        
        float S_star = K * q2 / (q2 - 1.0f);

        if (S <= S_star) {
            return std::max(0.0f, K-S);
        } else {
            float d1_S_star_baw = scalar_bs_d1_sgl(S, S_star, r, b_baw, vol, T);
            float A2 = (S_star / q2) * (1.0f - std::exp((b_baw - r) * T) * num::fast_normal_cdf(-d1_S_star_baw));
            float american_price = euro_put_price + A2 * std::pow(S / S_star, -q2);
            return std::max(american_price, K-S);
        }
    }

    // --- Robust Scalar Barone-Adesi-Whaley Call Approximation (float) ---
    float robust_scalar_baw_call_approx_sgl(float S, float K, float r, float q, float vol, float T) {
        if (q <= 1e-7f) { 
            if (T <= 1e-7f || vol <= 1e-7f) return std::max(0.0f, S - K);
            float d1 = scalar_bs_d1_sgl(S,K,r,q,vol,T);
            float d2 = d1 - vol*std::sqrt(T);
            return S * std::exp(-q*T) * num::fast_normal_cdf(d1) - K * std::exp(-r*T) * num::fast_normal_cdf(d2);
        }
        if (T <= 1e-7f) return std::max(0.0f, S - K);
        if (vol <= 1e-7f) return std::max(0.0f, S * std::exp(-q * T) - K * std::exp(-r * T));

        float d1_euro = scalar_bs_d1_sgl(S, K, r, q, vol, T);
        float d2_euro = d1_euro - vol * std::sqrt(T);
        float euro_call_price = S * std::exp(-q * T) * num::fast_normal_cdf(d1_euro) - 
                                K * std::exp(-r * T) * num::fast_normal_cdf(d2_euro);
        
        float b_baw = r - q;
        float M_baw = 2.0f * r / (vol*vol);
        float N_baw = 2.0f * b_baw / (vol*vol);

        float q1_discriminant = (N_baw - 1.0f) * (N_baw - 1.0f) + 4.0f * M_baw;
        if (q1_discriminant < 0.0f) return std::max(euro_call_price, S - K);

        float q1 = 0.5f * (-(N_baw - 1.0f) - std::sqrt(q1_discriminant)); 

        if (q1 >= -1e-7f || std::abs(q1 - 1.0f) < 1e-7f ) { 
             return std::max(euro_call_price, S-K);
        }
        
        float S_star = K * q1 / (q1 - 1.0f);
        
        if (S >= S_star && S_star > 0.0f) {
            return std::max(0.0f, S-K);
        } else {
            float d1_S_star_baw = scalar_bs_d1_sgl(S, S_star, r, b_baw, vol, T);
            float A1 = -(S_star / q1) * (1.0f - std::exp((b_baw - r) * T) * num::fast_normal_cdf(d1_S_star_baw));
            float american_price = euro_call_price + A1 * std::pow(S / S_star, q1);
            return std::max(american_price, S - K);
        }
    }
} // end anonymous namespace


// --- VectorDouble Implementations ---
void VectorDouble::AmericanPutApprox(const double *S_arr, const double *K_arr,
                                    const double *r_arr, const double *q_arr,
                                    const double *vol_arr, const double *T_arr,
                                    double *result_arr, size_t size) {
    // This can be SIMD accelerated if SimdOperationDouble::AmericanPutApprox is implemented
    // For now, using scalar robust BAW.
    for(size_t i=0; i<size; ++i) {
        result_arr[i] = robust_scalar_baw_put_approx_dbl(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
    }
}

void VectorDouble::AmericanCallApprox(const double *S_arr, const double *K_arr,
                                     const double *r_arr, const double *q_arr,
                                     const double *vol_arr, const double *T_arr,
                                     double *result_arr, size_t size) {
    for(size_t i=0; i<size; ++i) {
        result_arr[i] = robust_scalar_baw_call_approx_dbl(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
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
      result_arr[i] = robust_scalar_baw_put_approx_sgl(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
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
      result_arr[i] = robust_scalar_baw_call_approx_sgl(S_arr[i], K_arr[i], r_arr[i], q_arr[i], vol_arr[i], T_arr[i]);
  }
}


} // namespace opt
} // namespace alo
} // namespace engine