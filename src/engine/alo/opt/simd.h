/**
 * @file simd.h
 * @brief SIMD optimizations for the ALO engine
 * 
 * This file defines SIMD (Single Instruction, Multiple Data) operations
 * for accelerating option pricing calculations in the ALO engine.
 */
 #ifndef ENGINE_ALO_OPT_SIMD_H
 #define ENGINE_ALO_OPT_SIMD_H
 
 #include <immintrin.h>  // For AVX/AVX2 intrinsics
 #include <array>
 #include <cmath>
 #include <algorithm>
 
 // Platform-specific headers for CPU feature detection
 #if defined(_MSC_VER)
     // MSVC
     #include <intrin.h>
 #elif defined(__GNUC__) || defined(__clang__)
     // GCC or Clang
     #include <cpuid.h>
 #endif
 
 namespace engine {
 namespace alo {
 namespace opt {
 
 /**
  * @class SimdOps
  * @brief SIMD operations for AVX2 (256-bit vectors)
  * 
  * This class provides SIMD operations for accelerating option pricing
  * calculations using AVX2 instructions. All operations are designed
  * for deterministic execution.
  */
 class SimdOps {
 public:
     /**
      * @brief Load 4 doubles into a vector
      * 
      * @param ptr Pointer to array of 4 doubles
      * @return 256-bit vector containing 4 doubles
      */
     static inline __m256d load(const double* ptr) {
         return _mm256_loadu_pd(ptr);
     }
     
     /**
      * @brief Load 4 doubles from an array
      * 
      * @param arr Array of 4 doubles
      * @return 256-bit vector containing 4 doubles
      */
     static inline __m256d load(const std::array<double, 4>& arr) {
         return _mm256_loadu_pd(arr.data());
     }
     
     /**
      * @brief Store a vector into 4 doubles
      * 
      * @param ptr Pointer to array of 4 doubles
      * @param vec 256-bit vector containing 4 doubles
      */
     static inline void store(double* ptr, __m256d vec) {
         _mm256_storeu_pd(ptr, vec);
     }
     
     /**
      * @brief Store a vector into an array
      * 
      * @param arr Array of 4 doubles
      * @param vec 256-bit vector containing 4 doubles
      */
     static inline void store(std::array<double, 4>& arr, __m256d vec) {
         _mm256_storeu_pd(arr.data(), vec);
     }
     
     /**
      * @brief Set all elements to a scalar value
      * 
      * @param value Scalar value
      * @return 256-bit vector with all elements set to value
      */
     static inline __m256d set1(double value) {
         return _mm256_set1_pd(value);
     }
     
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
      * @brief Add two vectors
      * 
      * @param a First vector
      * @param b Second vector
      * @return Vector containing a + b
      */
     static inline __m256d add(__m256d a, __m256d b) {
         return _mm256_add_pd(a, b);
     }
     
     /**
      * @brief Subtract two vectors
      * 
      * @param a First vector
      * @param b Second vector
      * @return Vector containing a - b
      */
     static inline __m256d sub(__m256d a, __m256d b) {
         return _mm256_sub_pd(a, b);
     }
     
     /**
      * @brief Multiply two vectors
      * 
      * @param a First vector
      * @param b Second vector
      * @return Vector containing a * b
      */
     static inline __m256d mul(__m256d a, __m256d b) {
         return _mm256_mul_pd(a, b);
     }
     
     /**
      * @brief Divide two vectors
      * 
      * @param a First vector
      * @param b Second vector
      * @return Vector containing a / b
      */
     static inline __m256d div(__m256d a, __m256d b) {
         return _mm256_div_pd(a, b);
     }
     
     /**
      * @brief Maximum of two vectors
      * 
      * @param a First vector
      * @param b Second vector
      * @return Vector containing max(a, b)
      */
     static inline __m256d max(__m256d a, __m256d b) {
         return _mm256_max_pd(a, b);
     }
     
     /**
      * @brief Minimum of two vectors
      * 
      * @param a First vector
      * @param b Second vector
      * @return Vector containing min(a, b)
      */
     static inline __m256d min(__m256d a, __m256d b) {
         return _mm256_min_pd(a, b);
     }
     
     /**
      * @brief Natural logarithm of a vector
      * 
      * @param x Input vector
      * @return Vector containing log(x)
      */
     static inline __m256d log(__m256d x) {
         // AVX2 doesn't have a direct log function, use scalar computation
         std::array<double, 4> values;
         store(values, x);
         
         values[0] = std::log(values[0]);
         values[1] = std::log(values[1]);
         values[2] = std::log(values[2]);
         values[3] = std::log(values[3]);
         
         return load(values);
     }
     
     /**
      * @brief Exponential function of a vector
      * 
      * @param x Input vector
      * @return Vector containing exp(x)
      */
     static inline __m256d exp(__m256d x) {
         // AVX2 doesn't have a direct exp function, use scalar computation
         std::array<double, 4> values;
         store(values, x);
         
         values[0] = std::exp(values[0]);
         values[1] = std::exp(values[1]);
         values[2] = std::exp(values[2]);
         values[3] = std::exp(values[3]);
         
         return load(values);
     }
     
     /**
      * @brief Square root of a vector
      * 
      * @param x Input vector
      * @return Vector containing sqrt(x)
      */
     static inline __m256d sqrt(__m256d x) {
         return _mm256_sqrt_pd(x);
     }
     
     /**
      * @brief Error function of a vector
      * 
      * @param x Input vector
      * @return Vector containing erf(x)
      */
     static inline __m256d erf(__m256d x) {
         // AVX2 doesn't have a direct erf function, use scalar computation
         std::array<double, 4> values;
         store(values, x);
         
         values[0] = std::erf(values[0]);
         values[1] = std::erf(values[1]);
         values[2] = std::erf(values[2]);
         values[3] = std::erf(values[3]);
         
         return load(values);
     }
     
     /**
      * @brief Normal CDF of a vector
      * 
      * @param x Input vector
      * @return Vector containing normalCDF(x)
      */
     static inline __m256d normalCDF(__m256d x) {
         __m256d half = set1(0.5);
         __m256d sqrt2inv = set1(1.0 / std::sqrt(2.0));
         
         // normalCDF(x) = 0.5 * (1 + erf(x / sqrt(2)))
         return add(half, mul(half, erf(mul(x, sqrt2inv))));
     }
     
     /**
      * @brief Normal PDF of a vector
      * 
      * @param x Input vector
      * @return Vector containing normalPDF(x)
      */
     static inline __m256d normalPDF(__m256d x) {
         __m256d minusHalf = set1(-0.5);
         __m256d invSqrt2Pi = set1(1.0 / std::sqrt(2.0 * M_PI));
         
         // normalPDF(x) = exp(-0.5 * x^2) / sqrt(2 * PI)
         return mul(exp(mul(minusHalf, mul(x, x))), invSqrt2Pi);
     }
     
     /**
      * @brief Calculate Black-Scholes d1 for 4 options at once
      * 
      * @param S Spot prices
      * @param K Strike prices
      * @param r Risk-free rates
      * @param q Dividend yields
      * @param vol Volatilities
      * @param T Times to maturity
      * @return Vector containing d1 values
      */
     static inline __m256d bsD1(__m256d S, __m256d K, __m256d r, __m256d q, __m256d vol, __m256d T) {
         __m256d vol_sqrt_T = mul(vol, sqrt(T));
         __m256d half_vol_squared = mul(set1(0.5), mul(vol, vol));
         
         // d1 = (ln(S/K) + (r - q + 0.5 * vol^2) * T) / (vol * sqrt(T))
         return div(
             add(
                 log(div(S, K)),
                 mul(add(sub(r, q), half_vol_squared), T)
             ),
             vol_sqrt_T
         );
     }
     
     /**
      * @brief Calculate Black-Scholes d2 for 4 options at once
      * 
      * @param d1 d1 values
      * @param vol Volatilities
      * @param T Times to maturity
      * @return Vector containing d2 values
      */
     static inline __m256d bsD2(__m256d d1, __m256d vol, __m256d T) {
         // d2 = d1 - vol * sqrt(T)
         return sub(d1, mul(vol, sqrt(T)));
     }
     
     /**
      * @brief Calculate Black-Scholes put prices for 4 options at once
      * 
      * @param S Spot prices
      * @param K Strike prices
      * @param r Risk-free rates
      * @param q Dividend yields
      * @param vol Volatilities
      * @param T Times to maturity
      * @return Vector containing put option prices
      */
     static inline __m256d bsPut(__m256d S, __m256d K, __m256d r, __m256d q, __m256d vol, __m256d T) {
         __m256d zero = set1(0.0);
         
         // Handle degenerate cases
         __m256d vol_mask = _mm256_cmp_pd(vol, zero, _CMP_GT_OQ);
         __m256d T_mask = _mm256_cmp_pd(T, zero, _CMP_GT_OQ);
         __m256d valid_mask = _mm256_and_pd(vol_mask, T_mask);
         
         if (_mm256_movemask_pd(valid_mask) == 0) {
             // All elements are degenerate, return max(0, K-S)
             return max(zero, sub(K, S));
         }
         
         __m256d d1 = bsD1(S, K, r, q, vol, T);
         __m256d d2 = bsD2(d1, vol, T);
         
         // Negative d1 and d2 for put calculation
         __m256d neg_d1 = sub(zero, d1);
         __m256d neg_d2 = sub(zero, d2);
         
         // N(-d1) and N(-d2)
         __m256d Nd1 = normalCDF(neg_d1);
         __m256d Nd2 = normalCDF(neg_d2);
         
         // discount factors
         __m256d dr = exp(mul(sub(zero, r), T));
         __m256d dq = exp(mul(sub(zero, q), T));
         
         // put = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
         __m256d put_value = sub(
             mul(mul(K, dr), Nd2),
             mul(mul(S, dq), Nd1)
         );
         
         // For degenerate cases, use max(0, K-S)
         __m256d intrinsic = max(zero, sub(K, S));
         
         return _mm256_blendv_pd(intrinsic, put_value, valid_mask);
     }
     
     /**
      * @brief Calculate Black-Scholes call prices for 4 options at once
      * 
      * @param S Spot prices
      * @param K Strike prices
      * @param r Risk-free rates
      * @param q Dividend yields
      * @param vol Volatilities
      * @param T Times to maturity
      * @return Vector containing call option prices
      */
     static inline __m256d bsCall(__m256d S, __m256d K, __m256d r, __m256d q, __m256d vol, __m256d T) {
         __m256d zero = set1(0.0);
         
         // Handle degenerate cases
         __m256d vol_mask = _mm256_cmp_pd(vol, zero, _CMP_GT_OQ);
         __m256d T_mask = _mm256_cmp_pd(T, zero, _CMP_GT_OQ);
         __m256d valid_mask = _mm256_and_pd(vol_mask, T_mask);
         
         if (_mm256_movemask_pd(valid_mask) == 0) {
             // All elements are degenerate, return max(0, S-K)
             return max(zero, sub(S, K));
         }
         
         __m256d d1 = bsD1(S, K, r, q, vol, T);
         __m256d d2 = bsD2(d1, vol, T);
         
         // N(d1) and N(d2)
         __m256d Nd1 = normalCDF(d1);
         __m256d Nd2 = normalCDF(d2);
         
         // discount factors
         __m256d dr = exp(mul(sub(zero, r), T));
         __m256d dq = exp(mul(sub(zero, q), T));
         
         // call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
         __m256d call_value = sub(
             mul(mul(S, dq), Nd1),
             mul(mul(K, dr), Nd2)
         );
         
         // For degenerate cases, use max(0, S-K)
         __m256d intrinsic = max(zero, sub(S, K));
         
         return _mm256_blendv_pd(intrinsic, call_value, valid_mask);
     }
     
     /**
      * @brief Extract a single element from a vector
      * 
      * @param vec Input vector
      * @param index Index of element to extract (0-3)
      * @return Extracted element
      */
     static inline double extract(__m256d vec, int index) {
         std::array<double, 4> values;
         store(values, vec);
         return values[index];
     }
     
     /**
      * @brief Sum all elements in a vector
      * 
      * @param vec Input vector
      * @return Sum of all elements
      */
     static inline double sum(__m256d vec) {
         // Sum the elements of the vector
         __m128d low = _mm256_extractf128_pd(vec, 0);
         __m128d high = _mm256_extractf128_pd(vec, 1);
         __m128d sum_128 = _mm_add_pd(low, high);
         double sum_values[2];
         _mm_storeu_pd(sum_values, sum_128);
         return sum_values[0] + sum_values[1];
     }
 };
 
 /**
  * @brief SIMD support detection and CPU feature detection
  */
 class SimdDetect {
 public:
     /**
      * @brief Check if AVX2 instructions are supported
      * 
      * @return True if AVX2 is supported, false otherwise
      */
     static bool hasAVX2() {
         static const bool has_avx2 = checkAVX2Support();
         return has_avx2;
     }
     
     /**
      * @brief Check if AVX-512 instructions are supported
      * 
      * @return True if AVX-512 is supported, false otherwise
      */
     static bool hasAVX512() {
         static const bool has_avx512 = checkAVX512Support();
         return has_avx512;
     }
     
 private:
     static bool checkAVX2Support() {
         // Cross-platform implementation of AVX2 detection
 #if defined(_MSC_VER)
         // MSVC implementation
         int cpuInfo[4];
         int nIds = 0;
         
         __cpuidex(cpuInfo, 0, 0);
         nIds = cpuInfo[0];
         
         bool avxSupported = false;
         bool avx2Supported = false;
         
         if (nIds >= 1) {
             __cpuidex(cpuInfo, 1, 0);
             avxSupported = (cpuInfo[2] & (1 << 28)) != 0;  // Check for AVX
         }
         
         if (nIds >= 7) {
             __cpuidex(cpuInfo, 7, 0);
             avx2Supported = (cpuInfo[1] & (1 << 5)) != 0;  // Check for AVX2
         }
         
         return avxSupported && avx2Supported;
 #elif defined(__GNUC__) || defined(__clang__)
         // GCC or Clang implementation
         unsigned int eax, ebx, ecx, edx;
         
         // Check for basic CPUID support
         if (__get_cpuid_max(0, nullptr) < 7) {
             return false;
         }
         
         // Check for AVX support (CPUID.1:ECX.AVX[bit 28])
         __cpuid(1, eax, ebx, ecx, edx);
         bool avxSupported = (ecx & (1 << 28)) != 0;
         
         // Check for AVX2 support (CPUID.7.0:EBX.AVX2[bit 5])
         __cpuid_count(7, 0, eax, ebx, ecx, edx);
         bool avx2Supported = (ebx & (1 << 5)) != 0;
         
         return avxSupported && avx2Supported;
 #else
         // Fallback for other platforms - assume AVX2 is not available
         return false;
 #endif
     }
 
     static bool checkAVX512Support() {
         // Cross-platform implementation of AVX-512 detection
 #if defined(_MSC_VER)
         // MSVC implementation
         int cpuInfo[4];
         int nIds = 0;
         
         __cpuidex(cpuInfo, 0, 0);
         nIds = cpuInfo[0];
         
         if (nIds >= 7) {
             __cpuidex(cpuInfo, 7, 0);
             // Check for AVX-512 Foundation
             return (cpuInfo[1] & (1 << 16)) != 0;
         }
         
         return false;
 #elif defined(__GNUC__) || defined(__clang__)
         // GCC or Clang implementation
         unsigned int eax, ebx, ecx, edx;
         
         // Check for basic CPUID support
         if (__get_cpuid_max(0, nullptr) < 7) {
             return false;
         }
         
         // Check for AVX-512 Foundation (CPUID.7.0:EBX.AVX512F[bit 16])
         __cpuid_count(7, 0, eax, ebx, ecx, edx);
         return (ebx & (1 << 16)) != 0;
 #else
         // Fallback for other platforms - assume AVX-512 is not available
         return false;
 #endif
     }
 };
 
 /**
  * @brief Check if SIMD optimization is available
  * 
  * @return True if SIMD optimization is available, false otherwise
  */
 inline bool isSimdAvailable() {
     return SimdDetect::hasAVX2();
 }
 
 } // namespace opt
 } // namespace alo
 } // namespace engine
 
 #endif // ENGINE_ALO_OPT_SIMD_H