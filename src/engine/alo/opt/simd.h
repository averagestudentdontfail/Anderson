 #ifndef ENGINE_ALO_OPT_SIMD_H
 #define ENGINE_ALO_OPT_SIMD_H
 
 #include <immintrin.h>  
 #include <sleef.h>     
 #include <array>
 #include <cmath>
 #include <algorithm>
 #include <cstring>  
 #include <cstdint>  
 
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
 
 /**
  * @brief Threshold for using SIMD operations
  * 
  * Only use SIMD for operations with size >= this threshold
  */
 constexpr size_t SIMD_THRESHOLD = 32;
 
 /**
  * @brief Check if we should use SIMD for a given data size
  * 
  * @param size Size of the data to process
  * @return True if SIMD should be used, false otherwise
  */
 inline bool shouldUseSimd(size_t size) {
     return isSimdAvailable() && size >= SIMD_THRESHOLD;
 }
 
 /**
  * @class SimdOps
  * @brief Optimized SIMD operations for financial math
  * 
  * This class provides SIMD operations optimized for financial mathematics
  * calculations with AVX2 instructions (256-bit vectors) and SLEEF for 
  * transcendental functions.
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
      * @brief Square root of a vector using SLEEF or AVX2
      * 
      * @param x Input vector
      * @return Vector containing sqrt(x)
      */
     static inline __m256d sqrt(__m256d x) {
         // Use AVX2 sqrt (faster and more accurate)
         return _mm256_sqrt_pd(x);
     }
 
     /**
      * @brief Exponential of a vector using AVX2
      * 
      * @param x Input vector
      * @return Vector containing exp(x)
      */
     static inline __m256d exp(__m256d x) {
         // Custom optimized implementation
         // Constants for polynomial approximation
         const __m256d LOG2E = _mm256_set1_pd(1.4426950408889634);
         const __m256d LN2 = _mm256_set1_pd(0.6931471805599453);
         const __m256d ONE = _mm256_set1_pd(1.0);
         const __m256d C1 = _mm256_set1_pd(1.0);
         const __m256d C2 = _mm256_set1_pd(0.5);
         const __m256d C3 = _mm256_set1_pd(0.1666666666666667);
         const __m256d C4 = _mm256_set1_pd(0.041666666666666664);
         const __m256d C5 = _mm256_set1_pd(0.008333333333333333);
         
         // Range reduction: exp(x) = 2^i * exp(f) where i = floor(x/ln(2)), f = x - i*ln(2)
         __m256d tx = _mm256_mul_pd(x, LOG2E);
         __m256d ti = _mm256_floor_pd(tx);
         __m256d tf = _mm256_sub_pd(x, _mm256_mul_pd(ti, LN2));
         
         // Compute polynomial approximation of exp(f)
         __m256d result = ONE;
         result = _mm256_add_pd(result, _mm256_mul_pd(C1, tf));
         __m256d tf2 = _mm256_mul_pd(tf, tf);
         result = _mm256_add_pd(result, _mm256_mul_pd(C2, tf2));
         __m256d tf3 = _mm256_mul_pd(tf2, tf);
         result = _mm256_add_pd(result, _mm256_mul_pd(C3, tf3));
         __m256d tf4 = _mm256_mul_pd(tf3, tf);
         result = _mm256_add_pd(result, _mm256_mul_pd(C4, tf4));
         __m256d tf5 = _mm256_mul_pd(tf4, tf);
         result = _mm256_add_pd(result, _mm256_mul_pd(C5, tf5));
         
         // Scale by 2^i using scalar operations for reliability
         alignas(32) double int_parts[4];
         alignas(32) double result_array[4];
         _mm256_store_pd(int_parts, ti);
         _mm256_store_pd(result_array, result);
         
         for (int i = 0; i < 4; i++) {
             int exp = (int)int_parts[i] + 1023;  // IEEE-754 exponent bias
             exp = std::min(std::max(exp, 0), 2046);  // Clamp to valid range
             uint64_t bits = (uint64_t)exp << 52;
             double scale;
             memcpy(&scale, &bits, sizeof(double));
             result_array[i] *= scale;
         }
         
         return _mm256_load_pd(result_array);
     }
   
     /**
      * @brief Natural logarithm of a vector using AVX2
      * 
      * @param x Input vector
      * @return Vector containing log(x)
      */
     static inline __m256d log(__m256d x) {
         // Use scalar implementation for now
         alignas(32) double values[4];
         alignas(32) double results[4];
         _mm256_store_pd(values, x);
         
         for (int i = 0; i < 4; i++) {
             results[i] = std::log(values[i]);
         }
         
         return _mm256_load_pd(results);
     }
 
     /**
      * @brief Error function of a vector
      * 
      * @param x Input vector
      * @return Vector containing erf(x)
      */
     static inline __m256d erf(__m256d x) {
         // Use scalar implementation for now
         alignas(32) double values[4];
         alignas(32) double results[4];
         _mm256_store_pd(values, x);
         
         for (int i = 0; i < 4; i++) {
             results[i] = std::erf(values[i]);
         }
         
         return _mm256_load_pd(results);
     }
 
     /**
      * @brief Normal CDF calculation using optimized implementation
      * 
      * @param x Input vector
      * @return Vector containing normalCDF(x)
      */
     static inline __m256d normalCDF(__m256d x) {
         const __m256d HALF = _mm256_set1_pd(0.5);
         const __m256d ONE = _mm256_set1_pd(1.0);
         const __m256d SQRT_2_INV = _mm256_set1_pd(1.0 / std::sqrt(2.0));
         
         // normalCDF(x) = 0.5 * (1 + erf(x/sqrt(2)))
         __m256d scaled_x = _mm256_mul_pd(x, SQRT_2_INV);
         __m256d erf_term = erf(scaled_x);
         __m256d term1 = _mm256_add_pd(ONE, erf_term);
         return _mm256_mul_pd(HALF, term1);
     }
     
     /**
      * @brief Normal PDF calculation
      * 
      * @param x Input vector
      * @return Vector containing normalPDF(x)
      */
     static inline __m256d normalPDF(__m256d x) {
         const __m256d NEG_HALF = _mm256_set1_pd(-0.5);
         const __m256d INV_SQRT_2PI = _mm256_set1_pd(1.0 / std::sqrt(2.0 * M_PI));
         
         // normalPDF(x) = exp(-0.5 * x^2) / sqrt(2*PI)
         __m256d x_squared = _mm256_mul_pd(x, x);
         __m256d exponent = _mm256_mul_pd(NEG_HALF, x_squared);
         __m256d exp_term = exp(exponent);
         
         return _mm256_mul_pd(exp_term, INV_SQRT_2PI);
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
         // Prepare intermediate values
         __m256d vol_sqrt_T = _mm256_mul_pd(vol, _mm256_sqrt_pd(T));
         __m256d half = _mm256_set1_pd(0.5);
         __m256d vol_squared = _mm256_mul_pd(vol, vol);
         __m256d half_vol_squared = _mm256_mul_pd(half, vol_squared);
         
         // Calculate S/K
         __m256d S_div_K = _mm256_div_pd(S, K);
         
         // Calculate log(S/K) using scalar approach
         alignas(32) double s_k_values[4];
         alignas(32) double log_values[4];
         _mm256_store_pd(s_k_values, S_div_K);
         
         for (int i = 0; i < 4; i++) {
             log_values[i] = std::log(s_k_values[i]);
         }
         
         __m256d log_S_div_K = _mm256_load_pd(log_values);
         
         // Calculate drift term
         __m256d r_minus_q = _mm256_sub_pd(r, q);
         __m256d drift = _mm256_add_pd(r_minus_q, half_vol_squared);
         __m256d drift_T = _mm256_mul_pd(drift, T);
         
         // Combine terms and calculate d1
         __m256d numerator = _mm256_add_pd(log_S_div_K, drift_T);
         return _mm256_div_pd(numerator, vol_sqrt_T);
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
         __m256d vol_sqrt_T = _mm256_mul_pd(vol, _mm256_sqrt_pd(T));
         return _mm256_sub_pd(d1, vol_sqrt_T);
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
         __m256d zero = _mm256_setzero_pd();
         
         // Calculate d1 and d2
         __m256d d1 = bsD1(S, K, r, q, vol, T);
         __m256d d2 = bsD2(d1, vol, T);
         
         // Negate d1 and d2 for put calculation
         __m256d neg_d1 = _mm256_sub_pd(zero, d1);
         __m256d neg_d2 = _mm256_sub_pd(zero, d2);
         
         // Calculate N(-d1) and N(-d2)
         __m256d Nd1 = normalCDF(neg_d1);
         __m256d Nd2 = normalCDF(neg_d2);
         
         // Calculate discount factors
         __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
         __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q), T);
         __m256d dr = exp(neg_r_T);
         __m256d dq = exp(neg_q_T);
         
         // K * e^(-rT) * N(-d2)
         __m256d term1 = _mm256_mul_pd(K, dr);
         term1 = _mm256_mul_pd(term1, Nd2);
         
         // S * e^(-qT) * N(-d1)
         __m256d term2 = _mm256_mul_pd(S, dq);
         term2 = _mm256_mul_pd(term2, Nd1);
         
         // put = K * e^(-rT) * N(-d2) - S * e^(-qT) * N(-d1)
         return _mm256_sub_pd(term1, term2);
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
         __m256d zero = _mm256_setzero_pd();
         
         // Calculate d1 and d2
         __m256d d1 = bsD1(S, K, r, q, vol, T);
         __m256d d2 = bsD2(d1, vol, T);
         
         // Calculate N(d1) and N(d2)
         __m256d Nd1 = normalCDF(d1);
         __m256d Nd2 = normalCDF(d2);
         
         // Calculate discount factors
         __m256d neg_r_T = _mm256_mul_pd(_mm256_sub_pd(zero, r), T);
         __m256d neg_q_T = _mm256_mul_pd(_mm256_sub_pd(zero, q), T);
         __m256d dr = exp(neg_r_T);
         __m256d dq = exp(neg_q_T);
         
         // S * e^(-qT) * N(d1)
         __m256d term1 = _mm256_mul_pd(S, dq);
         term1 = _mm256_mul_pd(term1, Nd1);
         
         // K * e^(-rT) * N(d2)
         __m256d term2 = _mm256_mul_pd(K, dr);
         term2 = _mm256_mul_pd(term2, Nd2);
         
         // call = S * e^(-qT) * N(d1) - K * e^(-rT) * N(d2)
         return _mm256_sub_pd(term1, term2);
     }
     
     /**
      * @brief Extract a single element from a vector
      * 
      * @param vec Input vector
      * @param index Index of element to extract (0-3)
      * @return Extracted element
      */
     static inline double extract(__m256d vec, int index) {
         alignas(32) double values[4];
         _mm256_store_pd(values, vec);
         return values[index];
     }
     
     /**
      * @brief Sum all elements in a vector
      * 
      * @param vec Input vector
      * @return Sum of all elements
      */
     static inline double sum(__m256d vec) {
         // Sum the elements using horizontal adds
         __m128d low = _mm256_extractf128_pd(vec, 0);
         __m128d high = _mm256_extractf128_pd(vec, 1);
         __m128d sum_128 = _mm_add_pd(low, high);
         
         // Extract the result
         alignas(16) double sum_values[2];
         _mm_store_pd(sum_values, sum_128);
         return sum_values[0] + sum_values[1];
     }
 };
 
 } // namespace opt
 } // namespace alo
 } // namespace engine
 
 #endif // ENGINE_ALO_OPT_SIMD_H