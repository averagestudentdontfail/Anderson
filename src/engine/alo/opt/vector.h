#ifndef ENGINE_ALO_OPT_VECTOR_H
#define ENGINE_ALO_OPT_VECTOR_H

#include <immintrin.h>  
#include <sleef.h>
#include <array>
#include <cmath>
#include <algorithm>
#include <vector>

namespace engine {
namespace alo {
namespace opt {

// Core SIMD vector types
using SimdVec8f = __m256;   // 8 x float (32-bit)
using SimdVec4d = __m256d;  // 4 x double (64-bit)
    
#ifdef __AVX512F__
using SimdVec16f = __m512;  // 16 x float (32-bit)
using SimdVec8d = __m512d;  // 8 x double (64-bit)
#endif

// SIMD feature detection
enum SIMDSupport {
    NONE = 0,
    SSE2 = 1,
    AVX = 2,
    AVX2 = 3,
    AVX512 = 4
};

// Function declaration for SIMD detection
SIMDSupport detectSIMDSupport();

// Structure for Black-Scholes Greeks
struct BSGreeks {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

/**
 * @brief Vectorized math operations for arrays of data
 * 
 * This class provides optimized vector math operations for processing
 * multiple options simultaneously, with SLEEF-powered SIMD acceleration.
 */
class VectorMath {
public:
    /**
     * @brief Compute exponential of array elements using SLEEF
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void exp(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute natural logarithm of array elements using SLEEF
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void log(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute square root of array elements
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void sqrt(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute error function of array elements using SLEEF
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void erf(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute normal CDF of array elements using SLEEF
     * 
     * Uses the numerically stable approach based on erf/erfc
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void normalCDF(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute normal PDF of array elements using SLEEF
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void normalPDF(const double* x, double* result, size_t size);
    
    /**
     * @brief Multiply arrays element-wise
     * 
     * @param a First input array
     * @param b Second input array
     * @param result Output array
     * @param size Number of elements
     */
    static void multiply(const double* a, const double* b, double* result, size_t size);
    
    /**
     * @brief Add arrays element-wise
     * 
     * @param a First input array
     * @param b Second input array
     * @param result Output array
     * @param size Number of elements
     */
    static void add(const double* a, const double* b, double* result, size_t size);
    
    /**
     * @brief Subtract arrays element-wise
     * 
     * @param a First input array
     * @param b Second input array
     * @param result Output array
     * @param size Number of elements
     */
    static void subtract(const double* a, const double* b, double* result, size_t size);
    
    /**
     * @brief Divide arrays element-wise
     * 
     * @param a First input array
     * @param b Second input array
     * @param result Output array
     * @param size Number of elements
     */
    static void divide(const double* a, const double* b, double* result, size_t size);
    
    /**
     * @brief Compute Black-Scholes d1 for arrays of options
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @param result Output array for d1 values
     * @param size Number of options
     */
    static void bsD1(const double* S, const double* K, const double* r, const double* q, 
                     const double* vol, const double* T, double* result, size_t size);
    
    /**
     * @brief Compute Black-Scholes d2 from d1
     * 
     * @param d1 d1 values
     * @param vol Volatilities
     * @param T Times to maturity
     * @param result Output array for d2 values
     * @param size Number of options
     */
    static void bsD2(const double* d1, const double* vol, const double* T, 
                     double* result, size_t size);
    
    /**
     * @brief Batch compute Black-Scholes put prices with SLEEF acceleration
     * 
     * Uses high-precision implementation with SLEEF
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @param result Output array for put prices
     * @param size Number of options
     */
    static void bsPut(const double* S, const double* K, const double* r, const double* q, 
                     const double* vol, const double* T, double* result, size_t size);
    
    /**
     * @brief Batch compute Black-Scholes call prices with SLEEF acceleration
     * 
     * Uses high-precision implementation with SLEEF
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @param result Output array for call prices
     * @param size Number of options
     */
    static void bsCall(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* result, size_t size);

    /**
     * @brief Optimized fused operation: exp(x)*sqrt(y)
     * 
     * Computes the operation in a single pass to avoid temporary arrays and
     * multiple passes through memory, providing better performance.
     * 
     * @param x First input array
     * @param y Second input array (must be non-negative for sqrt)
     * @param result Output array
     * @param size Number of elements
     */
    static void expMultSqrt(const double* x, const double* y, double* result, size_t size);

    /**
     * @brief Fused operation to calculate d1 and d2 in a single pass
     * 
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @param d1 Output array for d1 values
     * @param d2 Output array for d2 values
     * @param size Number of options
     */
    static void bsD1D2(const double* S, const double* K, const double* r, const double* q, 
                      const double* vol, const double* T, double* d1, double* d2, size_t size);
    
    /**
     * @brief Calculate d1 and d2 in a single fused operation for improved cache efficiency
     *
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @param d1 Output array for d1 values
     * @param d2 Output array for d2 values
     * @param size Number of options
     */
    static void calculateD1D2Fused(const double* S, const double* K, const double* r, const double* q, 
                                 const double* vol, const double* T, double* d1, double* d2, size_t size);

    /**
     * @brief Fused operation for e^(-r*T) * N(x) - common in option pricing
     * 
     * @param x Input x values for N(x)
     * @param r Risk-free rates
     * @param T Times to maturity
     * @param result Output array
     * @param size Number of elements
     */
    static void discountedNormal(const double* x, const double* r, const double* T, 
                                double* result, size_t size);
    
    /**
     * @brief Process batch with manual loop unrolling for better instruction-level parallelism
     *
     * @param S Spot price (constant)
     * @param K Strike prices
     * @param r Risk-free rate (constant)
     * @param q Dividend yield (constant)
     * @param vol Volatility (constant)
     * @param T Time to maturity (constant)
     * @param results Output array
     * @param size Number of options
     */
    static void processUnrolledBatch(const double* S, const double* K, const double* r,
                                   const double* q, const double* vol, const double* T,
                                   double* results, size_t size);
    
    /**
     * @brief Calculate Black-Scholes put prices with Greeks in a single pass
     *
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @param results Output array for option prices and Greeks
     * @param size Number of options
     */
    static void bsPutWithGreeks(const double* S, const double* K, const double* r, const double* q, 
                              const double* vol, const double* T, BSGreeks* results, size_t size);
    
    /**
     * @brief SIMD-accelerated American put option approximation
     *
     * @param S Spot prices
     * @param K Strike prices
     * @param r Risk-free rates
     * @param q Dividend yields
     * @param vol Volatilities
     * @param T Times to maturity
     * @param results Output array
     * @param size Number of options
     */
    static void americanPutApprox(const double* S, const double* K, const double* r,
                                const double* q, const double* vol, const double* T,
                                double* results, size_t size);

    /**
     * @brief Convert array of doubles to array of floats
     *
     * @param input Input array of doubles
     * @return Vector of converted floats
     */
    static std::vector<float> convertToFloat(const std::vector<double>& input);

    /**
     * @brief Convert array of floats to array of doubles
     *
     * @param input Input array of floats
     * @return Vector of converted doubles
     */
    static std::vector<double> convertToDouble(const std::vector<float>& input);
};

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_VECTOR_H