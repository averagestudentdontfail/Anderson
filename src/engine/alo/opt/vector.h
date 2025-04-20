#ifndef ENGINE_ALO_OPT_VECTOR_H
#define ENGINE_ALO_OPT_VECTOR_H

#include "simd.h"
#include <sleef.h>
#include <immintrin.h>  
#include <array>
#include <cmath>
#include <algorithm>

namespace engine {
namespace alo {
namespace opt {

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
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void normalCDF(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute high-precision normal CDF of array elements
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void normalCDFHighPrecision(const double* x, double* result, size_t size);
    
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
     * @brief High-precision Black-Scholes put calculation
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
    static void bsPutHighPrecision(const double* S, const double* K, const double* r, 
                                  const double* q, const double* vol, const double* T, 
                                  double* result, size_t size);

    /**
     * @brief High-precision Black-Scholes call calculation
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
    static void bsCallHighPrecision(const double* S, const double* K, const double* r, 
                                   const double* q, const double* vol, const double* T, 
                                   double* result, size_t size);

    /**
     * @brief Generic binary operator template for fused operations
     * 
     * This template allows implementing other binary operations with similar structure
     * 
     * @tparam Op1 First operation
     * @tparam Op2 Second operation 
     * @tparam CombineOp Operation to combine results
     * @param a First input array
     * @param b Second input array
     * @param result Output array
     * @param size Number of elements
     * @param op1 First operation to apply
     * @param op2 Second operation to apply
     * @param combineOp Operation to combine results
     */
    template<typename Op1, typename Op2, typename CombineOp>
    static void binaryFusedOp(const double* a, const double* b, double* result, size_t size,
                          Op1 op1, Op2 op2, CombineOp combineOp);
};

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_VECTOR_H