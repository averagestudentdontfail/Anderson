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
};

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_VECTOR_H