#ifndef ENGINE_ALO_OPT_VECTOR_H
#define ENGINE_ALO_OPT_VECTOR_H

#include <vector>
#include <cstddef>
#include "../num/float.h"
#include <array>
#include <cmath>
#include <algorithm>

namespace engine {
namespace alo {
namespace opt {

// SIMD support detection
enum SIMDSupport {
    NONE,
    SSE2,
    AVX,
    AVX2,
    AVX512
};

// Declare the detection function
SIMDSupport detectSIMDSupport();

// Generic option batch class
struct OptionBatch {
    std::vector<double> spots;
    std::vector<double> strikes;
    std::vector<double> rates;
    std::vector<double> dividends;
    std::vector<double> vols;
    std::vector<double> times;
    std::vector<float> results;
    
    void resize(size_t size) {
        spots.resize(size);
        strikes.resize(size);
        rates.resize(size);
        dividends.resize(size);
        vols.resize(size);
        times.resize(size);
        results.resize(size);
    }
    
    size_t size() const {
        return spots.size();
    }
};

// Structure of Arrays (SoA) layout for single-precision batch processing
struct OptionBatchSingle {
    std::vector<float> spots;
    std::vector<float> strikes;
    std::vector<float> rates;
    std::vector<float> dividends;
    std::vector<float> vols;
    std::vector<float> times;
    std::vector<float> results;
    
    // Resize all arrays
    void resize(size_t size) {
        spots.resize(size);
        strikes.resize(size);
        rates.resize(size);
        dividends.resize(size);
        vols.resize(size);
        times.resize(size);
        results.resize(size);
    }
    
    // Get current size
    size_t size() const {
        return spots.size();
    }
};

// Structure of Arrays for double-precision batch processing
struct OptionBatchDouble {
    std::vector<double> spots;
    std::vector<double> strikes;
    std::vector<double> rates;
    std::vector<double> dividends;
    std::vector<double> vols;
    std::vector<double> times;
    std::vector<double> results;
    
    void resize(size_t size) {
        spots.resize(size);
        strikes.resize(size);
        rates.resize(size);
        dividends.resize(size);
        vols.resize(size);
        times.resize(size);
        results.resize(size);
    }
    
    size_t size() const {
        return spots.size();
    }
};

// Results structure with Greek for single precision
struct GreekSingle {
    float price;
    float delta;
    float gamma;
    float vega;
    float theta;
    float rho;
};

// Results structure with Greek for double precision
struct GreekDouble {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
};

/**
 * @class VectorDouble
 * @brief Vectorized operations for double-precision arrays
 */
class VectorDouble {
public:
    // Analytical European option pricing
    static void EuropeanPut(const double* S, const double* K, const double* r, const double* q,
                          const double* vol, const double* T, double* result, size_t size);
    
    static void EuropeanCall(const double* S, const double* K, const double* r, const double* q,
                           const double* vol, const double* T, double* result, size_t size);
    
    // Analytical European option pricing with Greek
    static void EuropeanPutGreek(const double* S, const double* K, const double* r, const double* q,
                                const double* vol, const double* T, GreekDouble* results, size_t size);
    
    static void EuropeanCallGreek(const double* S, const double* K, const double* r, const double* q,
                                 const double* vol, const double* T, GreekDouble* results, size_t size);
    
    // Numerical American option pricing
    static void AmericanPut(const double* S, const double* K, const double* r, const double* q,
                          const double* vol, const double* T, double* results, size_t size);
    
    static void AmericanCall(const double* S, const double* K, const double* r, const double* q,
                           const double* vol, const double* T, double* results, size_t size);
    
    // Conversion utilities
    static std::vector<float> convertToSingle(const std::vector<double>& input);
};

/**
 * @class VectorSingle
 * @brief Vectorized operations for single-precision arrays
 */
class VectorSingle {
public:
    // Analytical European option pricing
    static void EuropeanPut(const float* S, const float* K, const float* r, const float* q,
                          const float* vol, const float* T, float* result, size_t size);
    
    static void EuropeanCall(const float* S, const float* K, const float* r, const float* q,
                           const float* vol, const float* T, float* result, size_t size);
    
    // Analytical European option pricing with Greek
    static void EuropeanPutGreek(const float* S, const float* K, const float* r, const float* q,
                                const float* vol, const float* T, GreekSingle* results, size_t size);
    
    static void EuropeanCallGreek(const float* S, const float* K, const float* r, const float* q,
                                 const float* vol, const float* T, GreekSingle* results, size_t size);
    
    // Numerical American option approximation
    static void AmericanPut(const float* S, const float* K, const float* r, const float* q,
                          const float* vol, const float* T, float* results, size_t size);
    
    static void AmericanCall(const float* S, const float* K, const float* r, const float* q,
                           const float* vol, const float* T, float* results, size_t size);
    
    // Conversion utilities
    static std::vector<double> convertToDouble(const std::vector<float>& input);
};

/**
 * @class VectorMath
 * @brief Optimized vector math operations used by pricing functions
 */
class VectorMath {
public:
    // Basic math operations
    static void exp(const double* x, double* result, size_t size);
    static void log(const double* x, double* result, size_t size);
    static void sqrt(const double* x, double* result, size_t size);
    static void erf(const double* x, double* result, size_t size);
    static void normalCDF(const double* x, double* result, size_t size);
    static void normalPDF(const double* x, double* result, size_t size);
    static void multiply(const double* a, const double* b, double* result, size_t size);
    static void add(const double* a, const double* b, double* result, size_t size);
    static void subtract(const double* a, const double* b, double* result, size_t size);
    static void divide(const double* a, const double* b, double* result, size_t size);
    
    // Black-Scholes functions
    static void bsPut(const double* S, const double* K, const double* r, const double* q,
                     const double* vol, const double* T, double* result, size_t size);
    
    static void bsCall(const double* S, const double* K, const double* r, const double* q,
                      const double* vol, const double* T, double* result, size_t size);
    
    // American option approximations
    static void americanPutApprox(const double* S, const double* K, const double* r, const double* q,
                                 const double* vol, const double* T, double* result, size_t size);
    
    static void americanCallApprox(const double* S, const double* K, const double* r, const double* q,
                                  const double* vol, const double* T, double* result, size_t size);
};

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_VECTOR_H