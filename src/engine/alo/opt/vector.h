#ifndef ENGINE_ALO_OPT_VECTOR_H
#define ENGINE_ALO_OPT_VECTOR_H

#include "simd.h"
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
 * multiple options simultaneously, with SIMD acceleration when available.
 */
class VectorMath {
public:
    /**
     * @brief Compute exponential of array elements
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void exp(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute natural logarithm of array elements
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
     * @brief Compute error function of array elements
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void erf(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute normal CDF of array elements
     * 
     * @param x Input array
     * @param result Output array
     * @param size Number of elements
     */
    static void normalCDF(const double* x, double* result, size_t size);
    
    /**
     * @brief Compute normal PDF of array elements
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
     * @brief Batch compute Black-Scholes put prices
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
     * @brief Batch compute Black-Scholes call prices
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
    
private:
    // SIMD helper functions
    static __m256d simd_exp(__m256d x);
    static __m256d simd_log(__m256d x);
    static __m256d simd_erf(__m256d x);
    static __m256d simd_normalCDF(__m256d x);
    static __m256d simd_normalPDF(__m256d x);
};

/**
 * @brief SIMD floating-point operations on vectors
 * 
 * This class provides templated SIMD operations for processing
 * fixed-size vectors of floating-point values.
 * 
 * @tparam T Floating-point type (float or double)
 * @tparam N Vector size
 */
template <typename T, size_t N>
class SimdVec {
public:
    /**
     * @brief Constructor from array
     * 
     * @param data Input array
     */
    explicit SimdVec(const T* data) {
        std::copy(data, data + N, data_);
    }
    
    /**
     * @brief Constructor from initializer list
     * 
     * @param values Initializer list of values
     */
    SimdVec(std::initializer_list<T> values) {
        size_t i = 0;
        for (auto val : values) {
            if (i < N) {
                data_[i++] = val;
            } else {
                break;
            }
        }
        for (; i < N; ++i) {
            data_[i] = T(0);
        }
    }
    
    /**
     * @brief Constructor with single value for all elements
     * 
     * @param value Value to set all elements to
     */
    explicit SimdVec(T value) {
        std::fill(data_, data_ + N, value);
    }
    
    /**
     * @brief Default constructor (zero initialization)
     */
    SimdVec() {
        std::fill(data_, data_ + N, T(0));
    }
    
    /**
     * @brief Add another vector
     * 
     * @param other Vector to add
     * @return Result of addition
     */
    SimdVec operator+(const SimdVec& other) const {
        SimdVec result;
        for (size_t i = 0; i < N; ++i) {
            result.data_[i] = data_[i] + other.data_[i];
        }
        return result;
    }
    
    /**
     * @brief Subtract another vector
     * 
     * @param other Vector to subtract
     * @return Result of subtraction
     */
    SimdVec operator-(const SimdVec& other) const {
        SimdVec result;
        for (size_t i = 0; i < N; ++i) {
            result.data_[i] = data_[i] - other.data_[i];
        }
        return result;
    }
    
    /**
     * @brief Multiply by another vector element-wise
     * 
     * @param other Vector to multiply by
     * @return Result of multiplication
     */
    SimdVec operator*(const SimdVec& other) const {
        SimdVec result;
        for (size_t i = 0; i < N; ++i) {
            result.data_[i] = data_[i] * other.data_[i];
        }
        return result;
    }
    
    /**
     * @brief Divide by another vector element-wise
     * 
     * @param other Vector to divide by
     * @return Result of division
     */
    SimdVec operator/(const SimdVec& other) const {
        SimdVec result;
        for (size_t i = 0; i < N; ++i) {
            result.data_[i] = data_[i] / other.data_[i];
        }
        return result;
    }
    
    /**
     * @brief Get vector size
     * 
     * @return Number of elements
     */
    static constexpr size_t size() {
        return N;
    }
    
    /**
     * @brief Access element
     * 
     * @param i Element index
     * @return Reference to element
     */
    T& operator[](size_t i) {
        return data_[i];
    }
    
    /**
     * @brief Access element (const version)
     * 
     * @param i Element index
     * @return Const reference to element
     */
    const T& operator[](size_t i) const {
        return data_[i];
    }
    
    /**
     * @brief Copy to array
     * 
     * @param dest Destination array
     */
    void copyTo(T* dest) const {
        std::copy(data_, data_ + N, dest);
    }
    
    /**
     * @brief Get data pointer
     * 
     * @return Pointer to internal data
     */
    const T* data() const {
        return data_;
    }
    
    /**
     * @brief Get data pointer
     * 
     * @return Pointer to internal data
     */
    T* data() {
        return data_;
    }
    
private:
    alignas(32) T data_[N];
};

/**
 * @brief Specialized SIMD vector operations for 4 doubles
 * 
 * This class provides optimized SIMD operations for processing
 * vectors of 4 double values using AVX2 when available.
 */
class SimdVec4d : public SimdVec<double, 4> {
public:
    using SimdVec<double, 4>::SimdVec;
    
    /**
     * @brief Add another vector with SIMD
     * 
     * @param other Vector to add
     * @return Result of addition
     */
    SimdVec4d operator+(const SimdVec4d& other) const {
        if (SimdDetect::hasAVX2()) {
            SimdVec4d result;
            __m256d a = _mm256_loadu_pd(data());
            __m256d b = _mm256_loadu_pd(other.data());
            __m256d sum = _mm256_add_pd(a, b);
            _mm256_storeu_pd(result.data(), sum);
            return result;
        } else {
            // Fixed: Proper conversion from base class result to derived class
            SimdVec4d result;
            auto baseResult = SimdVec<double, 4>::operator+(other);
            std::copy(baseResult.data(), baseResult.data() + 4, result.data());
            return result;
        }
    }
    
    /**
     * @brief Subtract another vector with SIMD
     * 
     * @param other Vector to subtract
     * @return Result of subtraction
     */
    SimdVec4d operator-(const SimdVec4d& other) const {
        if (SimdDetect::hasAVX2()) {
            SimdVec4d result;
            __m256d a = _mm256_loadu_pd(data());
            __m256d b = _mm256_loadu_pd(other.data());
            __m256d diff = _mm256_sub_pd(a, b);
            _mm256_storeu_pd(result.data(), diff);
            return result;
        } else {
            // Fixed: Proper conversion from base class result to derived class
            SimdVec4d result;
            auto baseResult = SimdVec<double, 4>::operator-(other);
            std::copy(baseResult.data(), baseResult.data() + 4, result.data());
            return result;
        }
    }
    
    /**
     * @brief Multiply by another vector with SIMD
     * 
     * @param other Vector to multiply by
     * @return Result of multiplication
     */
    SimdVec4d operator*(const SimdVec4d& other) const {
        if (SimdDetect::hasAVX2()) {
            SimdVec4d result;
            __m256d a = _mm256_loadu_pd(data());
            __m256d b = _mm256_loadu_pd(other.data());
            __m256d prod = _mm256_mul_pd(a, b);
            _mm256_storeu_pd(result.data(), prod);
            return result;
        } else {
            // Fixed: Proper conversion from base class result to derived class
            SimdVec4d result;
            auto baseResult = SimdVec<double, 4>::operator*(other);
            std::copy(baseResult.data(), baseResult.data() + 4, result.data());
            return result;
        }
    }
    
    /**
     * @brief Divide by another vector with SIMD
     * 
     * @param other Vector to divide by
     * @return Result of division
     */
    SimdVec4d operator/(const SimdVec4d& other) const {
        if (SimdDetect::hasAVX2()) {
            SimdVec4d result;
            __m256d a = _mm256_loadu_pd(data());
            __m256d b = _mm256_loadu_pd(other.data());
            __m256d quot = _mm256_div_pd(a, b);
            _mm256_storeu_pd(result.data(), quot);
            return result;
        } else {
            // Fixed: Proper conversion from base class result to derived class
            SimdVec4d result;
            auto baseResult = SimdVec<double, 4>::operator/(other);
            std::copy(baseResult.data(), baseResult.data() + 4, result.data());
            return result;
        }
    }
    
    /**
     * @brief Square root with SIMD
     * 
     * @return Vector of square roots
     */
    SimdVec4d sqrt() const {
        if (SimdDetect::hasAVX2()) {
            SimdVec4d result;
            __m256d a = _mm256_loadu_pd(data());
            __m256d sqrt_a = _mm256_sqrt_pd(a);
            _mm256_storeu_pd(result.data(), sqrt_a);
            return result;
        } else {
            SimdVec4d result;
            for (size_t i = 0; i < 4; ++i) {
                result[i] = std::sqrt((*this)[i]);
            }
            return result;
        }
    }
    
    /**
     * @brief Maximum with SIMD
     * 
     * @param other Vector to compare with
     * @return Vector of maximum values
     */
    SimdVec4d max(const SimdVec4d& other) const {
        if (SimdDetect::hasAVX2()) {
            SimdVec4d result;
            __m256d a = _mm256_loadu_pd(data());
            __m256d b = _mm256_loadu_pd(other.data());
            __m256d max_ab = _mm256_max_pd(a, b);
            _mm256_storeu_pd(result.data(), max_ab);
            return result;
        } else {
            SimdVec4d result;
            for (size_t i = 0; i < 4; ++i) {
                result[i] = std::max((*this)[i], other[i]);
            }
            return result;
        }
    }
    
    /**
     * @brief Minimum with SIMD
     * 
     * @param other Vector to compare with
     * @return Vector of minimum values
     */
    SimdVec4d min(const SimdVec4d& other) const {
        if (SimdDetect::hasAVX2()) {
            SimdVec4d result;
            __m256d a = _mm256_loadu_pd(data());
            __m256d b = _mm256_loadu_pd(other.data());
            __m256d min_ab = _mm256_min_pd(a, b);
            _mm256_storeu_pd(result.data(), min_ab);
            return result;
        } else {
            SimdVec4d result;
            for (size_t i = 0; i < 4; ++i) {
                result[i] = std::min((*this)[i], other[i]);
            }
            return result;
        }
    }
};

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_VECTOR_H