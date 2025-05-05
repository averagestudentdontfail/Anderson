#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include <sleef.h>
#include "../opt/simd.h"
#include "../opt/vector.h"
#include "../num/float.h"

using namespace engine::alo::opt;

/**
 * Simple timer class for benchmarking
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::time_point<std::chrono::high_resolution_clock>::now();
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

/**
 * Test single-precision optimized functions
 */
void test_single_precision_functions() {
    constexpr size_t TEST_SIZE = 1000000;
    std::cout << "\n=== Single-Precision Function Test ===\n";
    
    // Create test data
    std::vector<float> x(TEST_SIZE);
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        // Generate values between -10 and 10
        x[i] = 20.0f * (static_cast<float>(i) / TEST_SIZE - 0.5f);
    }
    
    // Clamp extreme values
    for (auto& val : x) {
        val = std::min(std::max(val, -10.0f), 10.0f);
    }
    
    // Test error function
    {
        std::vector<float> result_std(TEST_SIZE);
        std::vector<float> result_fast(TEST_SIZE);
        
        // Standard implementation
        Timer timer;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            result_std[i] = std::erf(x[i]);
        }
        double std_time = timer.elapsed();
        
        // Fast implementation
        timer.reset();
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            result_fast[i] = engine::alo::num::fast_erf(x[i]);
        }
        double fast_time = timer.elapsed();
        
        // Calculate maximum difference
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            double diff = std::abs(result_std[i] - result_fast[i]);
            max_diff = std::max(max_diff, diff);
            
            if (std::abs(result_std[i]) > 1e-6f) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
            }
        }
        
        std::cout << "Error function (erf):\n";
        std::cout << "  Standard time:  " << std_time << " ms\n";
        std::cout << "  Fast time:      " << fast_time << " ms\n";
        std::cout << "  Speedup:        " << std_time / fast_time << "x\n";
        std::cout << "  Max abs diff:   " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff:   " << std::scientific << max_rel_diff << "\n\n";
    }
    
    // Test normal CDF
    {
        std::vector<float> result_std(TEST_SIZE);
        std::vector<float> result_fast(TEST_SIZE);
        
        // Standard implementation
        Timer timer;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            result_std[i] = 0.5f * (1.0f + std::erf(x[i] / std::sqrt(2.0f)));
        }
        double std_time = timer.elapsed();
        
        // Fast implementation
        timer.reset();
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            result_fast[i] = engine::alo::num::fast_normal_cdf(x[i]);
        }
        double fast_time = timer.elapsed();
        
        // Calculate maximum difference
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            double diff = std::abs(result_std[i] - result_fast[i]);
            max_diff = std::max(max_diff, diff);
            
            if (std::abs(result_std[i]) > 1e-6f) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
            }
        }
        
        std::cout << "Normal CDF function:\n";
        std::cout << "  Standard time:  " << std_time << " ms\n";
        std::cout << "  Fast time:      " << fast_time << " ms\n";
        std::cout << "  Speedup:        " << std_time / fast_time << "x\n";
        std::cout << "  Max abs diff:   " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff:   " << std::scientific << max_rel_diff << "\n\n";
    }
    
    // Test SIMD normal CDF (AVX2, 8-wide)
    {
        constexpr size_t VECTOR_SIZE = 8;
        const size_t aligned_size = (TEST_SIZE / VECTOR_SIZE) * VECTOR_SIZE;
        
        std::vector<float> result_std(aligned_size);
        std::vector<float> result_simd(aligned_size);
        
        // Standard implementation
        Timer timer;
        for (size_t i = 0; i < aligned_size; ++i) {
            result_std[i] = 0.5f * (1.0f + std::erf(x[i] / std::sqrt(2.0f)));
        }
        double std_time = timer.elapsed();
        
        // SIMD implementation (manually handling 8 values at a time)
        timer.reset();
        for (size_t i = 0; i < aligned_size; i += VECTOR_SIZE) {
            __m256 x_vec = _mm256_loadu_ps(&x[i]);
            __m256 result_vec = engine::alo::num::simd::normal_cdf_ps(x_vec);
            _mm256_storeu_ps(&result_simd[i], result_vec);
        }
        double simd_time = timer.elapsed();
        
        // Calculate maximum difference
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        for (size_t i = 0; i < aligned_size; ++i) {
            double diff = std::abs(result_std[i] - result_simd[i]);
            max_diff = std::max(max_diff, diff);
            
            if (std::abs(result_std[i]) > 1e-6f) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
            }
        }
        
        std::cout << "SIMD Normal CDF (8-wide):\n";
        std::cout << "  Standard time:  " << std_time << " ms\n";
        std::cout << "  SIMD time:      " << simd_time << " ms\n";
        std::cout << "  Speedup:        " << std_time / simd_time << "x\n";
        std::cout << "  Max abs diff:   " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff:   " << std::scientific << max_rel_diff << "\n\n";
    }
    
    // Test single-precision European Put pricing
    {
        constexpr size_t BS_TEST_SIZE = 100000;
        
        // Generate option parameters
        std::vector<float> S(BS_TEST_SIZE, 100.0f);
        std::vector<float> K(BS_TEST_SIZE);
        std::vector<float> r(BS_TEST_SIZE, 0.05f);
        std::vector<float> q(BS_TEST_SIZE, 0.02f);
        std::vector<float> vol(BS_TEST_SIZE, 0.2f);
        std::vector<float> T(BS_TEST_SIZE, 1.0f);
        
        // Vary strike prices
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            K[i] = 50.0f + 100.0f * static_cast<float>(i) / BS_TEST_SIZE;
        }
        
        std::vector<float> result_std(BS_TEST_SIZE);
        std::vector<float> result_simd(BS_TEST_SIZE);
        
        // Standard implementation
        Timer timer;
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            float d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5f * vol[i] * vol[i]) * T[i]) 
                     / (vol[i] * std::sqrt(T[i]));
            float d2 = d1 - vol[i] * std::sqrt(T[i]);
            
            float nd1 = 0.5f * (1.0f + engine::alo::num::fast_erf(-d1 / std::sqrt(2.0f)));
            float nd2 = 0.5f * (1.0f + engine::alo::num::fast_erf(-d2 / std::sqrt(2.0f)));
            
            result_std[i] = K[i] * std::exp(-r[i] * T[i]) * nd2 - S[i] * std::exp(-q[i] * T[i]) * nd1;
        }
        double std_time = timer.elapsed();
        
        // SIMD implementation
        timer.reset();
        VectorSingle::EuropeanPut(S.data(), K.data(), r.data(), q.data(), vol.data(), T.data(), result_simd.data(), BS_TEST_SIZE);
        double simd_time = timer.elapsed();
        
        // Calculate maximum difference
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        size_t max_diff_index = 0;
        
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            double diff = std::abs(result_std[i] - result_simd[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_index = i;
            }
            
            if (result_std[i] != 0.0f) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
            }
        }
        
        std::cout << "Single-precision European Put pricing:\n";
        std::cout << "  Standard time:  " << std_time << " ms\n";
        std::cout << "  SIMD time:      " << simd_time << " ms\n";
        std::cout << "  Speedup:        " << std_time / simd_time << "x\n";
        std::cout << "  Max abs diff:   " << std::scientific << max_diff 
                  << " at K=" << K[max_diff_index] << "\n";
        std::cout << "  Max rel diff:   " << std::scientific << max_rel_diff << "\n";
        std::cout << "  Options/second: " << BS_TEST_SIZE / (simd_time/1000) << " ops/sec\n\n";
        
        // Print a few sample values
        std::cout << "Sample values (first 5 options):\n";
        std::cout << "  Strike   Standard    SIMD        Diff\n";
        std::cout << "  ------   --------    --------    ---------\n";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  " << std::setw(6) << K[i] << "   " 
                      << std::setw(8) << result_std[i] << "    " 
                      << std::setw(8) << result_simd[i] << "    " 
                      << std::scientific << std::setprecision(3) 
                      << std::abs(result_std[i] - result_simd[i]) << "\n";
        }
        
        // Calculate overall accuracy and throughput
        double mean_rel_diff = 0.0;
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            if (result_std[i] != 0.0f) {
                mean_rel_diff += std::abs(result_std[i] - result_simd[i]) / std::abs(result_std[i]);
            }
        }
        mean_rel_diff /= BS_TEST_SIZE;
        
        std::cout << "  Mean rel diff:   " << std::scientific << mean_rel_diff << "\n";
        std::cout << "  Throughput:      " << std::fixed << BS_TEST_SIZE * 1000 / simd_time 
                  << " options/second\n\n";
    }
}

/**
 * Main function
 */
int main() {
    std::cout << "=== Single-Precision Optimization Test ===\n";
    test_single_precision_functions();
    
    return 0;
}