#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include <sleef.h>
#include "../engine/alo/opt/simd.h"
#include "../engine/alo/opt/vector.h"

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
        start_ = std::chrono::high_resolution_clock::now();
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

/**
 * Test each SLEEF-powered function against its standard math library equivalent
 */
void test_sleef_functions() {
    constexpr size_t TEST_SIZE = 1000000;
    std::cout << "\n=== SLEEF Integration Test ===\n";
    
    // Print SIMD support
    std::cout << "SIMD Support:\n";
    std::cout << "  AVX2:   " << (SimdDetect::hasAVX2() ? "Yes" : "No") << "\n";
    std::cout << "  AVX512: " << (SimdDetect::hasAVX512() ? "Yes" : "No") << "\n\n";
    
    if (!SimdDetect::hasAVX2()) {
        std::cout << "SLEEF test skipped (AVX2 not available)\n";
        return;
    }
    
    // Create test data
    std::vector<double> x(TEST_SIZE);
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        // Generate values between -10 and 10
        x[i] = 20.0 * (static_cast<double>(i) / TEST_SIZE - 0.5);
    }
    
    // Test exponential function
    {
        std::vector<double> result_std(TEST_SIZE);
        std::vector<double> result_sleef(TEST_SIZE);
        
        // Standard implementation
        Timer timer;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            // Clamp extreme values to avoid overflow
            double val = std::min(std::max(x[i], -700.0), 700.0);
            result_std[i] = std::exp(val);
        }
        double std_time = timer.elapsed();
        
        // SLEEF implementation
        timer.reset();
        VectorMath::exp(x.data(), result_sleef.data(), TEST_SIZE);
        double sleef_time = timer.elapsed();
        
        // Calculate maximum difference
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            double diff = std::abs(result_std[i] - result_sleef[i]);
            max_diff = std::max(max_diff, diff);
            
            if (result_std[i] != 0.0) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
            }
        }
        
        std::cout << "Exponential function (exp):\n";
        std::cout << "  Standard time:  " << std_time << " ms\n";
        std::cout << "  SLEEF time:     " << sleef_time << " ms\n";
        std::cout << "  Speedup:        " << std_time / sleef_time << "x\n";
        std::cout << "  Max abs diff:   " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff:   " << std::scientific << max_rel_diff << "\n\n";
    }
    
    // Test logarithm function
    {
        std::vector<double> x_pos(TEST_SIZE);
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            // Generate positive values
            x_pos[i] = 0.0001 + 10.0 * static_cast<double>(i) / TEST_SIZE;
        }
        
        std::vector<double> result_std(TEST_SIZE);
        std::vector<double> result_sleef(TEST_SIZE);
        
        // Standard implementation
        Timer timer;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            result_std[i] = std::log(x_pos[i]);
        }
        double std_time = timer.elapsed();
        
        // SLEEF implementation
        timer.reset();
        VectorMath::log(x_pos.data(), result_sleef.data(), TEST_SIZE);
        double sleef_time = timer.elapsed();
        
        // Calculate maximum difference
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        for (size_t i = 0; i < TEST_SIZE; ++i) {
            double diff = std::abs(result_std[i] - result_sleef[i]);
            max_diff = std::max(max_diff, diff);
            
            if (result_std[i] != 0.0) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
            }
        }
        
        std::cout << "Logarithm function (log):\n";
        std::cout << "  Standard time:  " << std_time << " ms\n";
        std::cout << "  SLEEF time:     " << sleef_time << " ms\n";
        std::cout << "  Speedup:        " << std_time / sleef_time << "x\n";
        std::cout << "  Max abs diff:   " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff:   " << std::scientific << max_rel_diff << "\n\n";
    }
    
    // Test Black-Scholes Put pricing
    {
        constexpr size_t BS_TEST_SIZE = 100000;
        
        // Generate random option parameters
        std::vector<double> S(BS_TEST_SIZE, 100.0);
        std::vector<double> K(BS_TEST_SIZE);
        std::vector<double> r(BS_TEST_SIZE, 0.05);
        std::vector<double> q(BS_TEST_SIZE, 0.02);
        std::vector<double> vol(BS_TEST_SIZE, 0.2);
        std::vector<double> T(BS_TEST_SIZE, 1.0);
        
        // Vary strike prices
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            K[i] = 70.0 + 60.0 * static_cast<double>(i) / BS_TEST_SIZE;
        }
        
        std::vector<double> result_std(BS_TEST_SIZE);
        std::vector<double> result_sleef(BS_TEST_SIZE);
        
        // Standard implementation
        Timer timer;
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) 
                      / (vol[i] * std::sqrt(T[i]));
            double d2 = d1 - vol[i] * std::sqrt(T[i]);
            
            double nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
            double nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
            
            result_std[i] = K[i] * std::exp(-r[i] * T[i]) * nd2 - S[i] * std::exp(-q[i] * T[i]) * nd1;
        }
        double std_time = timer.elapsed();
        
        // SLEEF implementation
        timer.reset();
        VectorMath::bsPut(S.data(), K.data(), r.data(), q.data(), vol.data(), T.data(), result_sleef.data(), BS_TEST_SIZE);
        double sleef_time = timer.elapsed();
        
        // Calculate maximum difference
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            double diff = std::abs(result_std[i] - result_sleef[i]);
            max_diff = std::max(max_diff, diff);
            
            if (result_std[i] != 0.0) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
            }
        }
        
        std::cout << "Black-Scholes Put pricing:\n";
        std::cout << "  Standard time:  " << std_time << " ms\n";
        std::cout << "  SLEEF time:     " << sleef_time << " ms\n";
        std::cout << "  Speedup:        " << std_time / sleef_time << "x\n";
        std::cout << "  Max abs diff:   " << std::scientific << max_diff << "\n";
        std::cout << "  Max rel diff:   " << std::scientific << max_rel_diff << "\n\n";
        
        // Print a few sample values
        std::cout << "Sample values (first 5 options):\n";
        std::cout << "  Strike   Standard    SLEEF       Diff\n";
        std::cout << "  ------   --------    --------    --------\n";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  " << std::setw(6) << K[i] << "   " 
                      << std::setw(8) << result_std[i] << "    " 
                      << std::setw(8) << result_sleef[i] << "    " 
                      << std::scientific << std::setprecision(3) 
                      << std::abs(result_std[i] - result_sleef[i]) << "\n";
        }
    }
}

/**
 * Main function
 */
int main() {
    test_sleef_functions();
    return 0;
}