#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include <sleef.h>
#include "../opt/simd.h"
#include "../opt/vector.h"

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
 * Helper function to print detailed comparison of option pricing at specific parameters
 */
void print_detailed_comparison(double S, double K, double r, double q, double vol, double T) {
    // Calculate all intermediate steps with std lib
    double vol_sqrt_T = vol * std::sqrt(T);
    double half_vol_squared = 0.5 * vol * vol;
    double log_S_div_K = std::log(S / K);
    double drift = r - q + half_vol_squared;
    double drift_T = drift * T;
    
    double d1_std = (log_S_div_K + drift_T) / vol_sqrt_T;
    double d2_std = d1_std - vol_sqrt_T;
    
    double Nd1_std = 0.5 * (1.0 + std::erf(-d1_std / std::sqrt(2.0)));
    double Nd2_std = 0.5 * (1.0 + std::erf(-d2_std / std::sqrt(2.0)));
    
    double discount_r = std::exp(-r * T);
    double discount_q = std::exp(-q * T);
    
    double put_std = K * discount_r * Nd2_std - S * discount_q * Nd1_std;
    
    // Calculate using our high precision SLEEF implementation
    double d1_sleef, d2_sleef;
    VectorMath::bsD1D2(&S, &K, &r, &q, &vol, &T, &d1_sleef, &d2_sleef, 1);
    
    double neg_d1 = -d1_sleef;
    double neg_d2 = -d2_sleef;
    
    double Nd1_sleef, Nd2_sleef;
    VectorMath::normalCDFHighPrecision(&neg_d1, &Nd1_sleef, 1);
    VectorMath::normalCDFHighPrecision(&neg_d2, &Nd2_sleef, 1);
    
    double put_sleef;
    VectorMath::bsPutHighPrecision(&S, &K, &r, &q, &vol, &T, &put_sleef, 1);
    
    // Print with higher precision
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "\nOption (S=" << S << ", K=" << K << "):\n";
    std::cout << "d1:      std=" << d1_std << " sleef=" << d1_sleef << " diff=" << std::abs(d1_std - d1_sleef) << "\n";
    std::cout << "d2:      std=" << d2_std << " sleef=" << d2_sleef << " diff=" << std::abs(d2_std - d2_sleef) << "\n";
    std::cout << "N(-d1):  std=" << Nd1_std << " sleef=" << Nd1_sleef << " diff=" << std::abs(Nd1_std - Nd1_sleef) << "\n";
    std::cout << "N(-d2):  std=" << Nd2_std << " sleef=" << Nd2_sleef << " diff=" << std::abs(Nd2_std - Nd2_sleef) << "\n";
    std::cout << "Put:     std=" << put_std << " sleef=" << put_sleef << " diff=" << std::abs(put_std - put_sleef) 
              << " rel_diff=" << (100.0 * std::abs(put_std - put_sleef) / std::abs(put_std)) << "%\n";
}

/**
 * Test normal CDF precision specifically
 */
void test_normal_cdf_precision() {
    std::cout << "\n=== Testing Normal CDF Precision ===\n";
    
    const int TEST_SIZE = 20;
    std::vector<double> x(TEST_SIZE);
    
    // Test range including extreme values
    for (int i = 0; i < TEST_SIZE; i++) {
        x[i] = -5.0 + i * 0.5;  // Range from -5 to +5
    }
    
    std::vector<double> result_std(TEST_SIZE);
    std::vector<double> result_simd(TEST_SIZE);
    std::vector<double> result_orig(TEST_SIZE);
    
    // Calculate with standard library
    for (int i = 0; i < TEST_SIZE; i++) {
        result_std[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0)));
    }
    
    // Calculate with our high precision implementation
    VectorMath::normalCDFHighPrecision(x.data(), result_simd.data(), TEST_SIZE);
    
    // Calculate with original implementation
    VectorMath::normalCDF(x.data(), result_orig.data(), TEST_SIZE);
    
    // Compare results
    std::cout << "     x      |   std::erf   |   High-Prec   |   Original   |   Diff-HP   |   Diff-Orig   \n";
    std::cout << "----------------------------------------------------------------------------------------\n";
    
    double max_diff_hp = 0.0;
    double max_rel_diff_hp = 0.0;
    double max_diff_orig = 0.0;
    double max_rel_diff_orig = 0.0;
    
    for (int i = 0; i < TEST_SIZE; i++) {
        double diff_hp = std::abs(result_std[i] - result_simd[i]);
        double rel_diff_hp = 0.0;
        
        double diff_orig = std::abs(result_std[i] - result_orig[i]);
        double rel_diff_orig = 0.0;
        
        if (std::abs(result_std[i]) > 1e-10) {
            rel_diff_hp = diff_hp / std::abs(result_std[i]);
            rel_diff_orig = diff_orig / std::abs(result_std[i]);
        }
        
        max_diff_hp = std::max(max_diff_hp, diff_hp);
        max_rel_diff_hp = std::max(max_rel_diff_hp, rel_diff_hp);
        max_diff_orig = std::max(max_diff_orig, diff_orig);
        max_rel_diff_orig = std::max(max_rel_diff_orig, rel_diff_orig);
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << std::setw(10) << x[i] << " | " 
                  << std::setw(12) << result_std[i] << " | " 
                  << std::setw(13) << result_simd[i] << " | "
                  << std::setw(13) << result_orig[i] << " | "
                  << std::scientific << std::setprecision(3)
                  << std::setw(11) << diff_hp << " | "
                  << std::setw(13) << diff_orig << "\n";
    }
    
    std::cout << "\nHigh Precision Implementation:\n";
    std::cout << "  Max Difference: " << std::scientific << max_diff_hp << "\n";
    std::cout << "  Max Relative Difference: " << max_rel_diff_hp * 100.0 << "%\n";
    
    std::cout << "\nOriginal Implementation:\n";
    std::cout << "  Max Difference: " << std::scientific << max_diff_orig << "\n";
    std::cout << "  Max Relative Difference: " << max_rel_diff_orig * 100.0 << "%\n";
}

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
            K[i] = 50.0 + 100.0 * static_cast<double>(i) / BS_TEST_SIZE;
        }
        
        std::vector<double> result_std(BS_TEST_SIZE);
        std::vector<double> result_sleef(BS_TEST_SIZE);
        std::vector<double> result_high_precision(BS_TEST_SIZE);
        
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
        
        // SLEEF implementation (original version)
        timer.reset();
        VectorMath::bsPut(S.data(), K.data(), r.data(), q.data(), vol.data(), T.data(), result_sleef.data(), BS_TEST_SIZE);
        double sleef_time = timer.elapsed();
        
        // SLEEF implementation (high precision version)
        timer.reset();
        VectorMath::bsPutHighPrecision(S.data(), K.data(), r.data(), q.data(), vol.data(), T.data(), 
                                     result_high_precision.data(), BS_TEST_SIZE);
        double high_precision_time = timer.elapsed();
        
        // Calculate maximum difference for original SLEEF
        double max_diff = 0.0;
        double max_rel_diff = 0.0;
        size_t max_diff_index = 0;
        
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            double diff = std::abs(result_std[i] - result_sleef[i]);
            if (diff > max_diff) {
                max_diff = diff;
                max_diff_index = i;
            }
            
            if (result_std[i] != 0.0) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff = std::max(max_rel_diff, rel_diff);
            }
        }
        
        // Calculate maximum difference for high precision SLEEF
        double max_diff_hp = 0.0;
        double max_rel_diff_hp = 0.0;
        size_t max_diff_index_hp = 0;
        
        for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
            double diff = std::abs(result_std[i] - result_high_precision[i]);
            if (diff > max_diff_hp) {
                max_diff_hp = diff;
                max_diff_index_hp = i;
            }
            
            if (result_std[i] != 0.0) {
                double rel_diff = diff / std::abs(result_std[i]);
                max_rel_diff_hp = std::max(max_rel_diff_hp, rel_diff);
            }
        }
        
        std::cout << "Black-Scholes Put pricing:\n";
        std::cout << "  Standard time:             " << std_time << " ms\n";
        std::cout << "  SLEEF time:                " << sleef_time << " ms\n";
        std::cout << "  High Precision SLEEF time: " << high_precision_time << " ms\n";
        std::cout << "  Speedup (SLEEF):           " << std_time / sleef_time << "x\n";
        std::cout << "  Speedup (High Precision):  " << std_time / high_precision_time << "x\n";
        std::cout << "  Max abs diff (SLEEF):      " << std::scientific << max_diff 
                  << " at K=" << K[max_diff_index] << "\n";
        std::cout << "  Max rel diff (SLEEF):      " << std::scientific << max_rel_diff << "\n";
        std::cout << "  Max abs diff (High Prec):  " << std::scientific << max_diff_hp 
                  << " at K=" << K[max_diff_index_hp] << "\n";
        std::cout << "  Max rel diff (High Prec):  " << std::scientific << max_rel_diff_hp << "\n\n";
        
        // Print where max error occurs
        std::cout << "Maximum error details (SLEEF):\n";
        print_detailed_comparison(S[max_diff_index], K[max_diff_index], r[max_diff_index], 
                                q[max_diff_index], vol[max_diff_index], T[max_diff_index]);

        // Print where max error occurs (High Precision)
        std::cout << "\nMaximum error details (High Precision):\n";
        print_detailed_comparison(S[max_diff_index_hp], K[max_diff_index_hp], r[max_diff_index_hp], 
                                q[max_diff_index_hp], vol[max_diff_index_hp], T[max_diff_index_hp]);
        
        // Print a few sample values
        std::cout << "\nSample values (first 5 options):\n";
        std::cout << "  Strike   Standard    SLEEF       HighPrec    Diff_SLEEF  Diff_HighPrec\n";
        std::cout << "  ------   --------    --------    --------    ---------   ------------\n";
        for (size_t i = 0; i < 5; ++i) {
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "  " << std::setw(6) << K[i] << "   " 
                      << std::setw(8) << result_std[i] << "    " 
                      << std::setw(8) << result_sleef[i] << "    " 
                      << std::setw(8) << result_high_precision[i] << "    " 
                      << std::scientific << std::setprecision(3) 
                      << std::abs(result_std[i] - result_sleef[i]) << "    "
                      << std::abs(result_std[i] - result_high_precision[i]) << "\n";
        }
    }
}

/**
 * Debug function to identify sources of Black-Scholes calculation discrepancies
 */
void debug_blackscholes_accuracy() {
    std::cout << "\n=== Black-Scholes Accuracy Debug ===\n";
    
    // Create option parameters where we saw issues
    constexpr size_t TEST_SIZE = 1000;
    
    std::vector<double> S(TEST_SIZE, 100.0);
    std::vector<double> K(TEST_SIZE);
    std::vector<double> r(TEST_SIZE, 0.05);
    std::vector<double> q(TEST_SIZE, 0.02);
    std::vector<double> vol(TEST_SIZE, 0.2);
    std::vector<double> T(TEST_SIZE, 1.0);
    
    // Create a wider range of strike prices to test
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        K[i] = 50.0 + 100.0 * (static_cast<double>(i) / TEST_SIZE);
    }
    
    // Allocate result arrays
    std::vector<double> d1_std(TEST_SIZE);
    std::vector<double> d2_std(TEST_SIZE);
    std::vector<double> Nd1_std(TEST_SIZE);
    std::vector<double> Nd2_std(TEST_SIZE);
    std::vector<double> put_std(TEST_SIZE);
    
    std::vector<double> d1_sleef(TEST_SIZE);
    std::vector<double> d2_sleef(TEST_SIZE);
    std::vector<double> Nd1_sleef(TEST_SIZE);
    std::vector<double> Nd2_sleef(TEST_SIZE);
    std::vector<double> put_sleef(TEST_SIZE);
    std::vector<double> put_high_precision(TEST_SIZE);
    
    // Calculate using standard library
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        double vol_sqrt_T = vol[i] * std::sqrt(T[i]);
        double half_vol_squared = 0.5 * vol[i] * vol[i];
        double log_S_div_K = std::log(S[i] / K[i]);
        double drift = r[i] - q[i] + half_vol_squared;
        double drift_T = drift * T[i];
        
        d1_std[i] = (log_S_div_K + drift_T) / vol_sqrt_T;
        d2_std[i] = d1_std[i] - vol_sqrt_T;
        
        Nd1_std[i] = 0.5 * (1.0 + std::erf(-d1_std[i] / std::sqrt(2.0)));
        Nd2_std[i] = 0.5 * (1.0 + std::erf(-d2_std[i] / std::sqrt(2.0)));
        
        double discount_r = std::exp(-r[i] * T[i]);
        double discount_q = std::exp(-q[i] * T[i]);
        
        put_std[i] = K[i] * discount_r * Nd2_std[i] - S[i] * discount_q * Nd1_std[i];
    }
    
    // Calculate using SLEEF
    VectorMath::bsD1D2(S.data(), K.data(), r.data(), q.data(), vol.data(), T.data(), 
                     d1_sleef.data(), d2_sleef.data(), TEST_SIZE);
    
    // Prepare negative d1 and d2 for N(-d1) and N(-d2)
    std::vector<double> neg_d1(TEST_SIZE);
    std::vector<double> neg_d2(TEST_SIZE);
    
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        neg_d1[i] = -d1_sleef[i];
        neg_d2[i] = -d2_sleef[i];
    }
    
    // Calculate N(-d1) and N(-d2) using normalCDFHighPrecision
    VectorMath::normalCDFHighPrecision(neg_d1.data(), Nd1_sleef.data(), TEST_SIZE);
    VectorMath::normalCDFHighPrecision(neg_d2.data(), Nd2_sleef.data(), TEST_SIZE);
    
    // Calculate put prices using both methods
    VectorMath::bsPut(S.data(), K.data(), r.data(), q.data(), vol.data(), T.data(), 
                    put_sleef.data(), TEST_SIZE);
    
    VectorMath::bsPutHighPrecision(S.data(), K.data(), r.data(), q.data(), vol.data(), T.data(), 
                                 put_high_precision.data(), TEST_SIZE);
    
    // Find maximum differences
    double max_diff_d1 = 0.0;
    double max_diff_d2 = 0.0;
    double max_diff_Nd1 = 0.0;
    double max_diff_Nd2 = 0.0;
    double max_diff_put = 0.0;
    double max_diff_put_hp = 0.0;
    
    size_t max_idx_d1 = 0;
    size_t max_idx_d2 = 0;
    size_t max_idx_Nd1 = 0;
    size_t max_idx_Nd2 = 0;
    size_t max_idx_put = 0;
    size_t max_idx_put_hp = 0;
    
    for (size_t i = 0; i < TEST_SIZE; ++i) {
        double diff_d1 = std::abs(d1_std[i] - d1_sleef[i]);
        if (diff_d1 > max_diff_d1) {
            max_diff_d1 = diff_d1;
            max_idx_d1 = i;
        }
        
        double diff_d2 = std::abs(d2_std[i] - d2_sleef[i]);
        if (diff_d2 > max_diff_d2) {
            max_diff_d2 = diff_d2;
            max_idx_d2 = i;
        }
        
        double diff_Nd1 = std::abs(Nd1_std[i] - Nd1_sleef[i]);
        if (diff_Nd1 > max_diff_Nd1) {
            max_diff_Nd1 = diff_Nd1;
            max_idx_Nd1 = i;
        }
        
        double diff_Nd2 = std::abs(Nd2_std[i] - Nd2_sleef[i]);
        if (diff_Nd2 > max_diff_Nd2) {
            max_diff_Nd2 = diff_Nd2;
            max_idx_Nd2 = i;
        }
        
        double diff_put = std::abs(put_std[i] - put_sleef[i]);
        if (diff_put > max_diff_put) {
            max_diff_put = diff_put;
            max_idx_put = i;
        }
        
        double diff_put_hp = std::abs(put_std[i] - put_high_precision[i]);
        if (diff_put_hp > max_diff_put_hp) {
            max_diff_put_hp = diff_put_hp;
            max_idx_put_hp = i;
        }
    }
    
    // Print maximum differences with their locations
    std::cout << "Maximum Differences:\n";
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "  d1: " << max_diff_d1 << " at K=" << K[max_idx_d1] << "\n";
    std::cout << "  d2: " << max_diff_d2 << " at K=" << K[max_idx_d2] << "\n";
    std::cout << "  N(-d1): " << max_diff_Nd1 << " at K=" << K[max_idx_Nd1] << "\n";
    std::cout << "  N(-d2): " << max_diff_Nd2 << " at K=" << K[max_idx_Nd2] << "\n";
    std::cout << "  Put (Regular): " << max_diff_put << " at K=" << K[max_idx_put] << "\n";
    std::cout << "  Put (High Precision): " << max_diff_put_hp << " at K=" << K[max_idx_put_hp] << "\n\n";
    
    // Print detailed analysis for the points with largest differences
    std::cout << "Detailed Analysis of Maximum Put Difference (Regular):\n";
    print_detailed_comparison(S[max_idx_put], K[max_idx_put], r[max_idx_put], 
                            q[max_idx_put], vol[max_idx_put], T[max_idx_put]);
    
    std::cout << "\nDetailed Analysis of Maximum Put Difference (High Precision):\n";
    print_detailed_comparison(S[max_idx_put_hp], K[max_idx_put_hp], r[max_idx_put_hp], 
                            q[max_idx_put_hp], vol[max_idx_put_hp], T[max_idx_put_hp]);
    
    // Print detailed table for a sample of points to identify patterns
    std::cout << "\nDetailed Comparison (Sample Points):\n";
    std::cout << "Strike    d1_std    d1_sleef    d2_std    d2_sleef    N(-d1)_std    N(-d1)_sleef    N(-d2)_std    N(-d2)_sleef    Put_std    Put_sleef    Put_high    Diff_old    Diff_new\n";
    std::cout << "------    ------    --------    ------    --------    ----------    ------------    ----------    ------------    -------    ---------    --------    --------    --------\n";
    
    // Print samples at regular intervals
    int interval = TEST_SIZE / 10;
    for (size_t i = 0; i < TEST_SIZE; i += interval) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(6) << K[i] << "    "
                  << std::setw(6) << d1_std[i] << "    "
                  << std::setw(8) << d1_sleef[i] << "    "
                  << std::setw(6) << d2_std[i] << "    "
                  << std::setw(8) << d2_sleef[i] << "    "
                  << std::setw(10) << Nd1_std[i] << "    "
                  << std::setw(12) << Nd1_sleef[i] << "    "
                  << std::setw(10) << Nd2_std[i] << "    "
                  << std::setw(12) << Nd2_sleef[i] << "    "
                  << std::setw(7) << put_std[i] << "    "
                  << std::setw(9) << put_sleef[i] << "    "
                  << std::setw(8) << put_high_precision[i] << "    "
                  << std::setw(8) << std::abs(put_std[i] - put_sleef[i]) << "    "
                  << std::setw(8) << std::abs(put_std[i] - put_high_precision[i]) << "\n";
    }
}

/**
 * Main function
 */
int main() {
    test_sleef_functions();
    test_normal_cdf_precision();
    debug_blackscholes_accuracy();
    return 0;
}