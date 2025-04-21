#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>
#include <immintrin.h>
#include <sleef.h>
#include <numeric> // For std::accumulate
#include "../opt/simd.h" 
#include "../opt/vector.h" 

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

// --- Previous benchmark and precision tests (can be kept or removed as needed) ---

void benchmark_sleef_functions() {
    // (Implementation from previous step - kept for context but output might be long)
    constexpr size_t TEST_SIZE = 1000000; 
    static_assert(TEST_SIZE % 4 == 0, "TEST_SIZE must be a multiple of 4 for AVX2 benchmarks without scalar fallback.");
    std::cout << "\n=== SLEEF Function Benchmarks (AVX2 - Double Precision) ===\n"; // ... rest of function ...
     // --- Benchmark Exponential function (exp) ---
    {
        std::cout << "--- Exponential function (exp) ---\n";
        std::vector<double> result_std(TEST_SIZE);
        std::vector<double> result_sleef_u10(TEST_SIZE);
        std::vector<double> x_exp(TEST_SIZE);
         for (size_t i = 0; i < TEST_SIZE; ++i) {
             double val_exp_erf = 20.0 * (static_cast<double>(i) / TEST_SIZE - 0.5);
             x_exp[i] = std::min(std::max(val_exp_erf, -700.0), 700.0); 
         }

        Timer timer;
        for (size_t i = 0; i < TEST_SIZE; ++i) { result_std[i] = std::exp(x_exp[i]); }
        double std_time = timer.elapsed();
        std::cout << "  Standard time:  " << std::fixed << std::setprecision(6) << std_time << " ms\n";
        
        timer.reset();
        for (size_t i = 0; i < TEST_SIZE; i += 4) {
            __m256d vec = _mm256_loadu_pd(x_exp.data() + i);
            __m256d res = Sleef_expd4_u10avx2(vec); 
            _mm256_storeu_pd(result_sleef_u10.data() + i, res);
        }
        double sleef_u10_time = timer.elapsed();
        std::cout << "  SLEEF u10 time: " << std::fixed << std::setprecision(6) << sleef_u10_time << " ms";
        if (sleef_u10_time > 1e-9) std::cout << " (Speedup vs std: " << std::fixed << std::setprecision(3) << std_time / sleef_u10_time << "x)\n\n";
        else std::cout << " (Speedup vs std: N/A)\n\n";
    }
     // --- Benchmark Logarithm function (log) ---
    {
        std::cout << "--- Logarithm function (log) ---\n";
         std::vector<double> result_std(TEST_SIZE);
         std::vector<double> result_sleef_u10(TEST_SIZE);
         std::vector<double> x_log(TEST_SIZE);
         for (size_t i = 0; i < TEST_SIZE; ++i) {
             x_log[i] = 0.0001 + 10.0 * static_cast<double>(i) / TEST_SIZE;
         }

        Timer timer;
        for (size_t i = 0; i < TEST_SIZE; ++i) { result_std[i] = std::log(x_log[i]); }
        double std_time = timer.elapsed();
        std::cout << "  Standard time:  " << std::fixed << std::setprecision(6) << std_time << " ms\n";
        
        timer.reset();
        for (size_t i = 0; i < TEST_SIZE; i += 4) {
            __m256d vec = _mm256_loadu_pd(x_log.data() + i);
            __m256d res = Sleef_logd4_u10avx2(vec); 
            _mm256_storeu_pd(result_sleef_u10.data() + i, res);
        }
        double sleef_u10_time = timer.elapsed();
        std::cout << "  SLEEF u10 time: " << std::fixed << std::setprecision(6) << sleef_u10_time << " ms";
        if (sleef_u10_time > 1e-9) std::cout << " (Speedup vs std: " << std::fixed << std::setprecision(3) << std_time / sleef_u10_time << "x)\n\n";
        else std::cout << " (Speedup vs std: N/A)\n\n";
    }
     // --- Benchmark Error functions (erf / erfc) ---
    {
         std::cout << "--- Error functions (erf / erfc) ---\n";
         std::vector<double> result_std_erf(TEST_SIZE);
         std::vector<double> result_sleef_erf_u10(TEST_SIZE);
         std::vector<double> result_sleef_erfc_u15(TEST_SIZE); 
         std::vector<double> x_erf(TEST_SIZE);
          for (size_t i = 0; i < TEST_SIZE; ++i) {
             double val_exp_erf = 20.0 * (static_cast<double>(i) / TEST_SIZE - 0.5);
             x_erf[i] = val_exp_erf;
         }

         Timer timer;
         for (size_t i = 0; i < TEST_SIZE; ++i) { result_std_erf[i] = std::erf(x_erf[i]); }
         double std_time = timer.elapsed();
         std::cout << "  Standard erf time:  " << std::fixed << std::setprecision(6) << std_time << " ms\n";

         timer.reset();
         for (size_t i = 0; i < TEST_SIZE; i += 4) {
             __m256d vec = _mm256_loadu_pd(x_erf.data() + i);
             __m256d res = Sleef_erfd4_u10avx2(vec); 
             _mm256_storeu_pd(result_sleef_erf_u10.data() + i, res);
         }
         double sleef_u10_time = timer.elapsed();
         std::cout << "  SLEEF erf u10 time: " << std::fixed << std::setprecision(6) << sleef_u10_time << " ms";
         if (sleef_u10_time > 1e-9) std::cout << " (Speedup vs std: " << std::fixed << std::setprecision(3) << std_time / sleef_u10_time << "x)\n";
         else std::cout << " (Speedup vs std: N/A)\n";

         timer.reset();
         for (size_t i = 0; i < TEST_SIZE; i += 4) {
             __m256d vec = _mm256_loadu_pd(x_erf.data() + i);
             __m256d res = Sleef_erfcd4_u15avx2(vec); 
             _mm256_storeu_pd(result_sleef_erfc_u15.data() + i, res);
         }
         double sleef_u15_time = timer.elapsed();
         std::cout << "  SLEEF erfc u15 time: " << std::fixed << std::setprecision(6) << sleef_u15_time << " ms (Note: measures erfc)\n\n";
    }
}

void test_normal_cdf_precision() {
     // (Implementation from previous step - kept for context)
     std::cout << "\n=== Testing Normal CDF Precision ===\n"; //... rest of function ...
    const int TEST_SIZE = 20;
    std::vector<double> x(TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) { x[i] = -5.0 + i * 0.5; }
    std::vector<double> result_std(TEST_SIZE);
    std::vector<double> result_simd(TEST_SIZE);
    for (int i = 0; i < TEST_SIZE; i++) { result_std[i] = 0.5 * (1.0 + std::erf(x[i] / std::sqrt(2.0))); }
    engine::alo::opt::VectorMath::normalCDF(x.data(), result_simd.data(), TEST_SIZE);
    std::cout << "     x      |   std::erf   |   SLEEF   |   Diff   \n";
    std::cout << "------------------------------------------------\n";
    double max_diff = 0.0; double max_rel_diff = 0.0;
     for (int i = 0; i < TEST_SIZE; i++) {
         double diff = std::abs(result_std[i] - result_simd[i]);
         double rel_diff = 0.0; if (std::abs(result_std[i]) > 1e-10) { rel_diff = diff / std::abs(result_std[i]); }
         max_diff = std::max(max_diff, diff); max_rel_diff = std::max(max_rel_diff, rel_diff);
         std::cout << std::fixed << std::setprecision(6);
         std::cout << std::setw(10) << x[i] << " | "  << std::setw(12) << result_std[i] << " | " << std::setw(9) << result_simd[i] << " | " << std::scientific << std::setprecision(3) << std::setw(9) << diff << "\n";
     }
    std::cout << "\nSummary:\n";
    std::cout << "  Max Difference: " << std::scientific << max_diff << "\n";
    std::cout << "  Max Relative Difference: " << std::fixed << std::setprecision(6) << max_rel_diff * 100.0 << "%\n";
}


// --- New Profiling Function ---

/**
 * @brief Profile the stages of Black-Scholes Put calculation
 */
void profile_black_scholes_put() {
    constexpr size_t BS_TEST_SIZE = 100000; // Smaller than function bench, but large enough
    std::cout << "\n=== Black-Scholes Put Calculation Profiling ===\n";
    std::cout << "Testing with " << BS_TEST_SIZE << " options.\n\n";

    // --- Prepare Data ---
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
    std::vector<double> result_vec(BS_TEST_SIZE);

    Timer total_timer;
    Timer stage_timer;
    
    // --- Standard Library Implementation Profiling ---
    std::cout << "--- Standard Library Profiling ---\n";
    double time_std_d1d2 = 0, time_std_cdf = 0, time_std_exp = 0, time_std_combine = 0;
    
    total_timer.reset();
    for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
        // Handle degenerate cases
        if (vol[i] <= 0.0 || T[i] <= 0.0) {
            result_std[i] = std::max(0.0, K[i] - S[i]);
            continue;
        }

        // Stage 1: d1/d2 Calculation
        stage_timer.reset();
        double vsqrtT = vol[i] * std::sqrt(T[i]);
        double d1 = (std::log(S[i] / K[i]) + (r[i] - q[i] + 0.5 * vol[i] * vol[i]) * T[i]) / vsqrtT;
        double d2 = d1 - vsqrtT;
        time_std_d1d2 += stage_timer.elapsed();

        // Stage 2: CDF Calculation N(-d1), N(-d2)
        stage_timer.reset();
        double nd1 = 0.5 * (1.0 + std::erf(-d1 / M_SQRT2)); // N(-d1)
        double nd2 = 0.5 * (1.0 + std::erf(-d2 / M_SQRT2)); // N(-d2)
        time_std_cdf += stage_timer.elapsed();

        // Stage 3: Discount Factor Calculation exp(-rT), exp(-qT)
        stage_timer.reset();
        double dr = std::exp(-r[i] * T[i]);
        double dq = std::exp(-q[i] * T[i]);
        time_std_exp += stage_timer.elapsed();
        
        // Stage 4: Combine Terms
        stage_timer.reset();
        result_std[i] = K[i] * dr * nd2 - S[i] * dq * nd1;
        time_std_combine += stage_timer.elapsed();
    }
    double total_std_time = total_timer.elapsed();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Total Time:      " << total_std_time << " ms\n";
    std::cout << "  Time d1/d2:      " << time_std_d1d2 << " ms (" << (time_std_d1d2 / total_std_time * 100.0) << "%)\n";
    std::cout << "  Time CDF(erf):   " << time_std_cdf << " ms (" << (time_std_cdf / total_std_time * 100.0) << "%)\n";
    std::cout << "  Time Discount(exp):" << time_std_exp << " ms (" << (time_std_exp / total_std_time * 100.0) << "%)\n";
    std::cout << "  Time Combine:    " << time_std_combine << " ms (" << (time_std_combine / total_std_time * 100.0) << "%)\n\n";


    // --- VectorMath (SLEEF/AVX2) Implementation Profiling ---
    std::cout << "--- VectorMath Profiling ---\n";
    double time_vec_d1d2 = 0, time_vec_cdf = 0, time_vec_exp = 0, time_vec_combine = 0;
    
    // Allocate intermediate arrays
    std::vector<double> d1_vec(BS_TEST_SIZE), d2_vec(BS_TEST_SIZE);
    std::vector<double> neg_d1_vec(BS_TEST_SIZE), neg_d2_vec(BS_TEST_SIZE);
    std::vector<double> Nd1_vec(BS_TEST_SIZE), Nd2_vec(BS_TEST_SIZE);
    std::vector<double> dr_vec(BS_TEST_SIZE), dq_vec(BS_TEST_SIZE);
    std::vector<double> term1_vec(BS_TEST_SIZE), term2_vec(BS_TEST_SIZE);
    std::vector<double> neg_rT_vec(BS_TEST_SIZE), neg_qT_vec(BS_TEST_SIZE);

    total_timer.reset();
    
    // Stage 1: d1/d2 Calculation
    stage_timer.reset();
    engine::alo::opt::VectorMath::bsD1D2(S.data(), K.data(), r.data(), q.data(), vol.data(), T.data(), 
                                        d1_vec.data(), d2_vec.data(), BS_TEST_SIZE);
    time_vec_d1d2 = stage_timer.elapsed();

    // Negate d1/d2 (negligible time, included in CDF stage for simplicity)
    for(size_t i=0; i<BS_TEST_SIZE; ++i) {
        neg_d1_vec[i] = -d1_vec[i];
        neg_d2_vec[i] = -d2_vec[i];
    }

    // Stage 2: CDF Calculation N(-d1), N(-d2)
    stage_timer.reset();
    engine::alo::opt::VectorMath::normalCDF(neg_d1_vec.data(), Nd1_vec.data(), BS_TEST_SIZE);
    engine::alo::opt::VectorMath::normalCDF(neg_d2_vec.data(), Nd2_vec.data(), BS_TEST_SIZE);
    time_vec_cdf = stage_timer.elapsed();

    // Stage 3: Discount Factor Calculation exp(-rT), exp(-qT)
    // Prepare inputs for exp
    for(size_t i=0; i<BS_TEST_SIZE; ++i) {
        neg_rT_vec[i] = -r[i] * T[i];
        neg_qT_vec[i] = -q[i] * T[i];
    }
    stage_timer.reset();
    engine::alo::opt::VectorMath::exp(neg_rT_vec.data(), dr_vec.data(), BS_TEST_SIZE);
    engine::alo::opt::VectorMath::exp(neg_qT_vec.data(), dq_vec.data(), BS_TEST_SIZE);
    time_vec_exp = stage_timer.elapsed();

    // Stage 4: Combine Terms K*dr*Nd2 - S*dq*Nd1
    stage_timer.reset();
    engine::alo::opt::VectorMath::multiply(K.data(), dr_vec.data(), term1_vec.data(), BS_TEST_SIZE);          // K*dr
    engine::alo::opt::VectorMath::multiply(term1_vec.data(), Nd2_vec.data(), term1_vec.data(), BS_TEST_SIZE); // (K*dr)*Nd2
    engine::alo::opt::VectorMath::multiply(S.data(), dq_vec.data(), term2_vec.data(), BS_TEST_SIZE);          // S*dq
    engine::alo::opt::VectorMath::multiply(term2_vec.data(), Nd1_vec.data(), term2_vec.data(), BS_TEST_SIZE); // (S*dq)*Nd1
    engine::alo::opt::VectorMath::subtract(term1_vec.data(), term2_vec.data(), result_vec.data(), BS_TEST_SIZE); // term1 - term2
    time_vec_combine = stage_timer.elapsed();

    // Handle degenerate cases after vector computation for simplicity in profiling stages
     for(size_t i=0; i<BS_TEST_SIZE; ++i) {
         if (vol[i] <= 0.0 || T[i] <= 0.0) {
            result_vec[i] = std::max(0.0, K[i] - S[i]);
         }
     }
    double total_vec_time = total_timer.elapsed();

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Total Time:      " << total_vec_time << " ms";
    if (total_std_time > 1e-9) std::cout << " (Speedup vs std: " << std::fixed << std::setprecision(3) << total_std_time / total_vec_time << "x)\n";
    else std::cout << "\n";
    std::cout << "  Time d1/d2:      " << time_vec_d1d2 << " ms (" << (time_vec_d1d2 / total_vec_time * 100.0) << "%)\n";
    std::cout << "  Time CDF:        " << time_vec_cdf << " ms (" << (time_vec_cdf / total_vec_time * 100.0) << "%)\n";
    std::cout << "  Time Discount(exp):" << time_vec_exp << " ms (" << (time_vec_exp / total_vec_time * 100.0) << "%)\n";
    std::cout << "  Time Combine:    " << time_vec_combine << " ms (" << (time_vec_combine / total_vec_time * 100.0) << "%)\n\n";

    // --- Verification ---
    double max_diff = 0.0;
    double sum_std = 0.0;
    double sum_vec = 0.0;
    for (size_t i = 0; i < BS_TEST_SIZE; ++i) {
        max_diff = std::max(max_diff, std::abs(result_std[i] - result_vec[i]));
        sum_std += result_std[i];
        sum_vec += result_vec[i];
    }
    std::cout << "Verification:\n";
    std::cout << "  Max Abs Diff: " << std::scientific << max_diff << "\n";
    std::cout << "  Std Sum:      " << std::fixed << sum_std << "\n";
    std::cout << "  Vec Sum:      " << std::fixed << sum_vec << "\n";

}


/**
 * Main function
 */
int main() {
    // Run the function benchmark (optional, can be commented out)
    benchmark_sleef_functions();
    
    // Run the precision test (optional, can be commented out)
    test_normal_cdf_precision();

    // Run the Black-Scholes profiling test
    profile_black_scholes_put();
        
    return 0;
}