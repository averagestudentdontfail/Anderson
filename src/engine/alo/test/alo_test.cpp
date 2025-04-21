#include "../aloengine.h"
#include "../mod/american.h"
#include "../mod/european.h"
#include "../opt/cache.h"
#include "../opt/simd.h" // Includes SimdOps
#include "../opt/vector.h" // Includes VectorMath
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric> // For std::accumulate

using namespace engine::alo;

/**
 * @brief Simple timer class for performance measurement
 */
class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = now - start_;
        return elapsed.count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};
 
/**
 * @brief Run a simple pricing test (Unchanged)
 */
void runSimplePricingTest() {
    std::cout << "\n=== Simple Pricing Test ===\n";
    
    // Create ALO engine
    ALOEngine engine(ACCURATE);
    
    // Option parameters
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.02;
    double vol = 0.2;
    double T = 1.0;
    
    // Price American put
    double americanPut = engine.calculateOption(S, K, r, q, vol, T, PUT);
    
    // Price American call
    double americanCall = engine.calculateOption(S, K, r, q, vol, T, CALL);
    
    // Price European put
    double europeanPut = ALOEngine::blackScholesPut(S, K, r, q, vol, T);
    
    // Price European call
    double europeanCall = ALOEngine::blackScholesCall(S, K, r, q, vol, T);
    
    // Calculate early exercise premiums
    double putPremium = americanPut - europeanPut;
    double callPremium = americanCall - europeanCall;
    
    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Option Parameters:\n";
    std::cout << "  Spot Price:         " << S << "\n";
    std::cout << "  Strike Price:       " << K << "\n";
    std::cout << "  Risk-Free Rate:     " << r << "\n";
    std::cout << "  Dividend Yield:     " << q << "\n";
    std::cout << "  Volatility:         " << vol << "\n";
    std::cout << "  Time to Maturity:   " << T << " year\n\n";
    
    std::cout << "Pricing Results:\n";
    std::cout << "  European Put:       " << europeanPut << "\n";
    std::cout << "  American Put:       " << americanPut << "\n";
    std::cout << "  Put Premium:        " << putPremium << "\n\n";
    
    std::cout << "  European Call:      " << europeanCall << "\n";
    std::cout << "  American Call:      " << americanCall << "\n";
    std::cout << "  Call Premium:       " << callPremium << "\n";
}
 
/**
 * @brief Run a batch pricing test (Unchanged)
 */
void runBatchPricingTest() {
    std::cout << "\n=== Batch Pricing Test ===\n";
    
    // Create ALO engine
    ALOEngine engine(ACCURATE);
    
    // Option parameters
    double S = 100.0;
    double r = 0.05;
    double q = 0.02;
    double vol = 0.2;
    double T = 1.0;
    
    // Create a range of strikes
    std::vector<double> strikes;
    for (double K = 80.0; K <= 120.0; K += 5.0) {
        strikes.push_back(K);
    }
    
    // Batch price American puts
    Timer timer;
    auto putPrices = engine.batchCalculatePut(S, strikes, r, q, vol, T);
    double batchTime = timer.elapsed();
    
    // Individual pricing for comparison
    timer.reset();
    std::vector<double> individualPrices;
    for (double K : strikes) {
        individualPrices.push_back(engine.calculateOption(S, K, r, q, vol, T, PUT));
    }
    double individualTimeTotal = timer.elapsed(); // Total time for all individual calls
    
    // Compute European prices for comparison
    std::vector<double> europeanPrices;
    for (double K : strikes) {
        europeanPrices.push_back(ALOEngine::blackScholesPut(S, K, r, q, vol, T));
    }
    
    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Common Parameters:\n";
    std::cout << "  Spot Price:         " << S << "\n";
    std::cout << "  Risk-Free Rate:     " << r << "\n";
    std::cout << "  Dividend Yield:     " << q << "\n";
    std::cout << "  Volatility:         " << vol << "\n";
    std::cout << "  Time to Maturity:   " << T << " year\n\n";
    
    std::cout << "Pricing Results:\n";
    std::cout << "Strike   European Put   American Put   Premium\n";
    std::cout << "------   ------------   ------------   -------\n";
    
    for (size_t i = 0; i < strikes.size(); ++i) {
        double premium = putPrices[i] - europeanPrices[i];
        std::cout << std::setw(6) << strikes[i] << "   "
                  << std::setw(12) << europeanPrices[i] << "   "
                  << std::setw(12) << putPrices[i] << "   "
                  << std::setw(7) << premium << "\n";
    }
    
    std::cout << "\nPerformance:\n";
    std::cout << "  Batch Pricing Time:      " << batchTime << " ms\n";
    // Calculate time per option for individual calls
    if (!strikes.empty()) {
        std::cout << "  Individual Pricing Time: " << individualTimeTotal / strikes.size() << " ms per option\n";
        // Calculate speedup based on time per option
        double batchTimePerOption = batchTime / strikes.size();
        double individualTimePerOption = individualTimeTotal / strikes.size();
        if (batchTimePerOption > 1e-9) { // Avoid division by zero
             std::cout << "  Speedup (Individual vs Batch): " << individualTimePerOption / batchTimePerOption << "x\n";
        } else {
             std::cout << "  Speedup (Individual vs Batch): N/A (Batch time too small)\n";
        }
    } else {
         std::cout << "  Individual Pricing Time: N/A (no strikes)\n";
         std::cout << "  Speedup (Individual vs Batch): N/A\n";
    }
}
 
/**
 * @brief Run a parallel pricing test (Unchanged)
 */
void runParallelPricingTest() {
    std::cout << "\n=== Parallel Pricing Test ===\n";
    
    // Create ALO engine
    ALOEngine engine(ACCURATE);
    
    // Option parameters
    double S = 100.0;
    double r = 0.05;
    double q = 0.02;
    double vol = 0.2;
    double T = 1.0;
    
    // Create a large range of strikes
    std::vector<double> strikes;
    for (double K = 50.0; K <= 150.0; K += 0.5) {
        strikes.push_back(K);
    }
    
    // Sequential pricing
    Timer timer;
    auto seqPrices = engine.batchCalculatePut(S, strikes, r, q, vol, T);
    double seqTime = timer.elapsed();
    
    // Parallel pricing
    timer.reset();
    auto parPrices = engine.parallelBatchCalculatePut(S, strikes, r, q, vol, T);
    double parTime = timer.elapsed();
    
    // Verify results
    bool resultsMatch = true;
    double maxDiff = 0.0;
    if (seqPrices.size() == parPrices.size()){
        for (size_t i = 0; i < strikes.size(); ++i) {
            double diff = std::abs(seqPrices[i] - parPrices[i]);
            maxDiff = std::max(maxDiff, diff);
            if (diff > 1e-9) { // Use a slightly larger tolerance for parallel FP differences
                resultsMatch = false;
            }
        }
    } else {
        resultsMatch = false; // Sizes don't match
    }

    
    // Print performance results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Performance for " << strikes.size() << " options:\n";
    std::cout << "  Sequential Time:  " << seqTime << " ms\n";
    std::cout << "  Parallel Time:    " << parTime << " ms\n";
    if (parTime > 1e-9) { // Avoid division by zero
         std::cout << "  Speedup:          " << seqTime / parTime << "x\n";
    } else {
         std::cout << "  Speedup:          N/A (Parallel time too small)\n";
    }
    std::cout << "  Results Match:    " << (resultsMatch ? "Yes" : "No") << "\n";
    std::cout << "  Maximum Difference: " << maxDiff << "\n";
}
 
/**
 * @brief Run a SIMD optimization test comparing Standard, Fused SIMD, and Separate SIMD
 */
void runSimdOptimizationTest() {
    std::cout << "\n=== SIMD Optimization Test (exp(x) * sqrt(y)) ===\n";
    
    // Create test data
    constexpr size_t dataSize = 1000000; // Ensure this is large enough for meaningful timing
    // Ensure dataSize is a multiple of 4 for easier SIMD handling without remainder loops
    static_assert(dataSize % 4 == 0, "dataSize must be a multiple of 4 for SIMD tests");

    std::vector<double> x(dataSize);
    std::vector<double> y(dataSize);
    std::vector<double> z_std(dataSize);      // Result from standard loop
    std::vector<double> z_fused(dataSize);    // Result from VectorMath::expMultSqrt
    std::vector<double> z_separate(dataSize); // Result from separate VectorMath calls
    
    // Allocate temporary arrays for the separate SIMD approach
    std::vector<double> temp_exp_x(dataSize);
    std::vector<double> temp_sqrt_y(dataSize);

    // Initialize with some values
    for (size_t i = 0; i < dataSize; ++i) {
        // Use more reasonable values that avoid numerical overflow/underflow issues
        x[i] = -5.0 + 10.0 * static_cast<double>(i) / dataSize;  // Range from -5 to 5
        y[i] = 0.1 + 10.0 * static_cast<double>(i) / dataSize;   // Range from 0.1 to 10.1 (ensure y >= 0 for sqrt)
    }
    
    Timer timer;
    
    // --- 1. Standard Calculation ---
    timer.reset();
    for (size_t i = 0; i < dataSize; ++i) {
        z_std[i] = std::exp(x[i]) * std::sqrt(y[i]);
    }
    double standardTime = timer.elapsed();
    double sumStd = std::accumulate(z_std.begin(), z_std.end(), 0.0);

    // --- 2. Fused SIMD Calculation ---
    timer.reset();
    opt::VectorMath::expMultSqrt(x.data(), y.data(), z_fused.data(), dataSize);
    double fusedSimdTime = timer.elapsed();
    double sumFused = std::accumulate(z_fused.begin(), z_fused.end(), 0.0);

    // --- 3. Separate SIMD Calculation ---
    timer.reset();
    opt::VectorMath::exp(x.data(), temp_exp_x.data(), dataSize);
    opt::VectorMath::sqrt(y.data(), temp_sqrt_y.data(), dataSize);
    opt::VectorMath::multiply(temp_exp_x.data(), temp_sqrt_y.data(), z_separate.data(), dataSize);
    double separateSimdTime = timer.elapsed();
    double sumSeparate = std::accumulate(z_separate.begin(), z_separate.end(), 0.0);

    // --- Verification ---
    double maxAbsDiffFused = 0.0;
    double maxRelDiffFused = 0.0;
    double maxAbsDiffSeparate = 0.0;
    double maxRelDiffSeparate = 0.0;

    for (size_t i = 0; i < dataSize; ++i) {
        // Compare Fused SIMD vs Standard
        double absDiffF = std::abs(z_std[i] - z_fused[i]);
        maxAbsDiffFused = std::max(maxAbsDiffFused, absDiffF);
        if (std::abs(z_std[i]) > 1e-10) {
            maxRelDiffFused = std::max(maxRelDiffFused, absDiffF / std::abs(z_std[i]));
        }

        // Compare Separate SIMD vs Standard
         double absDiffS = std::abs(z_std[i] - z_separate[i]);
        maxAbsDiffSeparate = std::max(maxAbsDiffSeparate, absDiffS);
        if (std::abs(z_std[i]) > 1e-10) {
            maxRelDiffSeparate = std::max(maxRelDiffSeparate, absDiffS / std::abs(z_std[i]));
        }
    }
    
    // --- Print performance results ---
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Performance for " << dataSize << " operations (exp(x) * sqrt(y)):\n";
    std::cout << "  Standard Time:       " << standardTime << " ms\n";
    
    std::cout << "  Fused SIMD Time:     " << fusedSimdTime << " ms ";
    if (standardTime > 1e-9) 
        std::cout << "(Speedup vs Std: " << std::setprecision(3) << standardTime / fusedSimdTime << "x)\n";
    else std::cout << "(Speedup vs Std: N/A)\n";
    
    std::cout << "  Separate SIMD Time:  " << separateSimdTime << " ms ";
     if (standardTime > 1e-9) 
        std::cout << "(Speedup vs Std: " << std::setprecision(3) << standardTime / separateSimdTime << "x)\n";
    else std::cout << "(Speedup vs Std: N/A)\n";

    std::cout << "\nVerification vs Standard:\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Fused SIMD Max Abs Diff:   " << maxAbsDiffFused << "\n";
    std::cout << "  Fused SIMD Max Rel Diff:   " << maxRelDiffFused << "\n";
    std::cout << "  Separate SIMD Max Abs Diff:" << maxAbsDiffSeparate << "\n";
    std::cout << "  Separate SIMD Max Rel Diff:" << maxRelDiffSeparate << "\n";
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Standard Sum:        " << sumStd << "\n";
    std::cout << "  Fused SIMD Sum:      " << sumFused << "\n";
    std::cout << "  Separate SIMD Sum:   " << sumSeparate << "\n";
    
    // Print some sample values for manual verification
    std::cout << "\nSample values (first 5 elements):\n";
    std::cout << "   x       y      Standard      Fused SIMD    Separate SIMD\n";
    std::cout << "------  ------  ------------  ------------  -------------\n";
    
    for (size_t i = 0; i < 5; ++i) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << std::setw(6) << x[i] << "  " 
                  << std::setw(6) << y[i] << "  "
                  << std::setw(12) << z_std[i] << "  "
                  << std::setw(12) << z_fused[i] << "  "
                  << std::setw(13) << z_separate[i] << "\n";
    }
}
 
/**
 * @brief Run a cache optimization test (Unchanged)
 */
void runCacheOptimizationTest() {
    std::cout << "\n=== Cache Optimization Test ===\n";
    
    // Create ALO engines
    ALOEngine engineNoCache(ACCURATE);
    ALOEngine engineWithCache(ACCURATE);
    
    // Clear cache in the second engine
    engineWithCache.clearCache();
    
    // Option parameters
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.02;
    double vol = 0.2;
    double T = 1.0;
    
    // First pricing to warm up both engines
    engineNoCache.calculateOption(S, K, r, q, vol, T, PUT);
    engineWithCache.calculateOption(S, K, r, q, vol, T, PUT);
    
    // Test without cache (repeat pricing 1000 times)
    Timer timer;
    double resultNoCache = 0.0;
    for (int i = 0; i < 1000; ++i) {
        resultNoCache = engineNoCache.calculateOption(S, K, r, q, vol, T, PUT);
    }
    double timeNoCache = timer.elapsed();
    
    // Test with cache (repeat pricing 1000 times)
    timer.reset();
    double resultWithCache = 0.0;
    for (int i = 0; i < 1000; ++i) {
        resultWithCache = engineWithCache.calculateOption(S, K, r, q, vol, T, PUT);
    }
    double timeWithCache = timer.elapsed();
    
    // Print performance results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Performance for 1000 repeated pricings:\n";
    std::cout << "  Time without Cache: " << timeNoCache << " ms\n";
    std::cout << "  Time with Cache:    " << timeWithCache << " ms\n";
    if (timeWithCache > 1e-9) { // Avoid division by zero
        std::cout << "  Speedup:            " << timeNoCache / timeWithCache << "x\n";
    } else {
         std::cout << "  Speedup:            N/A (Cache time too small)\n";
    }
    std::cout << "  Results Match:      " << (std::abs(resultNoCache - resultWithCache) < 1e-10 ? "Yes" : "No") << "\n";
    std::cout << "  Cache Size:         " << engineWithCache.getCacheSize() << " entries\n";
}
 
/**
 * @brief Main function
 */
int main() {
    std::cout << "=== ALO Engine Test Program ===\n";
    
    try {
        // Run various tests
        runSimplePricingTest();
        runBatchPricingTest();
        runParallelPricingTest();
        runSimdOptimizationTest(); // Run the updated SIMD test
        runCacheOptimizationTest();
        
        std::cout << "\nAll tests completed successfully.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}