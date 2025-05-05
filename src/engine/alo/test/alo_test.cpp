#include "../aloengine.h"
#include "../mod/american.h"
#include "../mod/european.h"
#include "../opt/cache.h"
#include "../opt/simd.h"
#include "../opt/vector.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <string>
#include <memory>
#include <algorithm>

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
 * @brief Run a simple pricing test
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
 * @brief Run a single-precision pricing test
 */
void runSinglePrecisionTest() {
    std::cout << "\n=== Single-Precision Pricing Test ===\n";
    
    // Create ALO engine
    ALOEngine engine(ACCURATE);
    
    // Option parameters
    double S = 100.0;
    double K = 100.0;
    double r = 0.05;
    double q = 0.02;
    double vol = 0.2;
    double T = 1.0;
    
    // Price options with double precision
    double europeanPutDouble = ALOEngine::blackScholesPut(S, K, r, q, vol, T);
    double europeanCallDouble = ALOEngine::blackScholesCall(S, K, r, q, vol, T);
    double americanPutDouble = engine.calculateOption(S, K, r, q, vol, T, PUT);
    double americanCallDouble = engine.calculateOption(S, K, r, q, vol, T, CALL);
    
    // Price options with single precision
    float europeanPutSingle = engine.calculateEuropeanSingle(S, K, r, q, vol, T, 0);
    float europeanCallSingle = engine.calculateEuropeanSingle(S, K, r, q, vol, T, 1);
    float americanPutSingle = engine.calculateAmericanSingle(S, K, r, q, vol, T, 0);
    float americanCallSingle = engine.calculateAmericanSingle(S, K, r, q, vol, T, 1);
    
    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Option Parameters:\n";
    std::cout << "  Spot Price:         " << S << "\n";
    std::cout << "  Strike Price:       " << K << "\n";
    std::cout << "  Risk-Free Rate:     " << r << "\n";
    std::cout << "  Dividend Yield:     " << q << "\n";
    std::cout << "  Volatility:         " << vol << "\n";
    std::cout << "  Time to Maturity:   " << T << " year\n\n";
    
    std::cout << "Double vs Single Precision Comparison:\n";
    std::cout << "                    Double       Single      Diff        Rel Diff\n";
    std::cout << "  European Put:   " << std::setw(10) << europeanPutDouble 
              << "   " << std::setw(10) << europeanPutSingle 
              << "   " << std::setw(10) << std::abs(europeanPutDouble - europeanPutSingle) 
              << "   " << std::setw(10) << std::abs(europeanPutDouble - europeanPutSingle)/europeanPutDouble << "\n";
    
    std::cout << "  European Call:  " << std::setw(10) << europeanCallDouble 
              << "   " << std::setw(10) << europeanCallSingle 
              << "   " << std::setw(10) << std::abs(europeanCallDouble - europeanCallSingle) 
              << "   " << std::setw(10) << std::abs(europeanCallDouble - europeanCallSingle)/europeanCallDouble << "\n";
    
    std::cout << "  American Put:   " << std::setw(10) << americanPutDouble 
              << "   " << std::setw(10) << americanPutSingle 
              << "   " << std::setw(10) << std::abs(americanPutDouble - americanPutSingle) 
              << "   " << std::setw(10) << std::abs(americanPutDouble - americanPutSingle)/americanPutDouble << "\n";
    
    std::cout << "  American Call:  " << std::setw(10) << americanCallDouble 
              << "   " << std::setw(10) << americanCallSingle 
              << "   " << std::setw(10) << std::abs(americanCallDouble - americanCallSingle) 
              << "   " << std::setw(10) << std::abs(americanCallDouble - americanCallSingle)/americanCallDouble << "\n";
}

/**
 * @brief Run a batch pricing test for single precision
 */
void runBatchSinglePrecisionTest() {
    std::cout << "\n=== Batch Single-Precision Test ===\n";
    
    // Create ALO engine
    ALOEngine engine(ACCURATE);
    
    // Option parameters
    float S = 100.0f;
    float r = 0.05f;
    float q = 0.02f;
    float vol = 0.2f;
    float T = 1.0f;
    
    // Create a range of strikes
    std::vector<float> strikes;
    for (float K = 80.0f; K <= 120.0f; K += 5.0f) {
        strikes.push_back(K);
    }
    
    // Batch price puts with single precision
    Timer timer;
    auto putPricesSingle = engine.batchCalculatePutSingle(S, strikes, r, q, vol, T);
    double singleTime = timer.elapsed();
    
    // Batch price puts with double precision
    timer.reset();
    std::vector<double> strikesDouble(strikes.begin(), strikes.end());
    auto putPricesDouble = engine.batchCalculatePut(S, strikesDouble, r, q, vol, T);
    double doubleTime = timer.elapsed();
    
    // Print results
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Common Parameters:\n";
    std::cout << "  Spot Price:         " << S << "\n";
    std::cout << "  Risk-Free Rate:     " << r << "\n";
    std::cout << "  Dividend Yield:     " << q << "\n";
    std::cout << "  Volatility:         " << vol << "\n";
    std::cout << "  Time to Maturity:   " << T << " year\n\n";
    
    std::cout << "Double vs Single Precision Results:\n";
    std::cout << "Strike   Double Prec.   Single Prec.   Diff       Rel Diff\n";
    std::cout << "------   ------------   ------------   --------   --------\n";
    
    double maxDiff = 0.0;
    double maxRelDiff = 0.0;
    
    for (size_t i = 0; i < strikes.size(); ++i) {
        double diff = std::abs(putPricesDouble[i] - putPricesSingle[i]);
        double relDiff = diff / putPricesDouble[i];
        
        maxDiff = std::max(maxDiff, diff);
        maxRelDiff = std::max(maxRelDiff, relDiff);
        
        std::cout << std::setw(6) << strikes[i] << "   "
                  << std::setw(12) << putPricesDouble[i] << "   "
                  << std::setw(12) << putPricesSingle[i] << "   "
                  << std::setw(8) << diff << "   "
                  << std::setw(8) << relDiff << "\n";
    }
    
    std::cout << "\nSummary:\n";
    std::cout << "  Maximum Absolute Difference: " << maxDiff << "\n";
    std::cout << "  Maximum Relative Difference: " << maxRelDiff << "\n";
    
    std::cout << "\nPerformance:\n";
    std::cout << "  Double-Precision Batch Time: " << doubleTime << " ms\n";
    std::cout << "  Single-Precision Batch Time: " << singleTime << " ms\n";
    std::cout << "  Speedup:                     " << doubleTime / singleTime << "x\n";
}

/**
 * @brief Run a performance benchmark comparing double and single precision
 */
void runPerformanceBenchmark() {
    std::cout << "\n=== Performance Benchmark ===\n";
    
    // Create engine
    ALOEngine engine(ACCURATE);
    
    // Number of options to price
    int numOptions = 1000000;
    
    std::cout << "Running performance benchmark with " << numOptions << " options...\n";
    
    // Run the built-in benchmark
    engine.runBenchmark(numOptions);
}

/**
 * @brief Main function
 */
int main() {
    std::cout << "=== ALO Engine Test Program ===\n";
    
    try {
        // Run various tests
        runSimplePricingTest();
        runSinglePrecisionTest();
        runBatchSinglePrecisionTest();
        runPerformanceBenchmark();
        
        std::cout << "\nAll tests completed successfully.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}