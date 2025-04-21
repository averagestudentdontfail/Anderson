#include "simd.h"
#include "../aloengine.h"
#include "../mod/american.h"
#include "../mod/european.h"
#include "cache.h"
#include "vector.h"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

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

  void reset() { start_ = std::chrono::high_resolution_clock::now(); }

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
 * @brief Run a batch pricing test
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
  double individualTime = timer.elapsed();

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
    std::cout << std::setw(6) << strikes[i] << "   " << std::setw(12)
              << europeanPrices[i] << "   " << std::setw(12) << putPrices[i]
              << "   " << std::setw(7) << premium << "\n";
  }

  std::cout << "\nPerformance:\n";
  std::cout << "  Batch Pricing Time:      " << batchTime << " ms\n";
  std::cout << "  Individual Pricing Time: " << individualTime << " ms\n";
  std::cout << "  Speedup:                 " << individualTime / batchTime
            << "x\n";
}

/**
 * @brief Run a parallel pricing test
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
  for (size_t i = 0; i < strikes.size(); ++i) {
    double diff = std::abs(seqPrices[i] - parPrices[i]);
    maxDiff = std::max(maxDiff, diff);
    if (diff > 1e-10) {
      resultsMatch = false;
    }
  }

  // Print performance results
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Performance for " << strikes.size() << " options:\n";
  std::cout << "  Sequential Time:  " << seqTime << " ms\n";
  std::cout << "  Parallel Time:    " << parTime << " ms\n";
  std::cout << "  Speedup:          " << seqTime / parTime << "x\n";
  std::cout << "  Results Match:    " << (resultsMatch ? "Yes" : "No") << "\n";
  std::cout << "  Maximum Difference: " << maxDiff << "\n";
}

/**
 * @brief Run a SIMD optimization test
 */
void runSimdOptimizationTest() {
  std::cout << "\n=== SIMD Optimization Test ===\n";

  // Create test data
  constexpr size_t dataSize = 1000000;
  std::vector<double> x(dataSize);
  std::vector<double> y(dataSize);
  std::vector<double> z1(dataSize);
  std::vector<double> z2(dataSize);

  // Initialize with some values
  for (size_t i = 0; i < dataSize; ++i) {
    x[i] = 0.1 * i;
    y[i] = 0.2 * i;
  }

  // Test vector operations (standard)
  Timer timer;
  for (size_t i = 0; i < dataSize; ++i) {
    z1[i] = std::exp(x[i]) * std::sqrt(y[i]);
  }
  double standardTime = timer.elapsed();

  // Test vector operations (SIMD)
  timer.reset();
  std::vector<double> exp_x(dataSize);
  std::vector<double> sqrt_y(dataSize);

  opt::VectorMath::exp(x.data(), exp_x.data(), dataSize);
  opt::VectorMath::sqrt(y.data(), sqrt_y.data(), dataSize);
  opt::VectorMath::multiply(exp_x.data(), sqrt_y.data(), z2.data(), dataSize);

  double simdTime = timer.elapsed();

  // Verify results
  double maxDiff = 0.0;
  for (size_t i = 0; i < dataSize; ++i) {
    double diff = std::abs(z1[i] - z2[i]);
    maxDiff = std::max(maxDiff, diff);
  }

  // Print performance results
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Performance for " << dataSize << " operations:\n";
  std::cout << "  Standard Time:    " << standardTime << " ms\n";
  std::cout << "  SIMD Time:        " << simdTime << " ms\n";
  std::cout << "  Speedup:          " << standardTime / simdTime << "x\n";
  std::cout << "  Maximum Difference: " << maxDiff << "\n";
}

/**
 * @brief Run a cache optimization test
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
  std::cout << "  Speedup:            " << timeNoCache / timeWithCache << "x\n";
  std::cout << "  Results Match:      "
            << (std::abs(resultNoCache - resultWithCache) < 1e-10 ? "Yes"
                                                                  : "No")
            << "\n";
  std::cout << "  Cache Size:         " << engineWithCache.getCacheSize()
            << " entries\n";
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
    runSimdOptimizationTest();
    runCacheOptimizationTest();

    std::cout << "\nAll tests completed successfully.\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
}