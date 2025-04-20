 #include "cache.h"
 #include <chrono>
 #include <algorithm>
 #include <unordered_map>
 #include <mutex>
 #include <shared_mutex>
 #include <thread>
 #include <random>
 
 namespace engine {
 namespace alo {
 namespace opt {
 
 /**
  * @brief Initialize the thread-local cache with optimal settings
  */
 void initializeThreadLocalCache() {
     // Set thread-local cache parameters based on thread ID
     thread_local bool initialized = false;
     
     if (!initialized) {
         auto& cache = getThreadLocalCache();
         
         // Get a pseudo-unique thread ID
         auto threadId = std::hash<std::thread::id>{}(std::this_thread::get_id());
         
         // Use a different max size for each thread to prevent synchronized eviction
         size_t baseSize = 50000;
         size_t variance = threadId % 10000;
         cache.setMaxSize(baseSize + variance);
         
         // Set TTL to 30 seconds 
         cache.setTTL(30000);
         
         initialized = true;
     }
 }
 
 /**
  * @brief Initialize the global tiered cache with optimal settings
  */
 void initializeTieredCache() {
     // This is a no-op since the tiered cache singleton 
     // is already initialized with good defaults
 }
 
 /**
  * @brief Get a batch of pricing results with caching
  * 
  * @param S Spot price
  * @param strikes Vector of strike prices
  * @param r Risk-free rate
  * @param q Dividend yield
  * @param vol Volatility
  * @param T Time to maturity
  * @param computeFunc Function to compute missing prices
  * @return Vector of prices
  */
 std::vector<double> getBatchCachedPrices(
     double S, 
     const std::vector<double>& strikes,
     double r, double q, double vol, double T,
     int optionType,
     const std::function<std::vector<double>(const std::vector<size_t>&, const std::vector<double>&)>& computeFunc) {
     
     // Ensure thread-local cache is initialized
     initializeThreadLocalCache();
     
     // Create option parameter keys for all strikes
     std::vector<OptionParams> keys;
     keys.reserve(strikes.size());
     
     for (const auto& K : strikes) {
         keys.push_back({S, K, r, q, vol, T, optionType});
     }
     
     // Check cache for existing results
     auto& tieredCache = getTieredPricingCache();
     auto cacheHits = tieredCache.batchLookup(keys);
     
     // If all results were in cache, return them
     if (cacheHits.size() == strikes.size()) {
         std::vector<double> results(strikes.size());
         for (const auto& hit : cacheHits) {
             results[hit.index] = hit.value;
         }
         return results;
     }
     
     // Collect indices and strikes for cache misses
     std::vector<size_t> missIndices;
     std::vector<double> missStrikes;
     
     // Create lookup table for cache hits
     std::vector<bool> found(strikes.size(), false);
     for (const auto& hit : cacheHits) {
         found[hit.index] = true;
     }
     
     // Collect indices and strikes for cache misses
     for (size_t i = 0; i < strikes.size(); ++i) {
         if (!found[i]) {
             missIndices.push_back(i);
             missStrikes.push_back(strikes[i]);
         }
     }
     
     // Compute missing prices
     std::vector<double> missPrices = computeFunc(missIndices, missStrikes);
     
     // Combine cache hits and computed prices
     std::vector<double> results(strikes.size());
     
     // First, fill in cache hits
     for (const auto& hit : cacheHits) {
         results[hit.index] = hit.value;
     }
     
     // Then, fill in computed prices and update cache
     for (size_t i = 0; i < missIndices.size(); ++i) {
         size_t originalIndex = missIndices[i];
         double price = missPrices[i];
         
         results[originalIndex] = price;
         
         // Update cache
         tieredCache.put(keys[originalIndex], price);
     }
     
     return results;
 }
 
 /**
  * @brief Helper function to warm up caches with common parameter sets
  * 
  * @param engine ALO engine to use for pricing
  */
 void warmupCache(const ALOEngine& engine) {
     // Common parameter sets for interest rates, volatilities, etc.
     std::vector<double> rates = {0.01, 0.02, 0.03, 0.04, 0.05};
     std::vector<double> vols = {0.1, 0.15, 0.2, 0.25, 0.3};
     std::vector<double> times = {0.25, 0.5, 1.0, 2.0};
     
     // Compute and cache a limited set of options with common parameters
     double S = 100.0;
     std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
     double q = 0.01;
     
     // Cache the most common parameter sets
     for (double r : rates) {
         for (double vol : vols) {
             for (double T : times) {
                 engine.batchCalculatePut(S, strikes, r, q, vol, T);
             }
         }
     }
 }
 
 } // namespace opt
 } // namespace alo
 } // namespace engine