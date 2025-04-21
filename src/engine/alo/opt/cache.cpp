#include "cache.h"
#include "../aloengine.h" // Include necessary engine header
#include <chrono>
#include <algorithm>
#include <vector>
#include <thread>       // For std::this_thread
#include <random>       // For potential future randomized eviction/init

namespace engine {
namespace alo {
namespace opt {

/**
 * @brief Initialize the thread-local cache with optimal settings
 * * This function ensures that each thread calling it gets a properly
 * initialized thread-local cache instance. Can be expanded later
 * to set different sizes/TTLs based on thread roles if needed.
 */
 void initializeThreadLocalCache() {
     // Access the thread-local cache to ensure it's constructed.
     // The default constructor values (50k size, 30s TTL) are used unless
     // specific per-thread settings are required later.
     auto& cache = getThreadLocalCache(); 
     
     // Example: Set a slightly different size based on thread ID hash
     // This might help decorrelate eviction patterns across threads.
     // std::hash<std::thread::id> hasher;
     // size_t thread_hash = hasher(std::this_thread::get_id());
     // cache.setMaxSize(50000 + (thread_hash % 10000)); 
     // cache.setTTL(30000); // 30 seconds TTL

     // Currently, just ensuring it's accessed is enough if defaults are okay.
     (void)cache; // Suppress unused variable warning if not modifying defaults
 }
 

/**
 * @brief Helper function to warm up caches with common parameter sets
 * * Populates the calling thread's local cache with results for frequently
 * expected parameter combinations.
 * * @param engine ALO engine instance (const reference) to use for pricing
 */
 void warmupCache(const ALOEngine& engine) {
     // Ensure the cache for this thread is initialized
     initializeThreadLocalCache(); 

     // Define common parameter ranges
     std::vector<double> rates = {0.01, 0.02, 0.03, 0.04, 0.05};
     std::vector<double> vols = {0.10, 0.15, 0.20, 0.25, 0.30, 0.35};
     std::vector<double> times = {0.25, 0.5, 1.0, 1.5, 2.0};
     std::vector<double> strikes_rel = {-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0}; // Relative to spot

     double S = 100.0;
     double q = 0.02; // Example fixed dividend yield

     std::cout << "Warming up cache for thread " << std::this_thread::get_id() << "..." << std::endl;
     int count = 0;
     
     // Calculate and cache results for common scenarios
     // Focus on puts as they are more computationally intensive for ALO
     for (double r : rates) {
         for (double vol : vols) {
             for (double T : times) {
                 for (double k_rel : strikes_rel) {
                     double K = S + k_rel;
                     if (K > 0) { // Ensure strike is positive
                        // Calculate the price (this will use getCachedPrice, populating the cache)
                        engine.calculateOption(S, K, r, q, vol, T, PUT); 
                        count++;
                     }
                 }
             }
         }
     }
     std::cout << "Cache warmup complete for thread " << std::this_thread::get_id() << ". Added approx " << count << " entries." << std::endl;

     // Optionally, add calls for common CALL options if needed
 }

} // namespace opt
} // namespace alo
} // namespace engine