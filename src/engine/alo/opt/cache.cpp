#include "cache.h"
#include <chrono>
#include <algorithm>
#include <thread>

namespace engine {
namespace alo {
namespace opt {

// Get a cached option price using the provided key
double getCachedPrice(const OptionKey& key, const std::function<double()>& computeFunc) {
    // Get thread-local cache
    auto& cache = getThreadLocalCache();
    
    // Check if price is in cache
    auto it = cache.find(key);
    if (it != cache.end()) {
        // Cache hit
        return it->second;
    }
    
    // Cache miss - compute the price
    double price = computeFunc();
    
    // Store in cache
    cache[key] = price;
    
    return price;
}

// Clear the thread-local cache
void clearThreadLocalCache() {
    auto& cache = getThreadLocalCache();
    cache.clear();
}

// Get the size of the thread-local cache
size_t getThreadLocalCacheSize() {
    return getThreadLocalCache().size();
}

} // namespace opt
} // namespace alo
} // namespace engine