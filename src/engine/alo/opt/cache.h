#ifndef ENGINE_ALO_OPT_CACHE_H
#define ENGINE_ALO_OPT_CACHE_H

#include <string>
#include <unordered_map>
#include <mutex>
#include <memory>
#include <chrono>
#include <functional>
#include <iomanip>
#include <sstream>
#include <array>
#include <list>
#include <shared_mutex>
#include <vector>
#include <atomic> 
#include <algorithm> 
#include <limits> 
#include <cstring> // <<< Added this include for std::memcpy

namespace engine {
namespace alo {

// Forward declare ALOEngine to avoid circular dependency if needed elsewhere
class ALOEngine; 

namespace opt {

/**
 * @brief Option parameters structure for efficient caching
 */
struct OptionParams {
    double S, K, r, q, vol, T;
    int type; // PUT=0, CALL=1, etc.
    
    // Equality operator for hash map lookups
    bool operator==(const OptionParams& other) const {
        // Use memcmp for potentially faster comparison if padding isn't an issue
        // static_assert(sizeof(OptionParams) == (6 * sizeof(double) + sizeof(int)), "Padding detected in OptionParams, memcmp might be unsafe");
        // return std::memcmp(this, &other, sizeof(OptionParams)) == 0;
        
        // Fallback to member-wise comparison for safety
         return S == other.S && K == other.K && r == other.r && 
                q == other.q && vol == other.vol && T == other.T && 
                type == other.type;
    }
};

/**
 * @brief Result of a cache lookup with status
 */
struct CacheResult {
    double value;
    bool found;
};

/**
 * @brief Batch cache hit result
 */
struct CacheHit {
    size_t index; // Original index from the input batch
    double value;
    bool found;   // Should always be true if included in results
};

/**
 * @brief Hash function for OptionParams
 */
struct OptionParamsHash {
    size_t operator()(const OptionParams& params) const {
        // FNV-1a hash algorithm - generally fast and good distribution
        size_t hash = 14695981039346656037ULL; // 64-bit offset basis
        const uint64_t prime = 1099511627776ULL; // 64-bit prime

        auto hash_double = [&hash, prime](double val) {
            // Treat double as bytes for hashing
            uint64_t bits;
            // Use std::memcpy safely now that <cstring> is included
            std::memcpy(&bits, &val, sizeof(double)); 
            // Avoid hashing negative zero differently from positive zero if necessary
            if (bits == 0x8000000000000000ULL) bits = 0; 
            
            const uint8_t* data = reinterpret_cast<const uint8_t*>(&bits);
            for (size_t i = 0; i < sizeof(uint64_t); ++i) {
                hash ^= static_cast<size_t>(data[i]);
                hash *= prime;
            }
        };
        
        hash_double(params.S);
        hash_double(params.K);
        hash_double(params.r);
        hash_double(params.q);
        hash_double(params.vol);
        hash_double(params.T);

        // Hash the integer type
        hash ^= static_cast<size_t>(params.type);
        hash *= prime;
        
        return hash;
    }
};


/**
 * @brief Fast thread-local pricing cache with LRU eviction and TTL
 */
class FastPricingCache {
public:
    /**
     * @brief Constructor
     * * @param maxSize Maximum cache size (0 for unlimited)
     * @param ttlMs Time-to-live in milliseconds (0 for no expiration)
     */
    FastPricingCache(size_t maxSize = 50000, int64_t ttlMs = 30000) // Default TTL 30s
        : maxSize_(maxSize), ttlMs_(ttlMs > 0 ? std::chrono::milliseconds(ttlMs) : std::chrono::milliseconds::zero()) {}
        
    /**
     * @brief Clear the cache
     */
    void clear() {
        cache_.clear();
        lruTracker_.clear();
    }
    
    /**
     * @brief Get the cache size
     * @return Number of items in cache
     */
    size_t size() const {
        // Note: Doesn't acquire lock, assumes reads are atomic enough for size check
        return cache_.size();
    }
    
    /**
     * @brief Get a price from the cache
     * * @param key Option parameters key
     * @param value Reference to store the price
     * @return True if price was found in cache and not expired, false otherwise
     */
    bool get(const OptionParams& key, double& value) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return false; // Not found
        }
        
        // Check expiration if TTL is enabled
        auto now = std::chrono::steady_clock::now();
        if (ttlMs_.count() > 0) {
            if ((now - it->second.timestamp) > ttlMs_) {
                // Item is expired, remove it
                lruTracker_.erase(it->second.lruIt);
                cache_.erase(it);
                return false;
            }
        }
        
        // Found and not expired, update LRU
        // Move accessed item to the front of the LRU list
        lruTracker_.splice(lruTracker_.begin(), lruTracker_, it->second.lruIt); 
        value = it->second.value;
        return true;
    }
    
    /**
     * @brief Cache a price
     * * @param key Option parameters key
     * @param value Price to cache
     */
    void put(const OptionParams& key, double value) {
         auto it = cache_.find(key);
         auto now = std::chrono::steady_clock::now();

        if (it != cache_.end()) {
            // Key exists, update value and move to front of LRU
            lruTracker_.splice(lruTracker_.begin(), lruTracker_, it->second.lruIt);
            it->second.value = value;
            it->second.timestamp = now; // Update timestamp on overwrite
            return;
        }

        // Key doesn't exist, check if eviction is needed
        if (maxSize_ > 0 && cache_.size() >= maxSize_) {
            // Evict the least recently used item (at the back of the list)
            if (!lruTracker_.empty()) {
                 cache_.erase(lruTracker_.back()); // Erase from map using key
                 lruTracker_.pop_back();          // Erase from list
            }
        }
        
        // Insert new item at the front
        lruTracker_.push_front(key);
        cache_[key] = {value, now, lruTracker_.begin()};
    }
        
    /**
     * @brief Batch lookup of multiple keys
     * * @param keys Vector of option parameter keys
     * @return Vector of cache hits
     */
    std::vector<CacheHit> batchLookup(const std::vector<OptionParams>& keys) {
        std::vector<CacheHit> results;
        results.reserve(keys.size()); // Reserve space
        auto now = std::chrono::steady_clock::now();
        std::vector<typename std::list<OptionParams>::iterator> to_move_to_front; // Track iterators to update LRU

        for (size_t i = 0; i < keys.size(); ++i) {
            auto it = cache_.find(keys[i]);
            if (it != cache_.end()) {
                 // Check expiration
                 if (ttlMs_.count() > 0 && (now - it->second.timestamp) > ttlMs_) {
                      // Expired, skip and schedule removal (removal happens on next write/get or explicit clean)
                 } else {
                     results.push_back({i, it->second.value, true});
                     // Mark for LRU update - avoid modifying list while iterating map potentially
                     to_move_to_front.push_back(it->second.lruIt);
                 }
            }
        }

        // Update LRU status for all hits after lookup is complete
        for(auto& lruIt : to_move_to_front) {
             lruTracker_.splice(lruTracker_.begin(), lruTracker_, lruIt);
        }
        
        return results;
    }
    
    /**
     * @brief Set the maximum cache size
     * * @param maxSize Maximum number of items in cache
     */
    void setMaxSize(size_t maxSize) {
        maxSize_ = maxSize;
        // Evict items if new size is smaller than current size
        while (maxSize_ > 0 && cache_.size() > maxSize_) {
            if (!lruTracker_.empty()) {
                 cache_.erase(lruTracker_.back());
                 lruTracker_.pop_back();
            } else {
                break; // Should not happen if cache_ is not empty
            }
        }
    }
    
    /**
     * @brief Set the time-to-live for cache items
     * * @param ttlMs Time-to-live in milliseconds (0 for no expiration)
     */
    void setTTL(int64_t ttlMs) {
         ttlMs_ = (ttlMs > 0) ? std::chrono::milliseconds(ttlMs) : std::chrono::milliseconds::zero();
    }
    
    /**
     * @brief Remove expired items from the cache
     * * @return Number of items removed
     */
    size_t removeExpired() {
        if (ttlMs_.count() <= 0 || cache_.empty()) {
            return 0;
        }
        
        auto now = std::chrono::steady_clock::now();
        size_t removed = 0;
        auto it = lruTracker_.end();
        
        // Iterate from least recently used (back)
        while (it != lruTracker_.begin()) {
            --it; // Move to the actual last element first
            auto mapIt = cache_.find(*it);
            if (mapIt != cache_.end()) {
                 if ((now - mapIt->second.timestamp) > ttlMs_) {
                    auto key_to_remove = *it; // Copy key before iterator invalidation
                    it = lruTracker_.erase(it); // Erase returns iterator to the next element
                    cache_.erase(key_to_remove);
                    removed++;
                    // Need to restart check potentially if erase invalidates 'it' relative to begin()
                    // Safest is often to rebuild or just check periodically
                 } 
                 // No else needed, if not expired, keep going towards the front (more recent)
            } else {
                 // Inconsistency: item in list but not map. Remove from list.
                 it = lruTracker_.erase(it);
            }
             // Need careful iterator handling if erasing from the middle/back
             // For simplicity in a background task, one might iterate the map
             // and collect expired keys, then erase them. Here, we iterate the list.
             // If erase(it) makes it point to begin(), the loop condition handles it.
        }
         // Check the first element too if it wasn't already checked
         if (!lruTracker_.empty()) {
              auto mapIt = cache_.find(lruTracker_.front());
              if (mapIt != cache_.end() && (now - mapIt->second.timestamp) > ttlMs_) {
                   cache_.erase(lruTracker_.front());
                   lruTracker_.pop_front();
                   removed++;
              }
         }

        return removed;
    }
    
private:
     struct CacheEntry {
        double value;
        std::chrono::steady_clock::time_point timestamp; // Time when inserted/updated
        typename std::list<OptionParams>::iterator lruIt; // Iterator into the LRU tracking list
    };
    
    std::unordered_map<OptionParams, CacheEntry, OptionParamsHash> cache_;
    std::list<OptionParams> lruTracker_; // Front=MRU, Back=LRU
    size_t maxSize_;
    std::chrono::milliseconds ttlMs_;

    // Note: This thread-local cache doesn't need internal mutexes.
    // Synchronization would be needed if accessing this specific instance 
    // from multiple threads, but getThreadLocalCache() ensures each thread has its own.
};


/**
 * @brief Get the thread-local cache instance
 * * @return Reference to thread-local cache
 */
inline FastPricingCache& getThreadLocalCache() {
    // Initialize with potentially different settings per thread if desired,
    // or keep defaults as set in the constructor.
    thread_local FastPricingCache thread_cache; 
    return thread_cache;
}


// --- Tiered Cache (Simplified) ---

/**
 * @brief Multi-tier cache (Simplified to use ONLY Thread-Local)
 * * This class acts as the main interface but now directly uses the
 * high-performance thread-local cache, bypassing shared/approximate tiers
 * to minimize overhead based on performance testing.
 */
class TieredPricingCache {
public:
    /**
     * @brief Constructor (does nothing extra now)
     */
    TieredPricingCache() = default; 
    
    /**
     * @brief Get a price from the cache (uses Thread-Local ONLY)
     * * @param key Option parameters key
     * @param value Reference to store the price
     * @return True if price was found in cache, false otherwise
     */
    bool get(const OptionParams& key, double& value) {
        // Directly use the thread-local cache
        return getThreadLocalCache().get(key, value);
    }
    
    /**
     * @brief Put a price into the cache (uses Thread-Local ONLY)
     * * @param key Option parameters key
     * @param value Price to cache
     */
    void put(const OptionParams& key, double value) {
        // Directly store in the thread-local cache
        getThreadLocalCache().put(key, value);
    }
    
    /**
     * @brief Clear all caches (clears Thread-Local ONLY)
     */
    void clear() {
        // Note: This only clears the calling thread's local cache.
        // A mechanism to clear all threads' caches would be more complex.
        getThreadLocalCache().clear(); 
    }
    
    /**
     * @brief Get the calling thread's local cache size
     * * @return Number of items in thread-local cache
     */
    size_t localSize() const {
         // Note: This only returns the size for the calling thread.
        return getThreadLocalCache().size();
    }
    
    /**
     * @brief Batch lookup of multiple keys (uses Thread-Local ONLY)
     * * @param keys Vector of option parameter keys
     * @return Vector of cache hits
     */
    std::vector<CacheHit> batchLookup(const std::vector<OptionParams>& keys) {
        // Directly use the thread-local batch lookup
        return getThreadLocalCache().batchLookup(keys);
    }
    
    // --- Methods related to Shared/Approximate cache are removed or no-oped ---
    
    size_t sharedSize() const { return 0; } // No shared cache used

    // Keep placeholder methods if needed for interface compatibility elsewhere,
    // but they should do nothing. Or remove them if not needed.
    // void clearShared() {}
    // void clearApproximate() {}
    
private:
    // Shared and Approximate caches are no longer members or are unused
    // std::shared_ptr<SharedLRUCache> sharedCache_; // Removed
    // std::shared_ptr<ApproximateCache> approxCache_; // Removed
};

/**
 * @brief Singleton instance of the (simplified) tiered pricing cache
 * * Provides access to the cache system, which now primarily uses thread-local storage.
 * * @return Reference to the global tiered pricing cache
 */
inline TieredPricingCache& getTieredPricingCache() {
    // This instance now mainly manages access patterns to thread-local caches
    static TieredPricingCache instance; 
    return instance;
}


// --- Utility Functions ---

/**
 * @brief Get or compute a value using the Tiered Cache (now thread-local focused)
 * * @param key Option parameters key
 * @param computeFunc Function to compute the value if not in cache
 * @return Cached or computed value
 */
inline double getCachedPrice(const OptionParams& key, const std::function<double()>& computeFunc) {
    double value;
    
    // Use the simplified tiered cache interface
    if (getTieredPricingCache().get(key, value)) {
        return value;
    }
    
    // Value not in cache, compute it
    value = computeFunc();
    
    // Store in cache
    getTieredPricingCache().put(key, value);
    
    return value;
}

/**
 * @brief Helper function to initialize thread-local caches on worker threads if needed
 * * Call this at the beginning of each thread's execution if specific per-thread
 * initialization (like different sizes/TTLs) is desired beyond defaults.
 */
void initializeThreadLocalCache(); // Declaration, definition in cache.cpp

/**
 * @brief Helper function to warm up caches with common parameter sets
 * * @param engine ALO engine instance (const reference) to use for pricing
 */
 void warmupCache(const ALOEngine& engine); // Declaration, definition in cache.cpp


// --- Legacy ICache Interface and Implementations (Optional - kept for reference/compatibility) ---

/**
 * @brief Interface for cache implementation (legacy support)
 * * @tparam K Key type
 * @tparam V Value type
 */
template <typename K, typename V>
class ICache {
public:
    virtual ~ICache() = default;
    virtual bool get(const K& key, V& value) const = 0;
    virtual void put(const K& key, const V& value) = 0;
    virtual void clear() = 0;
    virtual size_t size() const = 0;
};

/**
 * @brief Simple thread-safe cache implementation (legacy support)
 */
template <typename K, typename V>
class SimpleCache : public ICache<K, V> {
 // (Implementation remains the same as in current src/engine/alo/opt/cache.h)
 public:
     explicit SimpleCache(size_t maxSize = 0) : maxSize_(maxSize) {}
    
     bool get(const K& key, V& value) const override {
         std::lock_guard<std::mutex> lock(mutex_); // Use basic mutex
         auto it = cache_.find(key);
         if (it != cache_.end()) {
             value = it->second;
             return true;
         }
         return false;
     }
    
     void put(const K& key, const V& value) override {
         std::lock_guard<std::mutex> lock(mutex_);
         if (maxSize_ > 0 && cache_.size() >= maxSize_ && cache_.find(key) == cache_.end()) {
             // Simple eviction: just don't add if full (or could clear randomly/oldest)
             // For simplicity, just don't add if full.
             return; 
         }
         cache_[key] = value;
     }
    
     void clear() override {
         std::lock_guard<std::mutex> lock(mutex_);
         cache_.clear();
     }
    
     size_t size() const override {
         std::lock_guard<std::mutex> lock(mutex_);
         return cache_.size();
     }
    
 protected:
     std::unordered_map<K, V> cache_;
     mutable std::mutex mutex_; // Use basic mutex for simplicity here
     size_t maxSize_;
};

/**
 * @brief Legacy cache for option pricing results (string-based keys)
 */
class PricingCache : public SimpleCache<std::string, double> {
 // (Implementation remains the same as in current src/engine/alo/opt/cache.h)
 public:
     explicit PricingCache(size_t maxSize = 10000) : SimpleCache<std::string, double>(maxSize) {}
    
     static std::string generateKey(double S, double K, double r, double q, double vol, double T,
                                   const std::string& type, const std::string& style) {
         std::ostringstream key_stream;
         key_stream << std::fixed << std::setprecision(10) // Using high precision for key
                   << S << "_" << K << "_" << r << "_" << q << "_" << vol << "_" << T << "_" << type << "_" << style;
         return key_stream.str();
     }
};

/**
 * @brief Singleton instance of the legacy pricing cache
 */
inline PricingCache& getPricingCache() {
    static PricingCache instance;
    return instance;
}


} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_CACHE_H