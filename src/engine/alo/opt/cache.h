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

namespace engine {
namespace alo {
namespace opt {

/**
 * @brief Option parameters structure for efficient caching
 */
struct OptionParams {
    double S, K, r, q, vol, T;
    int type; // PUT=0, CALL=1, etc.
    
    // Equality operator for hash map lookups
    bool operator==(const OptionParams& other) const {
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
    size_t index;
    double value;
    bool found;
};

/**
 * @brief Hash function for OptionParams
 */
struct OptionParamsHash {
    size_t operator()(const OptionParams& params) const {
        // FNV-1a hash algorithm - much faster than string concatenation
        size_t hash = 0x811c9dc5;
        auto hash_double = [&hash](double val) {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(&val);
            for (size_t i = 0; i < sizeof(double); ++i) {
                hash ^= data[i];
                hash *= 0x01000193;
            }
        };
        
        hash_double(params.S);
        hash_double(params.K);
        hash_double(params.r);
        hash_double(params.q);
        hash_double(params.vol);
        hash_double(params.T);
        hash ^= params.type;
        hash *= 0x01000193;
        
        return hash;
    }
};

/**
 * @brief Approximate matching function for cache lookups
 */
struct ApproxMatchParams {
    // Tolerance for parameter matching
    double epsilon = 1e-10;
    
    // Controls whether to match parameters approximately
    bool match_spot = false;
    bool match_strike = false;
    bool match_rate = true;
    bool match_dividend = true;
    bool match_vol = false; 
    bool match_time = false;
    
    // Require exact option type match
    bool exact_type = true;
};

/**
 * @brief Fast thread-local pricing cache with binary keys
 */
class FastPricingCache {
public:
    /**
     * @brief Constructor
     * 
     * @param maxSize Maximum cache size (0 for unlimited)
     * @param ttlMs Time-to-live in milliseconds (0 for no expiration)
     */
    FastPricingCache(size_t maxSize = 10000, int64_t ttlMs = 0)
        : maxSize_(maxSize), ttlMs_(ttlMs) {}
        
    /**
     * @brief Clear the cache
     */
    void clear() {
        if (ttlMs_ > 0) {
            removeExpired();
        }
        cache_.clear();
    }
    
    /**
     * @brief Get the cache size
     * @return Number of items in cache
     */
    size_t size() const {
        return cache_.size();
    }
    
    /**
     * @brief Get a price from the cache
     * 
     * @param key Option parameters key
     * @param value Reference to store the price
     * @return True if price was found in cache, false otherwise
     */
    bool get(const OptionParams& key, double& value) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            return false;
        }
        
        // Check expiration if TTL is enabled
        if (ttlMs_ > 0) {
            auto now = std::chrono::steady_clock::now();
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second.timestamp).count();
            
            if (age > ttlMs_) {
                // Item is expired
                cache_.erase(it);
                return false;
            }
        }
        
        value = it->second.value;
        it->second.lastAccess = std::chrono::steady_clock::now();
        return true;
    }
    
    /**
     * @brief Cache a price
     * 
     * @param key Option parameters key
     * @param value Price to cache
     */
    void put(const OptionParams& key, double value) {
        // Only cache if cache is not too large
        if (maxSize_ > 0 && cache_.size() >= maxSize_) {
            // Eviction strategy: remove least recently used items
            evictLRUItems(maxSize_ / 10); // Remove 10% of oldest items
        }
        
        auto now = std::chrono::steady_clock::now();
        cache_[key] = {value, now, now};
    }
    
    /**
     * @brief Get a price using approximate matching
     * 
     * @param key Option parameters key
     * @param value Reference to store the price
     * @param approx Approximation parameters
     * @return True if a matching price was found, false otherwise
     */
    bool getApproximate(const OptionParams& key, double& value, const ApproxMatchParams& approx) {
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            if (matchesApprox(it->first, key, approx)) {
                value = it->second.value;
                it->second.lastAccess = std::chrono::steady_clock::now();
                return true;
            }
        }
        return false;
    }
    
    /**
     * @brief Batch lookup of multiple keys
     * 
     * @param keys Vector of option parameter keys
     * @return Vector of cache hits
     */
    std::vector<CacheHit> batchLookup(const std::vector<OptionParams>& keys) {
        std::vector<CacheHit> results;
        results.reserve(keys.size());
        
        for (size_t i = 0; i < keys.size(); ++i) {
            double value = 0.0;
            bool found = get(keys[i], value);
            
            if (found) {
                results.push_back({i, value, true});
            }
        }
        
        return results;
    }
    
    /**
     * @brief Set the maximum cache size
     * 
     * @param maxSize Maximum number of items in cache
     */
    void setMaxSize(size_t maxSize) {
        maxSize_ = maxSize;
        
        // If new size is smaller than current size, evict items
        if (maxSize_ > 0 && cache_.size() > maxSize_) {
            evictLRUItems(cache_.size() - maxSize_);
        }
    }
    
    /**
     * @brief Set the time-to-live for cache items
     * 
     * @param ttlMs Time-to-live in milliseconds (0 for no expiration)
     */
    void setTTL(int64_t ttlMs) {
        ttlMs_ = ttlMs;
        
        // Remove expired items if TTL is enabled
        if (ttlMs_ > 0) {
            removeExpired();
        }
    }
    
    /**
     * @brief Remove expired items from the cache
     * 
     * @return Number of items removed
     */
    size_t removeExpired() {
        if (ttlMs_ <= 0) {
            return 0;
        }
        
        auto now = std::chrono::steady_clock::now();
        size_t removed = 0;
        
        for (auto it = cache_.begin(); it != cache_.end();) {
            auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - it->second.timestamp).count();
            
            if (age > ttlMs_) {
                it = cache_.erase(it);
                removed++;
            } else {
                ++it;
            }
        }
        
        return removed;
    }
    
private:
    struct CacheEntry {
        double value;
        std::chrono::steady_clock::time_point timestamp;
        std::chrono::steady_clock::time_point lastAccess;
    };
    
    /**
     * @brief Check if two sets of parameters match approximately
     */
    bool matchesApprox(const OptionParams& a, const OptionParams& b, const ApproxMatchParams& approx) const {
        // Type always needs to match if exact_type is true
        if (approx.exact_type && a.type != b.type) {
            return false;
        }
        
        // Check each parameter according to matching rules
        if (approx.match_spot) {
            if (std::abs(a.S - b.S) > approx.epsilon * std::max(1.0, std::abs(a.S))) {
                return false;
            }
        } else if (a.S != b.S) {
            return false;
        }
        
        if (approx.match_strike) {
            if (std::abs(a.K - b.K) > approx.epsilon * std::max(1.0, std::abs(a.K))) {
                return false;
            }
        } else if (a.K != b.K) {
            return false;
        }
        
        if (approx.match_rate) {
            if (std::abs(a.r - b.r) > approx.epsilon * std::max(1.0, std::abs(a.r))) {
                return false;
            }
        } else if (a.r != b.r) {
            return false;
        }
        
        if (approx.match_dividend) {
            if (std::abs(a.q - b.q) > approx.epsilon * std::max(1.0, std::abs(a.q))) {
                return false;
            }
        } else if (a.q != b.q) {
            return false;
        }
        
        if (approx.match_vol) {
            if (std::abs(a.vol - b.vol) > approx.epsilon * a.vol) {
                return false;
            }
        } else if (a.vol != b.vol) {
            return false;
        }
        
        if (approx.match_time) {
            if (std::abs(a.T - b.T) > approx.epsilon * a.T) {
                return false;
            }
        } else if (a.T != b.T) {
            return false;
        }
        
        return true;
    }
    
    /**
     * @brief Evict least recently used items
     * 
     * @param count Number of items to evict
     */
    void evictLRUItems(size_t count) {
        if (cache_.empty() || count == 0) {
            return;
        }
        
        // Store pairs of (lastAccess, key) for sorting
        std::vector<std::pair<std::chrono::steady_clock::time_point, OptionParams>> items;
        items.reserve(cache_.size());
        
        for (const auto& entry : cache_) {
            items.emplace_back(entry.second.lastAccess, entry.first);
        }
        
        // Sort by access time (oldest first)
        std::sort(items.begin(), items.end(), 
            [](const auto& a, const auto& b) { return a.first < b.first; });
        
        // Remove oldest items
        size_t removeCount = std::min(count, items.size());
        for (size_t i = 0; i < removeCount; ++i) {
            cache_.erase(items[i].second);
        }
    }
    
    std::unordered_map<OptionParams, CacheEntry, OptionParamsHash> cache_;
    size_t maxSize_;
    int64_t ttlMs_;
};

/**
 * @brief Get the thread-local cache instance
 * 
 * @return Reference to thread-local cache
 */
inline FastPricingCache& getThreadLocalCache() {
    thread_local FastPricingCache cache;
    return cache;
}

/**
 * @brief Interface for cache implementation (legacy support)
 * 
 * @tparam K Key type
 * @tparam V Value type
 */
template <typename K, typename V>
class ICache {
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~ICache() = default;
    
    /**
     * @brief Get a value from the cache
     * 
     * @param key Cache key
     * @param value Reference to store the value
     * @return True if key exists in cache, false otherwise
     */
    virtual bool get(const K& key, V& value) const = 0;
    
    /**
     * @brief Put a value into the cache
     * 
     * @param key Cache key
     * @param value Value to cache
     */
    virtual void put(const K& key, const V& value) = 0;
    
    /**
     * @brief Clear the cache
     */
    virtual void clear() = 0;
    
    /**
     * @brief Get the size of the cache
     * 
     * @return Number of items in the cache
     */
    virtual size_t size() const = 0;
};

/**
 * @brief Shared LRU cache implementation with eviction policy
 */
class SharedLRUCache {
public:
    /**
     * @brief Constructor
     * 
     * @param maxSize Maximum cache size (0 for unlimited)
     */
    explicit SharedLRUCache(size_t maxSize = 100000) 
        : maxSize_(maxSize) {}
    
    /**
     * @brief Get a value from the cache
     * 
     * @param key Cache key
     * @param value Reference to store the value
     * @return True if key exists in cache, false otherwise
     */
    bool get(const OptionParams& key, double& value) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        auto it = keyMap_.find(key);
        if (it == keyMap_.end()) {
            return false;
        }
        
        // Move the accessed item to the front of the list (most recently used)
        auto listIt = it->second;
        
        // We need to modify the list, so upgrade to exclusive lock
        lock.unlock();
        std::unique_lock<std::shared_mutex> excl_lock(mutex_);
        
        // Check if the entry still exists after upgrading lock
        it = keyMap_.find(key);
        if (it == keyMap_.end()) {
            return false;
        }
        
        listIt = it->second;
        value = listIt->second;
        
        // Move to front - requires exclusive lock
        lruList_.erase(listIt);
        lruList_.push_front({key, value});
        keyMap_[key] = lruList_.begin();
        
        return true;
    }
    
    /**
     * @brief Put a value into the cache
     * 
     * @param key Cache key
     * @param value Value to cache
     */
    void put(const OptionParams& key, double value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        // Check if key already exists
        auto it = keyMap_.find(key);
        if (it != keyMap_.end()) {
            // Update existing entry and move to front
            lruList_.erase(it->second);
            lruList_.push_front({key, value});
            keyMap_[key] = lruList_.begin();
            return;
        }
        
        // If we've reached the maximum size, remove the least recently used item
        if (maxSize_ > 0 && lruList_.size() >= maxSize_) {
            // Remove the last element (least recently used)
            auto last = lruList_.end();
            --last;
            keyMap_.erase(last->first);
            lruList_.pop_back();
        }
        
        // Add new entry at the front
        lruList_.push_front({key, value});
        keyMap_[key] = lruList_.begin();
    }
    
    /**
     * @brief Clear the cache
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        keyMap_.clear();
        lruList_.clear();
    }
    
    /**
     * @brief Get the cache size
     * 
     * @return Number of items in cache
     */
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return lruList_.size();
    }
    
    /**
     * @brief Batch lookup of multiple keys
     * 
     * @param keys Vector of option parameter keys
     * @return Vector of cache hits
     */
    std::vector<CacheHit> batchLookup(const std::vector<OptionParams>& keys) {
        std::vector<CacheHit> results;
        results.reserve(keys.size());
        
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        for (size_t i = 0; i < keys.size(); ++i) {
            auto it = keyMap_.find(keys[i]);
            if (it != keyMap_.end()) {
                results.push_back({i, it->second->second, true});
            }
        }
        
        // We don't update LRU order for batch lookups to avoid lock contention
        // Individual gets will update access order
        
        return results;
    }
    
private:
    using KeyValuePair = std::pair<OptionParams, double>;
    using LRUList = std::list<KeyValuePair>;
    using KeyMap = std::unordered_map<OptionParams, LRUList::iterator, OptionParamsHash>;
    
    LRUList lruList_;
    KeyMap keyMap_;
    mutable std::shared_mutex mutex_;
    size_t maxSize_;
};

/**
 * @brief Approximate cache for parameter interpolation
 */
class ApproximateCache {
public:
    /**
     * @brief Constructor
     * 
     * @param maxSize Maximum cache size (0 for unlimited)
     */
    explicit ApproximateCache(size_t maxSize = 10000)
        : maxSize_(maxSize) {}
    
    /**
     * @brief Get an approximate value from the cache
     * 
     * @param key Option parameters key
     * @param value Reference to store the value
     * @param approx Approximation parameters
     * @return True if a matching value was found, false otherwise
     */
    bool get(const OptionParams& key, double& value, const ApproxMatchParams& approx) {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        
        // Epsilon matching
        for (const auto& entry : cache_) {
            if (matches(entry.first, key, approx)) {
                value = entry.second;
                return true;
            }
        }
        
        return false;
    }
    
    /**
     * @brief Put a value into the cache
     * 
     * @param key Option parameters key
     * @param value Value to cache
     */
    void put(const OptionParams& key, double value) {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        
        // If we've reached the maximum size, remove a random element
        if (maxSize_ > 0 && cache_.size() >= maxSize_) {
            // Just remove the first element
            auto it = cache_.begin();
            if (it != cache_.end()) {
                cache_.erase(it);
            }
        }
        
        cache_[key] = value;
    }
    
    /**
     * @brief Clear the cache
     */
    void clear() {
        std::unique_lock<std::shared_mutex> lock(mutex_);
        cache_.clear();
    }
    
    /**
     * @brief Get the cache size
     * 
     * @return Number of items in cache
     */
    size_t size() const {
        std::shared_lock<std::shared_mutex> lock(mutex_);
        return cache_.size();
    }
    
private:
    /**
     * @brief Check if parameters match approximately
     */
    bool matches(const OptionParams& a, const OptionParams& b, const ApproxMatchParams& approx) const {
        // Type always needs to match
        if (a.type != b.type) {
            return false;
        }
        
        // Based on epsilon from approximate parameters
        double eps = approx.epsilon;
        
        // All parameters must be within epsilon relative difference
        if (std::abs(a.S - b.S) > eps * std::max(1.0, std::abs(a.S))) return false;
        if (std::abs(a.K - b.K) > eps * std::max(1.0, std::abs(a.K))) return false;
        if (std::abs(a.r - b.r) > eps * std::max(0.001, std::abs(a.r))) return false;
        if (std::abs(a.q - b.q) > eps * std::max(0.001, std::abs(a.q))) return false;
        if (std::abs(a.vol - b.vol) > eps * a.vol) return false;
        if (std::abs(a.T - b.T) > eps * a.T) return false;
        
        return true;
    }
    
    std::unordered_map<OptionParams, double, OptionParamsHash> cache_;
    mutable std::shared_mutex mutex_;
    size_t maxSize_;
};

/**
 * @brief Multi-tier cache combining thread-local, shared, and approximate caches
 */
class TieredPricingCache {
public:
    /**
     * @brief Constructor
     */
    TieredPricingCache()
        : sharedCache_(std::make_shared<SharedLRUCache>()),
          approxCache_(std::make_shared<ApproximateCache>()) {}
    
    /**
     * @brief Get a price from the cache
     * 
     * @param key Option parameters key
     * @param value Reference to store the price
     * @return True if price was found in cache, false otherwise
     */
    bool get(const OptionParams& key, double& value) {
        // First check thread-local cache (fastest)
        if (getThreadLocalCache().get(key, value)) {
            return true;
        }
        
        // Next check shared cache
        if (sharedCache_->get(key, value)) {
            // Found in shared cache, add to thread-local for future lookups
            getThreadLocalCache().put(key, value);
            return true;
        }
        
        // Try approximate matching last
        ApproxMatchParams approx;
        approx.epsilon = 1e-6; // Stricter epsilon for production use
        
        if (approxCache_->get(key, value, approx)) {
            return true;
        }
        
        return false;
    }
    
    /**
     * @brief Put a price into the cache
     * 
     * @param key Option parameters key
     * @param value Price to cache
     */
    void put(const OptionParams& key, double value) {
        // Store in all tiers
        getThreadLocalCache().put(key, value);
        sharedCache_->put(key, value);
        approxCache_->put(key, value);
    }
    
    /**
     * @brief Clear all caches
     */
    void clear() {
        getThreadLocalCache().clear();
        sharedCache_->clear();
        approxCache_->clear();
    }
    
    /**
     * @brief Get the shared cache size
     * 
     * @return Number of items in shared cache
     */
    size_t sharedSize() const {
        return sharedCache_->size();
    }
    
    /**
     * @brief Batch lookup of multiple keys
     * 
     * @param keys Vector of option parameter keys
     * @return Vector of cache hits
     */
    std::vector<CacheHit> batchLookup(const std::vector<OptionParams>& keys) {
        // First check thread-local cache
        auto localHits = getThreadLocalCache().batchLookup(keys);
        
        // If we found everything, return immediately
        if (localHits.size() == keys.size()) {
            return localHits;
        }
        
        // For keys not found in thread-local cache, check shared cache
        std::vector<OptionParams> remaining;
        std::vector<size_t> indices;
        
        // Track which keys were not found in thread-local cache
        std::vector<bool> found(keys.size(), false);
        for (const auto& hit : localHits) {
            found[hit.index] = true;
        }
        
        // Collect keys not found yet
        for (size_t i = 0; i < keys.size(); ++i) {
            if (!found[i]) {
                remaining.push_back(keys[i]);
                indices.push_back(i);
            }
        }
        
        // Check shared cache
        auto sharedHits = sharedCache_->batchLookup(remaining);
        
        // Add shared hits to result and update thread-local cache
        for (const auto& hit : sharedHits) {
            size_t originalIndex = indices[hit.index];
            localHits.push_back({originalIndex, hit.value, true});
            
            // Update thread-local cache for future lookups
            getThreadLocalCache().put(remaining[hit.index], hit.value);
            
            // Mark as found
            found[originalIndex] = true;
        }
        
        // We don't check approximate cache for batch lookups for determinism
        
        return localHits;
    }
    
private:
    std::shared_ptr<SharedLRUCache> sharedCache_;
    std::shared_ptr<ApproximateCache> approxCache_;
};

/**
 * @brief Singleton instance of the tiered pricing cache
 * 
 * @return Reference to the global tiered pricing cache
 */
inline TieredPricingCache& getTieredPricingCache() {
    static TieredPricingCache instance;
    return instance;
}

/**
 * @brief Simple thread-safe cache implementation (legacy support)
 * 
 * @tparam K Key type
 * @tparam V Value type
 */
template <typename K, typename V>
class SimpleCache : public ICache<K, V> {
public:
    /**
     * @brief Constructor
     * 
     * @param maxSize Maximum size of the cache (0 for unlimited)
     */
    explicit SimpleCache(size_t maxSize = 0) : maxSize_(maxSize) {}
    
    /**
     * @brief Get a value from the cache
     * 
     * @param key Cache key
     * @param value Reference to store the value
     * @return True if key exists in cache, false otherwise
     */
    bool get(const K& key, V& value) const override {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    
    /**
     * @brief Put a value into the cache
     * 
     * @param key Cache key
     * @param value Value to cache
     */
    void put(const K& key, const V& value) override {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // If we've reached the maximum size, don't add more items
        if (maxSize_ > 0 && cache_.size() >= maxSize_ && cache_.find(key) == cache_.end()) {
            return;
        }
        
        cache_[key] = value;
    }
    
    /**
     * @brief Clear the cache
     */
    void clear() override {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }
    
    /**
     * @brief Get the size of the cache
     * 
     * @return Number of items in the cache
     */
    size_t size() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.size();
    }
    
protected:
    std::unordered_map<K, V> cache_;
    mutable std::mutex mutex_;
    size_t maxSize_;
};

/**
 * @brief Legacy cache for option pricing results (string-based keys)
 */
class PricingCache : public SimpleCache<std::string, double> {
public:
    /**
     * @brief Constructor
     * 
     * @param maxSize Maximum size of the cache (0 for unlimited)
     */
    explicit PricingCache(size_t maxSize = 10000) : SimpleCache<std::string, double>(maxSize) {}
    
    /**
     * @brief Generate a cache key from option parameters (legacy)
     */
    static std::string generateKey(double S, double K, double r, double q, double vol, double T,
                                  const std::string& type, const std::string& style) {
        std::ostringstream key_stream;
        key_stream << std::fixed << std::setprecision(10)
                  << S << "_" << K << "_" << r << "_" << q << "_" << vol << "_" << T << "_" << type << "_" << style;
        return key_stream.str();
    }
};

/**
 * @brief Singleton instance of the pricing cache (legacy support)
 * 
 * @return Reference to the global pricing cache
 */
inline PricingCache& getPricingCache() {
    static PricingCache instance;
    return instance;
}

/**
 * @brief Get or compute a value with caching (utility function)
 * 
 * @tparam K Key type
 * @tparam V Value type
 * @param cache Cache to use
 * @param key Cache key
 * @param computeFunc Function to compute the value if not in cache
 * @return Cached or computed value
 */
template <typename K, typename V>
V getOrCompute(ICache<K, V>& cache, const K& key, const std::function<V()>& computeFunc) {
    V value;
    if (cache.get(key, value)) {
        return value;
    }
    
    value = computeFunc();
    cache.put(key, value);
    return value;
}

/**
 * @brief Fast cache or compute utility for OptionParams
 */
inline double getCachedPrice(const OptionParams& key, const std::function<double()>& computeFunc) {
    double value;
    
    // Try multi-tier cache first
    if (getTieredPricingCache().get(key, value)) {
        return value;
    }
    
    // Fall back to thread-local cache only for backwards compatibility
    auto& cache = getThreadLocalCache();
    
    if (cache.get(key, value)) {
        return value;
    }
    
    value = computeFunc();
    
    // Store in both caches
    cache.put(key, value);
    getTieredPricingCache().put(key, value);
    
    return value;
}

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_CACHE_H