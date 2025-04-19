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
 * @brief Fast thread-local pricing cache with binary keys
 */
class FastPricingCache {
public:
    /**
     * @brief Clear the cache
     */
    void clear() {
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
    bool get(const OptionParams& key, double& value) const {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            value = it->second;
            return true;
        }
        return false;
    }
    
    /**
     * @brief Cache a price
     * 
     * @param key Option parameters key
     * @param value Price to cache
     */
    void put(const OptionParams& key, double value) {
        // Only cache if cache is not too large
        if (cache_.size() < maxSize_) {
            cache_[key] = value;
        } else if (cache_.size() >= maxSize_ && evictionCounter_++ % 10 == 0) {
            // Periodically clear 25% of the cache to prevent continuous resizing
            size_t toRemove = maxSize_ / 4;
            for (size_t i = 0; i < toRemove && !cache_.empty(); ++i) {
                cache_.erase(cache_.begin());
            }
        }
    }
    
    /**
     * @brief Set the maximum cache size
     * 
     * @param maxSize Maximum number of items in cache
     */
    void setMaxSize(size_t maxSize) {
        maxSize_ = maxSize;
    }
    
private:
    std::unordered_map<OptionParams, double, OptionParamsHash> cache_;
    size_t maxSize_ = 10000;
    size_t evictionCounter_ = 0;
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
    auto& cache = getThreadLocalCache();
    
    if (cache.get(key, value)) {
        return value;
    }
    
    value = computeFunc();
    cache.put(key, value);
    return value;
}

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_CACHE_H