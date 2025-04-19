/**
 * @file cache.cpp
 * @brief Implementation of cache optimization for the ALO engine
 */
 #include "cache.h"
 #include <chrono>
 #include <algorithm>
 #include <unordered_map>
 #include <mutex>
 #include <shared_mutex>
 
 namespace engine {
 namespace alo {
 namespace opt {
 
 /**
  * @brief Time-bounded cache with LRU eviction policy
  * 
  * This class implements a cache with time-based expiration and
  * least-recently-used (LRU) eviction policy when the cache gets full.
  */
 template <typename K, typename V>
 class TimeBoundedCache : public ICache<K, V> {
 public:
     /**
      * @brief Constructor
      * 
      * @param maxSize Maximum number of items in the cache (0 for unlimited)
      * @param maxAgeMs Maximum age of items in milliseconds (0 for no expiration)
      */
     TimeBoundedCache(size_t maxSize = 10000, int64_t maxAgeMs = 0)
         : maxSize_(maxSize), maxAgeMs_(maxAgeMs) {}
     
     /**
      * @brief Get a value from the cache
      * 
      * @param key Cache key
      * @param value Reference to store the value
      * @return True if key exists in cache and is not expired, false otherwise
      */
     bool get(const K& key, V& value) const override {
         std::shared_lock<std::shared_mutex> lock(mutex_);
         
         auto it = cache_.find(key);
         if (it == cache_.end()) {
             return false;
         }
         
         // Check if the item is expired
         if (maxAgeMs_ > 0) {
             auto now = std::chrono::steady_clock::now();
             auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                 now - it->second.timestamp).count();
             
             if (age > maxAgeMs_) {
                 // Item is expired, but we can't modify the cache here
                 // with a shared lock
                 return false;
             }
         }
         
         // Update access time (this is a const method, so we need to cast away constness)
         auto& nonConstCache = const_cast<Cache&>(cache_);
         nonConstCache[key].lastAccess = std::chrono::steady_clock::now();
         
         value = it->second.value;
         return true;
     }
     
     /**
      * @brief Put a value into the cache
      * 
      * @param key Cache key
      * @param value Value to cache
      */
     void put(const K& key, const V& value) override {
         std::unique_lock<std::shared_mutex> lock(mutex_);
         
         // Check if we need to evict items
         if (maxSize_ > 0 && cache_.size() >= maxSize_ && cache_.find(key) == cache_.end()) {
             evictLRU();
         }
         
         // Add or update the item
         auto now = std::chrono::steady_clock::now();
         cache_[key] = {value, now, now};
     }
     
     /**
      * @brief Clear the cache
      */
     void clear() override {
         std::unique_lock<std::shared_mutex> lock(mutex_);
         cache_.clear();
     }
     
     /**
      * @brief Get the size of the cache
      * 
      * @return Number of items in the cache
      */
     size_t size() const override {
         std::shared_lock<std::shared_mutex> lock(mutex_);
         return cache_.size();
     }
     
     /**
      * @brief Remove expired items from the cache
      * 
      * @return Number of items removed
      */
     size_t removeExpired() {
         if (maxAgeMs_ <= 0) {
             return 0;  // No expiration policy
         }
         
         std::unique_lock<std::shared_mutex> lock(mutex_);
         auto now = std::chrono::steady_clock::now();
         size_t removedCount = 0;
         
         for (auto it = cache_.begin(); it != cache_.end();) {
             auto age = std::chrono::duration_cast<std::chrono::milliseconds>(
                 now - it->second.timestamp).count();
             
             if (age > maxAgeMs_) {
                 it = cache_.erase(it);
                 removedCount++;
             } else {
                 ++it;
             }
         }
         
         return removedCount;
     }
     
 private:
     /**
      * @brief Evict the least recently used items from the cache
      * 
      * @param count Number of items to evict (default: 1)
      */
     void evictLRU(size_t count = 1) {
         if (cache_.empty()) {
             return;
         }
         
         // Find the LRU items
         std::vector<K> lruKeys;
         lruKeys.reserve(count);
         
         for (size_t i = 0; i < count && i < cache_.size(); ++i) {
             auto lruIt = cache_.begin();
             auto oldestAccess = lruIt->second.lastAccess;
             
             for (auto it = cache_.begin(); it != cache_.end(); ++it) {
                 if (it->second.lastAccess < oldestAccess) {
                     oldestAccess = it->second.lastAccess;
                     lruIt = it;
                 }
             }
             
             lruKeys.push_back(lruIt->first);
             if (i < count - 1) {  // Don't erase until we've found all LRU keys
                 cache_.erase(lruIt);
             }
         }
         
         // Erase the LRU items
         for (const auto& key : lruKeys) {
             cache_.erase(key);
         }
     }
     
     struct CacheEntry {
         V value;
         std::chrono::steady_clock::time_point timestamp;
         std::chrono::steady_clock::time_point lastAccess;
     };
     
     using Cache = std::unordered_map<K, CacheEntry>;
     
     Cache cache_;
     mutable std::shared_mutex mutex_;
     size_t maxSize_;
     int64_t maxAgeMs_;
 };
 
 /**
  * @brief Thread-local cache for the current calculation
  * 
  * This provides very fast caching for the current calculation thread,
  * avoiding the need for locks in most cases.
  */
 class ThreadLocalPricingCache {
 public:
     /**
      * @brief Get the thread-local instance
      * 
      * @return Reference to the thread-local cache
      */
     static ThreadLocalPricingCache& instance() {
         thread_local ThreadLocalPricingCache cache;
         return cache;
     }
     
     /**
      * @brief Get a cached price
      * 
      * @param key Cache key
      * @param price Reference to store the price
      * @return True if price was found in cache, false otherwise
      */
     bool getPrice(const std::string& key, double& price) const {
         auto it = cache_.find(key);
         if (it != cache_.end()) {
             price = it->second;
             return true;
         }
         return false;
     }
     
     /**
      * @brief Cache a price
      * 
      * @param key Cache key
      * @param price Price to cache
      */
     void cachePrice(const std::string& key, double price) {
         cache_[key] = price;
         
         // If cache gets too large, clear it
         if (cache_.size() > 10000) {
             cache_.clear();
         }
     }
     
     /**
      * @brief Clear the cache
      */
     void clear() {
         cache_.clear();
     }
     
     /**
      * @brief Get the cache size
      * 
      * @return Number of items in cache
      */
     size_t size() const {
         return cache_.size();
     }
     
 private:
     ThreadLocalPricingCache() = default;
     
     std::unordered_map<std::string, double> cache_;
 };
 
 /**
  * @brief Multi-tier cache implementation
  * 
  * This cache combines thread-local and global caches for best performance.
  */
 class MultiTierPricingCache : public PricingCache {
 public:
     /**
      * @brief Constructor
      * 
      * @param maxSize Maximum size of the global cache
      */
     explicit MultiTierPricingCache(size_t maxSize = 100000)
         : PricingCache(maxSize) {}
     
     /**
      * @brief Get a price from the cache
      * 
      * Checks thread-local cache first, then global cache.
      * 
      * @param key Cache key
      * @param value Reference to store the price
      * @return True if price was found in cache, false otherwise
      */
     bool get(const std::string& key, double& value) const override {
         // First check thread-local cache (no locks needed)
         if (ThreadLocalPricingCache::instance().getPrice(key, value)) {
             return true;
         }
         
         // If not found, check the global cache
         if (PricingCache::get(key, value)) {
             // Store in thread-local cache for future lookups
             ThreadLocalPricingCache::instance().cachePrice(key, value);
             return true;
         }
         
         return false;
     }
     
     /**
      * @brief Put a price in the cache
      * 
      * Stores in both thread-local and global caches.
      * 
      * @param key Cache key
      * @param value Price to cache
      */
     void put(const std::string& key, const double& value) override {
         // Store in thread-local cache
         ThreadLocalPricingCache::instance().cachePrice(key, value);
         
         // Store in global cache
         PricingCache::put(key, value);
     }
     
     /**
      * @brief Clear all caches
      */
     void clear() override {
         // Clear thread-local cache
         ThreadLocalPricingCache::instance().clear();
         
         // Clear global cache
         PricingCache::clear();
     }
 };
 
 /**
  * @brief Singleton instance of the multi-tier pricing cache
  * 
  * This is the preferred cache to use for the ALO engine.
  * 
  * @return Reference to the global multi-tier pricing cache
  */
 inline PricingCache& getMultiTierPricingCache() {
     static MultiTierPricingCache instance;
     return instance;  // This is legal because MultiTierPricingCache is a subclass of PricingCache
 }
 
 /**
  * @brief Factory function to create appropriate type of cache
  * 
  * @param maxSize Maximum cache size
  * @param maxAgeMs Maximum age in milliseconds
  * @return Cache instance
  */
 template <typename K, typename V>
 std::unique_ptr<ICache<K, V>> createCache(size_t maxSize = 0, int64_t maxAgeMs = 0) {
     if (maxAgeMs > 0) {
         return std::make_unique<TimeBoundedCache<K, V>>(maxSize, maxAgeMs);
     } else {
         return std::make_unique<SimpleCache<K, V>>(maxSize);
     }
 }
 
 /**
  * @brief Get the recommended pricing cache based on workload
  * 
  * @param highConcurrency True if high concurrency is expected
  * @param largeWorkload True if large number of calculations is expected
  * @return Reference to appropriate cache instance
  */
 PricingCache& getRecommendedPricingCache(bool highConcurrency, bool largeWorkload) {
     if (highConcurrency) {
         if (largeWorkload) {
             return getMultiTierPricingCache();
         } else {
             return getPricingCache();
         }
     } else {
         return getPricingCache();
     }
 }
 
 } // namespace opt
 } // namespace alo
 } // namespace engine