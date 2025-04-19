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
 
 namespace engine {
 namespace alo {
 namespace opt {
 
 /**
  * @brief Interface for cache implementation
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
  * @brief Simple thread-safe cache implementation
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
     
 private:
     std::unordered_map<K, V> cache_;
     mutable std::mutex mutex_;
     size_t maxSize_;
 };
 
 /**
  * @brief Cache for option pricing results
  * 
  * This cache stores option pricing results keyed by option parameters.
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
      * @brief Generate a cache key from option parameters
      * 
      * @param S Spot price
      * @param K Strike price
      * @param r Risk-free rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity
      * @param type Option type string ("PUT", "CALL")
      * @param style Option style string ("AMERICAN", "EUROPEAN")
      * @return Cache key string
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
  * @brief Singleton instance of the pricing cache
  * 
  * @return Reference to the global pricing cache
  */
 inline PricingCache& getPricingCache() {
     static PricingCache instance;
     return instance;
 }
 
 /**
  * @brief Get or compute a value with caching
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
 
 } // namespace opt
 } // namespace alo
 } // namespace engine
 
 #endif // ENGINE_ALO_OPT_CACHE_H