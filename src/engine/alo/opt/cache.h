#ifndef ENGINE_ALO_OPT_CACHE_H
#define ENGINE_ALO_OPT_CACHE_H

#include <unordered_map>
#include <functional>
#include <cstdint>

namespace engine {
namespace alo {
namespace opt {

/**
 * @brief Option parameters structure for efficient caching
 */
struct OptionKey {
    double S, K, r, q, vol, T;
    int type; // PUT=0, CALL=1, etc.
    
    // Equality operator for hash map lookups
    bool operator==(const OptionKey& other) const {
        return S == other.S && K == other.K && r == other.r && 
               q == other.q && vol == other.vol && T == other.T && 
               type == other.type;
    }
};

/**
 * @brief Hash function for OptionKey
 */
struct OptionKeyHash {
    size_t operator()(const OptionKey& params) const {
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
 * @brief Get the thread-local cache instance
 * 
 * @return Reference to thread-local cache
 */
inline std::unordered_map<OptionKey, double, OptionKeyHash>& getThreadLocalCache() {
    thread_local std::unordered_map<OptionKey, double, OptionKeyHash> cache;
    return cache;
}

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_CACHE_H