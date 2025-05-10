#ifndef ENGINE_ALO_OPT_CACHE_H
#define ENGINE_ALO_OPT_CACHE_H

#include <unordered_map>
#include <functional> 
#include <cstdint>    

namespace engine {
namespace alo {
namespace opt {

/**
 * @brief Option parameters structure for efficient caching (double precision)
 */
struct OptionKeyDouble { // Renamed from OptionKey
    double S, K, r, q, vol, T;
    int type; // PUT=0, CALL=1, etc.
    
    bool operator==(const OptionKeyDouble& other) const {
        return S == other.S && K == other.K && r == other.r && 
               q == other.q && vol == other.vol && T == other.T && 
               type == other.type;
    }
};

/**
 * @brief Hash function for OptionKeyDouble (double precision)
 */
struct OptionKeyDoubleHash { // Renamed from OptionKeyHash
    std::size_t operator()(const OptionKeyDouble& params) const {
        std::size_t hash = 2166136261u; 
        auto hash_double = [&](double val) {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(&val);
            for (size_t i = 0; i < sizeof(double); ++i) {
                hash ^= static_cast<std::size_t>(data[i]);
                hash *= 16777619u; 
            }
        };
        hash_double(params.S);
        hash_double(params.K);
        hash_double(params.r);
        hash_double(params.q);
        hash_double(params.vol);
        hash_double(params.T);
        hash ^= (static_cast<std::size_t>(params.type) << 1);
        hash *= 16777619u;
        return hash;
    }
};

/**
 * @brief Get the thread-local cache instance for double-precision results
 * 
 * @return Reference to thread-local cache (double precision)
 */
inline std::unordered_map<OptionKeyDouble, double, OptionKeyDoubleHash>& getThreadLocalCacheDouble() { // Renamed
    thread_local static std::unordered_map<OptionKeyDouble, double, OptionKeyDoubleHash> cache_double; // Renamed variable
    return cache_double;
}

/**
 * @brief Option parameters structure for efficient caching (single precision)
 */
struct OptionKeySingle {
    float S, K, r, q, vol, T;
    int type; // PUT=0, CALL=1, etc.

    bool operator==(const OptionKeySingle& other) const {
        return S == other.S && K == other.K && r == other.r &&
               q == other.q && vol == other.vol && T == other.T &&
               type == other.type;
    }
};

/**
 * @brief Hash function for OptionKeySingle (single precision)
 */
struct OptionKeySingleHash {
    std::size_t operator()(const OptionKeySingle& params) const {
        std::size_t hash = 2166136261u;
        auto hash_float = [&](float val) {
            const uint8_t* data = reinterpret_cast<const uint8_t*>(&val);
            for (size_t i = 0; i < sizeof(float); ++i) {
                hash ^= static_cast<std::size_t>(data[i]);
                hash *= 16777619u;
            }
        };
        hash_float(params.S);
        hash_float(params.K);
        hash_float(params.r);
        hash_float(params.q);
        hash_float(params.vol);
        hash_float(params.T);
        hash ^= (static_cast<std::size_t>(params.type) << 1);
        hash *= 16777619u;
        return hash;
    }
};

/**
 * @brief Get the thread-local cache instance for single-precision results
 * 
 * @return Reference to thread-local cache (single precision)
 */
inline std::unordered_map<OptionKeySingle, float, OptionKeySingleHash>& getThreadLocalCacheSingle() {
    thread_local static std::unordered_map<OptionKeySingle, float, OptionKeySingleHash> cache_single;
    return cache_single;
}

} // namespace opt
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_OPT_CACHE_H