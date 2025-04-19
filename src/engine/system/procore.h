#ifndef ENGINE_SYSTEM_PROCORE_H
#define ENGINE_SYSTEM_PROCORE_H

#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <thread>
#include <mutex>
#include <memory>
#include <bitset>
#include <cstdint>

namespace engine {
namespace system {

/**
 * @brief Type of processor core
 */
enum class CoreType {
    UNKNOWN,    // Unknown core type
    EFFICIENCY, // Efficiency core (e.g., ARM little cores, Intel E-cores)
    PERFORMANCE // Performance core (e.g., ARM big cores, Intel P-cores)
};

/**
 * @brief Information about a processor core
 */
struct CoreInfo {
    int id;                  // Core ID
    int socket;              // Socket/package ID
    int physical_id;         // Physical core ID
    int sibling;             // Hyperthreading sibling
    CoreType type;           // Core type
    int l1d_cache_size_kb;   // L1 data cache size in KB
    int l1i_cache_size_kb;   // L1 instruction cache size in KB
    int l2_cache_size_kb;    // L2 cache size in KB
    int l3_cache_size_kb;    // L3 cache size in KB (0 if not present)
    double max_frequency_mhz; // Maximum frequency in MHz
    std::string model_name;   // CPU model name
    
    CoreInfo() 
        : id(0), socket(0), physical_id(0), sibling(-1), type(CoreType::UNKNOWN),
          l1d_cache_size_kb(0), l1i_cache_size_kb(0), l2_cache_size_kb(0), 
          l3_cache_size_kb(0), max_frequency_mhz(0.0) {}
};

/**
 * @brief Class for managing processor cores
 * 
 * This class provides functionality for querying system CPU information,
 * setting thread affinity, and managing core allocation.
 */
class ProcessorCoreManager {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the manager
     */
    static ProcessorCoreManager& getInstance();
    
    /**
     * @brief Initialize the manager
     * 
     * Detects system cores and their properties
     * 
     * @return true if initialization succeeded
     */
    bool initialize();
    
    /**
     * @brief Get information about all available cores
     * 
     * @return Vector of core information
     */
    const std::vector<CoreInfo>& getAllCores() const { return cores_; }
    
    /**
     * @brief Get information about a specific core
     * 
     * @param core_id Core ID
     * @return Core information (nullptr if not found)
     */
    const CoreInfo* getCoreInfo(int core_id) const;
    
    /**
     * @brief Pin the current thread to a specific core
     * 
     * @param core_id Core ID
     * @return true if successful
     */
    bool pinCurrentThread(int core_id);
    
    /**
     * @brief Pin a specific thread to a core
     * 
     * @param thread Thread to pin
     * @param core_id Core ID
     * @return true if successful
     */
    bool pinThread(std::thread& thread, int core_id);
    
    /**
     * @brief Pin a specific thread to multiple cores
     * 
     * @param thread Thread to pin
     * @param core_ids Vector of core IDs
     * @return true if successful
     */
    bool pinThreadToMultipleCores(std::thread& thread, const std::vector<int>& core_ids);
    
    /**
     * @brief Get IDs of all performance cores
     * 
     * @return Vector of core IDs
     */
    std::vector<int> getPerformanceCoreIds() const;
    
    /**
     * @brief Get IDs of all efficiency cores
     * 
     * @return Vector of core IDs
     */
    std::vector<int> getEfficiencyCoreIds() const;
    
    /**
     * @brief Get IDs of cores sharing the same L3 cache
     * 
     * @param core_id Reference core ID
     * @return Vector of core IDs sharing L3 cache with the reference core
     */
    std::vector<int> getCoresShareL3Cache(int core_id) const;
    
    /**
     * @brief Get the total number of cores
     * 
     * @return Number of cores
     */
    size_t getCoreCount() const { return cores_.size(); }
    
    /**
     * @brief Get the number of physical cores (excluding hyperthreading siblings)
     * 
     * @return Number of physical cores
     */
    size_t getPhysicalCoreCount() const;
    
    /**
     * @brief Reserve a core for exclusive use
     * 
     * @param purpose Purpose of reservation (for tracking)
     * @param prefer_performance true to prefer performance cores
     * @return Core ID if successful, -1 if no cores available
     */
    int reserveCore(const std::string& purpose, bool prefer_performance = true);
    
    /**
     * @brief Reserve multiple cores for exclusive use
     * 
     * @param count Number of cores to reserve
     * @param purpose Purpose of reservation
     * @param prefer_performance true to prefer performance cores
     * @param consecutive true to try to get consecutive core IDs
     * @return Vector of reserved core IDs
     */
    std::vector<int> reserveCores(size_t count, const std::string& purpose, 
                                 bool prefer_performance = true,
                                 bool consecutive = false);
    
    /**
     * @brief Release a previously reserved core
     * 
     * @param core_id Core ID to release
     * @return true if successful
     */
    bool releaseCore(int core_id);
    
    /**
     * @brief Release all cores reserved for a specific purpose
     * 
     * @param purpose Purpose string
     * @return Number of cores released
     */
    size_t releaseAllCores(const std::string& purpose);
    
    /**
     * @brief Get the current core allocations
     * 
     * @return Map of core IDs to purpose strings
     */
    std::unordered_map<int, std::string> getAllocations() const;

private:
    // Private constructor for singleton
    ProcessorCoreManager() = default;
    
    // Private destructor
    ~ProcessorCoreManager() = default;
    
    // Disallow copying
    ProcessorCoreManager(const ProcessorCoreManager&) = delete;
    ProcessorCoreManager& operator=(const ProcessorCoreManager&) = delete;
    
    // Core information
    std::vector<CoreInfo> cores_;
    
    // Core allocations
    std::unordered_map<int, std::string> allocations_;
    
    // Socket to cores mapping
    std::unordered_map<int, std::vector<int>> socket_to_cores_;
    
    // Mutex for thread safety
    mutable std::mutex mutex_;
    
    // Platform-specific implementation of thread pinning
    bool pinThreadImpl(std::thread::native_handle_type thread, int core_id);
    bool pinThreadToMultipleCoresImpl(std::thread::native_handle_type thread, 
                                    const std::vector<int>& core_ids);
    
    // Platform-specific detection of core properties
    bool detectCoresImpl();
    
    // Helper methods
    void organizeBySocket();
    void detectCoreTypes();
    void printCoreInfo() const;
};

/**
 * @brief RAII class for temporary core reservation
 */
class ScopedCoreReservation {
public:
    /**
     * @brief Constructor - reserves a core
     * 
     * @param purpose Purpose of reservation
     * @param prefer_performance true to prefer performance cores
     */
    explicit ScopedCoreReservation(const std::string& purpose, 
                                   bool prefer_performance = true);
    
    /**
     * @brief Destructor - releases the core
     */
    ~ScopedCoreReservation();
    
    /**
     * @brief Get the reserved core ID
     * 
     * @return Core ID (-1 if reservation failed)
     */
    int getCoreId() const { return core_id_; }
    
    /**
     * @brief Pin the current thread to the reserved core
     * 
     * @return true if successful
     */
    bool pinCurrentThread();
    
    /**
     * @brief Pin a specific thread to the reserved core
     * 
     * @param thread Thread to pin
     * @return true if successful
     */
    bool pinThread(std::thread& thread);
    
private:
    int core_id_;
    std::string purpose_;
};

/**
 * @brief RAII class for temporary multi-core reservation
 */
class ScopedMultiCoreReservation {
public:
    /**
     * @brief Constructor - reserves multiple cores
     * 
     * @param count Number of cores to reserve
     * @param purpose Purpose of reservation
     * @param prefer_performance true to prefer performance cores
     * @param consecutive true to try to get consecutive core IDs
     */
    ScopedMultiCoreReservation(size_t count, const std::string& purpose,
                              bool prefer_performance = true,
                              bool consecutive = false);
    
    /**
     * @brief Destructor - releases all cores
     */
    ~ScopedMultiCoreReservation();
    
    /**
     * @brief Get the reserved core IDs
     * 
     * @return Vector of core IDs
     */
    const std::vector<int>& getCoreIds() const { return core_ids_; }
    
    /**
     * @brief Pin the current thread to one of the reserved cores
     * 
     * @param index Index into the reserved cores vector
     * @return true if successful
     */
    bool pinCurrentThread(size_t index = 0);
    
    /**
     * @brief Pin a specific thread to one of the reserved cores
     * 
     * @param thread Thread to pin
     * @param index Index into the reserved cores vector
     * @return true if successful
     */
    bool pinThread(std::thread& thread, size_t index = 0);
    
    /**
     * @brief Pin a specific thread to all reserved cores
     * 
     * @param thread Thread to pin
     * @return true if successful
     */
    bool pinThreadToAllCores(std::thread& thread);
    
private:
    std::vector<int> core_ids_;
    std::string purpose_;
};

} // namespace system
} // namespace engine

#endif // ENGINE_SYSTEM_PROCORE_H