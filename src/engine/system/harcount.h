#ifndef ENGINE_SYSTEM_HARCOUNT_H
#define ENGINE_SYSTEM_HARCOUNT_H

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>
#include <mutex>
#include <memory>

#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <unistd.h>
#endif

namespace engine {
namespace system {

/**
 * @brief Type of hardware counter to monitor
 */
enum class CounterType {
    CYCLES,             // CPU cycles
    INSTRUCTIONS,       // Retired instructions
    CACHE_REFERENCES,   // Cache references
    CACHE_MISSES,       // Cache misses
    BRANCH_INSTRUCTIONS, // Branch instructions
    BRANCH_MISSES,      // Branch mispredictions
    BUS_CYCLES,         // Bus cycles
    L1D_LOADS,          // L1 data cache loads
    L1D_LOAD_MISSES,    // L1 data cache load misses
    L1D_STORES,         // L1 data cache stores
    L1D_STORE_MISSES,   // L1 data cache store misses
    L1I_LOAD_MISSES,    // L1 instruction cache load misses
    LLC_LOADS,          // Last level cache loads
    LLC_LOAD_MISSES,    // Last level cache load misses
    LLC_STORES,         // Last level cache stores
    LLC_STORE_MISSES,   // Last level cache store misses
    DTLB_LOAD_MISSES,   // Data TLB load misses
    ITLB_LOAD_MISSES,   // Instruction TLB load misses
    PAGE_FAULTS,        // Page faults
    CONTEXT_SWITCHES,   // Context switches
    CPU_MIGRATIONS,     // CPU migrations
    ALIGNMENT_FAULTS,   // Alignment faults
    EMULATION_FAULTS,   // Emulation faults
    // Add more as needed
};

/**
 * @brief Result from reading hardware counters
 */
struct CounterResult {
    int64_t value;      // Counter value
    int64_t time_enabled;  // Time counter was enabled (if available)
    int64_t time_running;  // Time counter was running (if available)
    double scaled_value; // Scaled value accounting for multiplexing (if needed)
};

/**
 * @brief Class for accessing hardware performance counters
 * 
 * This class provides a platform-independent interface to access
 * hardware performance counters on the CPU.
 */
class HardwareCounter {
public:
    /**
     * @brief Construct a new Hardware Counter object
     * 
     * @param type Type of counter to monitor
     * @param name Name for this counter instance
     */
    HardwareCounter(CounterType type, const std::string& name = "");
    
    /**
     * @brief Destructor
     */
    ~HardwareCounter();
    
    /**
     * @brief Start the counter
     * 
     * @param pid Process ID to monitor (0 for current process)
     * @param cpu CPU to monitor (-1 for any CPU)
     * @return true if counter was started successfully
     */
    bool start(pid_t pid = 0, int cpu = -1);
    
    /**
     * @brief Stop the counter
     * 
     * @return true if counter was stopped successfully
     */
    bool stop();
    
    /**
     * @brief Reset the counter
     * 
     * @return true if counter was reset successfully
     */
    bool reset();
    
    /**
     * @brief Read the current counter value
     * 
     * @return Counter result
     */
    CounterResult read() const;
    
    /**
     * @brief Get the name of this counter instance
     * 
     * @return Counter name
     */
    const std::string& getName() const { return name_; }
    
    /**
     * @brief Get the type of this counter
     * 
     * @return Counter type
     */
    CounterType getType() const { return type_; }
    
    /**
     * @brief Check if hardware counters are available on this system
     * 
     * @return true if hardware counters are available
     */
    static bool isAvailable();
    
    /**
     * @brief Get a human-readable name for a counter type
     * 
     * @param type Counter type
     * @return Human-readable name
     */
    static std::string getCounterName(CounterType type);

private:
    CounterType type_;
    std::string name_;
    bool running_;
    int64_t start_value_;
    int file_descriptor_;
    
#ifdef __linux__
    // Linux-specific perf event configuration
    struct perf_event_attr attr_;
#endif

    // Platform-specific counter setup
    bool setupCounter(pid_t pid, int cpu);
    
    // Get the platform-specific counter ID for a counter type
    static uint64_t getCounterConfig(CounterType type);
};

/**
 * @brief Manager for multiple hardware counters
 */
class HardwareCounterManager {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the manager
     */
    static HardwareCounterManager& getInstance();
    
    /**
     * @brief Create a new counter
     * 
     * @param type Counter type
     * @param name Counter name (optional)
     * @return Shared pointer to the counter
     */
    std::shared_ptr<HardwareCounter> createCounter(CounterType type, const std::string& name = "");
    
    /**
     * @brief Get an existing counter by name
     * 
     * @param name Counter name
     * @return Shared pointer to the counter (nullptr if not found)
     */
    std::shared_ptr<HardwareCounter> getCounter(const std::string& name);
    
    /**
     * @brief Start all counters
     * 
     * @param pid Process ID to monitor (0 for current process)
     * @param cpu CPU to monitor (-1 for any CPU)
     */
    void startAll(pid_t pid = 0, int cpu = -1);
    
    /**
     * @brief Stop all counters
     */
    void stopAll();
    
    /**
     * @brief Reset all counters
     */
    void resetAll();
    
    /**
     * @brief Read all counters
     * 
     * @return Map of counter names to results
     */
    std::map<std::string, CounterResult> readAll() const;
    
    /**
     * @brief Check if hardware counters are available
     * 
     * @return true if hardware counters are available
     */
    bool isAvailable() const { return HardwareCounter::isAvailable(); }

private:
    // Private constructor for singleton
    HardwareCounterManager() = default;
    
    // Private destructor
    ~HardwareCounterManager() = default;
    
    // Disallow copying
    HardwareCounterManager(const HardwareCounterManager&) = delete;
    HardwareCounterManager& operator=(const HardwareCounterManager&) = delete;
    
    // Counter storage
    std::unordered_map<std::string, std::shared_ptr<HardwareCounter>> counters_;
    mutable std::mutex mutex_;
};

/**
 * @brief RAII class for measuring with a hardware counter
 */
class ScopedHardwareCounter {
public:
    /**
     * @brief Constructor - starts the counter
     * 
     * @param counter Hardware counter to use
     */
    explicit ScopedHardwareCounter(std::shared_ptr<HardwareCounter> counter);
    
    /**
     * @brief Destructor - stops the counter and records the result
     */
    ~ScopedHardwareCounter();
    
    /**
     * @brief Get the latest result
     * 
     * @return Counter result
     */
    CounterResult getResult() const;
    
private:
    std::shared_ptr<HardwareCounter> counter_;
    CounterResult result_;
};

} // namespace system
} // namespace engine

#endif // ENGINE_SYSTEM_HARCOUNT_H