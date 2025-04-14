#ifndef ENGINE_SYSTEM_LATMON_H
#define ENGINE_SYSTEM_LATMON_H

#include <chrono>
#include <string>
#include <vector>
#include <deque>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <memory>
#include <functional>

namespace engine {
namespace system {

/**
 * @brief Statistics for latency measurements
 */
struct LatencyStats {
    uint64_t count;         // Number of samples
    double min_us;          // Minimum latency in microseconds
    double max_us;          // Maximum latency in microseconds
    double avg_us;          // Average latency in microseconds
    double median_us;       // Median latency in microseconds
    double p95_us;          // 95th percentile latency in microseconds
    double p99_us;          // 99th percentile latency in microseconds
    double stddev_us;       // Standard deviation in microseconds
    double sum_us;          // Sum of all latencies in microseconds

    LatencyStats() 
        : count(0), min_us(0.0), max_us(0.0), avg_us(0.0), 
          median_us(0.0), p95_us(0.0), p99_us(0.0), stddev_us(0.0), sum_us(0.0) {}
};

/**
 * @brief Class for tracking latency measurements
 * 
 * This class tracks latency measurements for various operations in the system,
 * computing statistics like mean, median, percentiles, etc.
 */
class LatencyMonitor {
public:
    /**
     * @brief Constructor
     * 
     * @param name Name of this latency monitor
     * @param window_size Number of samples to keep for percentile calculations
     */
    explicit LatencyMonitor(const std::string& name, size_t window_size = 1000);
    
    /**
     * @brief Record a latency measurement
     * 
     * @param latency_us Latency in microseconds
     */
    void recordLatency(double latency_us);
    
    /**
     * @brief Start a latency measurement
     * 
     * @return Token to be used with endMeasurement
     */
    uint64_t startMeasurement();
    
    /**
     * @brief End a latency measurement
     * 
     * @param token Token from startMeasurement
     * @return Measured latency in microseconds
     */
    double endMeasurement(uint64_t token);
    
    /**
     * @brief Get the name of this monitor
     * 
     * @return Monitor name
     */
    const std::string& getName() const { return name_; }
    
    /**
     * @brief Get current latency statistics
     * 
     * @return Latency statistics
     */
    LatencyStats getStats() const;
    
    /**
     * @brief Reset all statistics
     */
    void reset();
    
    /**
     * @brief Get the current window size
     * 
     * @return Window size
     */
    size_t getWindowSize() const { return window_size_; }
    
    /**
     * @brief Set the window size
     * 
     * @param size New window size
     */
    void setWindowSize(size_t size);
    
private:
    std::string name_;
    size_t window_size_;
    std::deque<double> recent_samples_;
    
    // Running statistics
    std::atomic<uint64_t> count_{0};
    std::atomic<double> min_us_{std::numeric_limits<double>::max()};
    std::atomic<double> max_us_{0.0};
    std::atomic<double> sum_us_{0.0};
    std::atomic<double> sum_squared_us_{0.0};
    
    // Mutex for sample window
    mutable std::mutex mutex_;
    
    // Calculate percentile from samples
    double calculatePercentile(double percentile) const;
    
    // High-resolution clock for measurements
    using clock_type = std::chrono::high_resolution_clock;
    using time_point = clock_type::time_point;
    
    // Map to store start times
    mutable std::mutex start_times_mutex_;
    std::unordered_map<uint64_t, time_point> start_times_;
    std::atomic<uint64_t> next_token_{1};
};

/**
 * @brief Manager for multiple latency monitors
 */
class LatencyMonitorManager {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the manager
     */
    static LatencyMonitorManager& getInstance();
    
    /**
     * @brief Create a new latency monitor
     * 
     * @param name Monitor name
     * @param window_size Window size for percentile calculations
     * @return Shared pointer to the monitor
     */
    std::shared_ptr<LatencyMonitor> createMonitor(const std::string& name, size_t window_size = 1000);
    
    /**
     * @brief Get an existing monitor by name
     * 
     * @param name Monitor name
     * @return Shared pointer to the monitor (nullptr if not found)
     */
    std::shared_ptr<LatencyMonitor> getMonitor(const std::string& name);
    
    /**
     * @brief Get statistics for all monitors
     * 
     * @return Map of monitor names to statistics
     */
    std::unordered_map<std::string, LatencyStats> getAllStats() const;
    
    /**
     * @brief Reset all monitors
     */
    void resetAll();
    
private:
    // Private constructor for singleton
    LatencyMonitorManager() = default;
    
    // Private destructor
    ~LatencyMonitorManager() = default;
    
    // Disallow copying
    LatencyMonitorManager(const LatencyMonitorManager&) = delete;
    LatencyMonitorManager& operator=(const LatencyMonitorManager&) = delete;
    
    // Monitor storage
    std::unordered_map<std::string, std::shared_ptr<LatencyMonitor>> monitors_;
    mutable std::mutex mutex_;
};

/**
 * @brief RAII class for measuring latency
 */
class ScopedLatencyMeasurement {
public:
    /**
     * @brief Constructor - starts the measurement
     * 
     * @param monitor Latency monitor to use
     */
    explicit ScopedLatencyMeasurement(std::shared_ptr<LatencyMonitor> monitor);
    
    /**
     * @brief Destructor - ends the measurement and records the latency
     */
    ~ScopedLatencyMeasurement();
    
    /**
     * @brief Get the current measurement duration
     * 
     * @return Current duration in microseconds
     */
    double getCurrentDuration() const;
    
private:
    std::shared_ptr<LatencyMonitor> monitor_;
    uint64_t token_;
};

/**
 * @brief Macro for convenient latency measurement
 * 
 * Usage: MEASURE_LATENCY("operation_name");
 */
#define MEASURE_LATENCY(name) \
    auto CONCAT_IMPL(latency_monitor_, __LINE__) = \
        engine::system::LatencyMonitorManager::getInstance().getMonitor(name); \
    if (!CONCAT_IMPL(latency_monitor_, __LINE__)) { \
        CONCAT_IMPL(latency_monitor_, __LINE__) = \
            engine::system::LatencyMonitorManager::getInstance().createMonitor(name); \
    } \
    engine::system::ScopedLatencyMeasurement CONCAT_IMPL(latency_measurement_, __LINE__)( \
        CONCAT_IMPL(latency_monitor_, __LINE__))

/**
 * @brief Helper macro to concatenate tokens
 */
#define CONCAT_IMPL(a, b) a ## b

} // namespace system
} // namespace engine