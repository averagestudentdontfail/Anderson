#ifndef PERF_MON_H
#define PERF_MON_H

#include "timemon.h"
#include <atomic>
#include <chrono>
#include <map>
#include <mutex>
#include <string>
#include <memory>
#include <vector>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <limits>
#include <iostream>
#include <thread>
#include <unordered_map>

namespace profile {

/**
 * @brief Counter class for tracking metrics
 * 
 * This class provides thread-safe counters with various
 * operations (increment, decrement, set, etc.)
 */
class Counter {
public:
    /**
     * @brief Constructor
     * 
     * @param name Counter name
     * @param initial_value Initial value (default 0)
     */
    explicit Counter(const std::string& name, int64_t initial_value = 0)
        : name_(name), value_(initial_value) {}
    
    /**
     * @brief Get the counter name
     * 
     * @return Counter name
     */
    const std::string& name() const {
        return name_;
    }
    
    /**
     * @brief Get the current value
     * 
     * @return Current counter value
     */
    int64_t value() const {
        return value_.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Increment the counter
     * 
     * @param amount Amount to increment (default 1)
     * @return New counter value
     */
    int64_t increment(int64_t amount = 1) {
        return value_.fetch_add(amount, std::memory_order_relaxed) + amount;
    }
    
    /**
     * @brief Decrement the counter
     * 
     * @param amount Amount to decrement (default 1)
     * @return New counter value
     */
    int64_t decrement(int64_t amount = 1) {
        return value_.fetch_sub(amount, std::memory_order_relaxed) - amount;
    }
    
    /**
     * @brief Set the counter to a specific value
     * 
     * @param value New value
     */
    void set(int64_t value) {
        value_.store(value, std::memory_order_relaxed);
    }
    
    /**
     * @brief Reset the counter to zero
     */
    void reset() {
        value_.store(0, std::memory_order_relaxed);
    }
    
private:
    std::string name_;
    std::atomic<int64_t> value_;
};

/**
 * @brief Rate counter for tracking rates (e.g., requests/second)
 * 
 * This class calculates rates over time windows.
 */
class RateCounter {
public:
    /**
     * @brief Constructor
     * 
     * @param name Counter name
     * @param window_size_ms Size of the rate calculation window in milliseconds
     */
    RateCounter(const std::string& name, int64_t window_size_ms = 1000)
        : name_(name), window_size_ms_(window_size_ms), count_(0),
          rate_(0.0), last_update_time_(0) {}
    
    /**
     * @brief Get the counter name
     * 
     * @return Counter name
     */
    const std::string& name() const {
        return name_;
    }
    
    /**
     * @brief Get the current rate
     * 
     * @return Current rate (events per second)
     */
    double rate() {
        update_rate();
        return rate_.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Increment the counter
     * 
     * @param amount Amount to increment (default 1)
     */
    void increment(int64_t amount = 1) {
        count_.fetch_add(amount, std::memory_order_relaxed);
        update_rate();
    }
    
    /**
     * @brief Reset the counter
     */
    void reset() {
        count_.store(0, std::memory_order_relaxed);
        rate_.store(0.0, std::memory_order_relaxed);
        last_update_time_.store(0, std::memory_order_relaxed);
    }
    
private:
    /**
     * @brief Update the rate calculation
     */
    void update_rate() {
        int64_t now = HighResolutionTimer::now_ms();
        int64_t last_time = last_update_time_.load(std::memory_order_relaxed);
        
        if (last_time == 0) {
            // First update
            last_update_time_.store(now, std::memory_order_relaxed);
            return;
        }
        
        int64_t elapsed_ms = now - last_time;
        if (elapsed_ms >= window_size_ms_) {
            // Time to update the rate
            int64_t current_count = count_.load(std::memory_order_relaxed);
            double new_rate = static_cast<double>(current_count) * 1000.0 / elapsed_ms;
            
            // Reset the counter
            count_.store(0, std::memory_order_relaxed);
            
            // Update the rate and last update time
            rate_.store(new_rate, std::memory_order_relaxed);
            last_update_time_.store(now, std::memory_order_relaxed);
        }
    }
    
    std::string name_;
    int64_t window_size_ms_;
    std::atomic<int64_t> count_;
    std::atomic<double> rate_;
    std::atomic<int64_t> last_update_time_;
};

/**
 * @brief Gauge for tracking metrics that can arbitrarily go up and down
 * 
 * This class is useful for metrics like memory usage, queue sizes, etc.
 */
class Gauge {
public:
    /**
     * @brief Constructor
     * 
     * @param name Gauge name
     * @param initial_value Initial value (default 0)
     */
    explicit Gauge(const std::string& name, double initial_value = 0.0)
        : name_(name), value_(initial_value) {}
    
    /**
     * @brief Get the gauge name
     * 
     * @return Gauge name
     */
    const std::string& name() const {
        return name_;
    }
    
    /**
     * @brief Get the current value
     * 
     * @return Current gauge value
     */
    double value() const {
        return value_.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Set the gauge to a specific value
     * 
     * @param value New value
     */
    void set(double value) {
        value_.store(value, std::memory_order_relaxed);
    }
    
    /**
     * @brief Increment the gauge
     * 
     * @param amount Amount to increment
     */
    void increment(double amount = 1.0) {
        // Use a CAS loop for atomic addition of doubles
        double current = value_.load(std::memory_order_relaxed);
        double desired;
        do {
            desired = current + amount;
        } while (!value_.compare_exchange_weak(
            current, desired, 
            std::memory_order_relaxed, std::memory_order_relaxed));
    }
    
    /**
     * @brief Decrement the gauge
     * 
     * @param amount Amount to decrement
     */
    void decrement(double amount = 1.0) {
        increment(-amount);
    }
    
    /**
     * @brief Reset the gauge to zero
     */
    void reset() {
        value_.store(0.0, std::memory_order_relaxed);
    }
    
private:
    std::string name_;
    std::atomic<double> value_;
};

/**
 * @brief Histogram for collecting distribution statistics
 * 
 * This class tracks the distribution of values, including min, max, mean,
 * percentiles, etc.
 */
class Histogram {
public:
    /**
     * @brief Constructor
     * 
     * @param name Histogram name
     * @param buckets Vector of bucket boundaries
     */
    Histogram(const std::string& name, const std::vector<double>& buckets = {})
        : name_(name), count_(0), sum_(0.0), min_(std::numeric_limits<double>::max()),
          max_(std::numeric_limits<double>::lowest()) {
        // Set up buckets
        if (buckets.empty()) {
            // Default exponential buckets (1, 2, 5, 10, 20, 50, 100, ...)
            for (double v = 1.0; v <= 1000000.0; v *= (v >= 5.0 ? 2.0 : 2.5)) {
                bucket_boundaries_.push_back(v);
            }
        } else {
            bucket_boundaries_ = buckets;
            std::sort(bucket_boundaries_.begin(), bucket_boundaries_.end());
        }
        
        // Initialize buckets
        bucket_counts_.resize(bucket_boundaries_.size() + 1, 0);
    }
    
    /**
     * @brief Get the histogram name
     * 
     * @return Histogram name
     */
    const std::string& name() const {
        return name_;
    }
    
    /**
     * @brief Add a value to the histogram
     * 
     * @param value Value to add
     */
    void observe(double value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Increment the total count
        count_++;
        
        // Add to sum for mean calculation
        sum_ += value;
        
        // Update min and max
        min_ = std::min(min_, value);
        max_ = std::max(max_, value);
        
        // Add recent value
        recent_values_.push_back(value);
        if (recent_values_.size() > 1000) {  // Keep the most recent 1000 values
            recent_values_.erase(recent_values_.begin());
        }
        
        // Find the appropriate bucket
        size_t bucket_index = 0;
        while (bucket_index < bucket_boundaries_.size() && value > bucket_boundaries_[bucket_index]) {
            bucket_index++;
        }
        
        // Increment the bucket count
        bucket_counts_[bucket_index]++;
    }
    
    /**
     * @brief Get the total count of observations
     * 
     * @return Observation count
     */
    int64_t count() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_;
    }
    
    /**
     * @brief Get the sum of all observations
     * 
     * @return Sum of observations
     */
    double sum() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return sum_;
    }
    
    /**
     * @brief Get the mean value
     * 
     * @return Mean value
     */
    double mean() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ > 0 ? sum_ / count_ : 0.0;
    }
    
    /**
     * @brief Get the minimum value
     * 
     * @return Minimum value
     */
    double min() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ > 0 ? min_ : 0.0;
    }
    
    /**
     * @brief Get the maximum value
     * 
     * @return Maximum value
     */
    double max() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ > 0 ? max_ : 0.0;
    }
    
    /**
     * @brief Get the standard deviation
     * 
     * @return Standard deviation
     */
    double stddev() const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (count_ <= 1) {
            return 0.0;
        }
        
        double avg = sum_ / count_;
        double variance = 0.0;
        
        for (double val : recent_values_) {
            double diff = val - avg;
            variance += diff * diff;
        }
        
        variance /= recent_values_.size();
        return std::sqrt(variance);
    }
    
    /**
     * @brief Get a specific percentile value
     * 
     * @param percentile Percentile (0-100)
     * @return Value at the specified percentile
     */
    double percentile(double percentile) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (recent_values_.empty()) {
            return 0.0;
        }
        
        // Make a copy of recent values for sorting
        std::vector<double> values = recent_values_;
        std::sort(values.begin(), values.end());
        
        // Calculate the position
        double pos = percentile / 100.0 * (values.size() - 1);
        size_t idx_lower = static_cast<size_t>(std::floor(pos));
        size_t idx_upper = static_cast<size_t>(std::ceil(pos));
        
        // Interpolate
        if (idx_lower == idx_upper) {
            return values[idx_lower];
        } else {
            double weight_upper = pos - idx_lower;
            double weight_lower = 1.0 - weight_upper;
            return weight_lower * values[idx_lower] + weight_upper * values[idx_upper];
        }
    }
    
    /**
     * @brief Get the bucket boundaries
     * 
     * @return Vector of bucket boundaries
     */
    const std::vector<double>& bucket_boundaries() const {
        return bucket_boundaries_;
    }
    
    /**
     * @brief Get the bucket counts
     * 
     * @return Vector of bucket counts
     */
    std::vector<int64_t> bucket_counts() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return bucket_counts_;
    }
    
    /**
     * @brief Reset all histogram data
     */
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        count_ = 0;
        sum_ = 0.0;
        min_ = std::numeric_limits<double>::max();
        max_ = std::numeric_limits<double>::lowest();
        recent_values_.clear();
        std::fill(bucket_counts_.begin(), bucket_counts_.end(), 0);
    }
    
private:
    std::string name_;
    int64_t count_;
    double sum_;
    double min_;
    double max_;
    std::vector<double> bucket_boundaries_;
    std::vector<int64_t> bucket_counts_;
    std::vector<double> recent_values_;
    mutable std::mutex mutex_;
};

/**
 * @brief Hardware performance counter
 * 
 * This class provides access to hardware performance counters
 * such as CPU cycles, cache misses, branch mispredictions, etc.
 */
class HardwareCounter {
public:
    /**
     * @brief Types of hardware counters
     */
    enum class Type {
        CYCLES,           // CPU cycles
        INSTRUCTIONS,     // Completed instructions
        CACHE_REFERENCES, // Cache references
        CACHE_MISSES,     // Cache misses
        BRANCH_MISSES,    // Branch mispredictions
    };
    
    /**
     * @brief Constructor
     * 
     * @param name Counter name
     * @param type Counter type
     */
    HardwareCounter(const std::string& name, Type type)
        : name_(name), type_(type), count_(0) {}
    
    /**
     * @brief Get the counter name
     * 
     * @return Counter name
     */
    const std::string& name() const {
        return name_;
    }
    
    /**
     * @brief Get the counter type
     * 
     * @return Counter type
     */
    Type type() const {
        return type_;
    }
    
    /**
     * @brief Start the counter
     */
    void start() {
        // Platform-specific code to start the counter
        // Would typically use PAPI, perf_event_open, or similar
        
        // For now, just record the start time as cycles
        start_cycles_ = read_cpu_cycles();
    }
    
    /**
     * @brief Stop the counter and record the result
     */
    void stop() {
        // Platform-specific code to stop the counter
        
        // For now, just calculate elapsed cycles
        uint64_t end_cycles = read_cpu_cycles();
        count_ += (end_cycles - start_cycles_);
    }
    
    /**
     * @brief Get the current count
     * 
     * @return Counter value
     */
    uint64_t count() const {
        return count_;
    }
    
    /**
     * @brief Reset the counter
     */
    void reset() {
        count_ = 0;
        start_cycles_ = 0;
    }
    
private:
    /**
     * @brief Read CPU cycles (platform-specific)
     * 
     * @return Current CPU cycle count
     */
    static uint64_t read_cpu_cycles() {
#if defined(__x86_64__) || defined(_M_X64)
        // x86-64 implementation
        unsigned int low, high;
        __asm__ volatile("rdtsc" : "=a" (low), "=d" (high));
        return ((uint64_t)high << 32) | low;
#else
        // Fallback to system time
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
#endif
    }
    
    std::string name_;
    Type type_;
    uint64_t count_;
    uint64_t start_cycles_;
};

/**
 * @brief Singleton class for performance monitoring
 * 
 * This class manages various performance metrics and provides
 * a centralized interface for recording and querying them.
 */
class PerfMon {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the PerfMon singleton
     */
    static PerfMon& instance() {
        static PerfMon instance;
        return instance;
    }
    
    /**
     * @brief Create a counter
     * 
     * @param name Counter name
     * @param initial_value Initial value (default 0)
     * @return Shared pointer to the counter
     */
    std::shared_ptr<Counter> create_counter(const std::string& name, int64_t initial_value = 0) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto counter = std::make_shared<Counter>(name, initial_value);
        counters_[name] = counter;
        return counter;
    }
    
    /**
     * @brief Get a counter
     * 
     * @param name Counter name
     * @return Shared pointer to the counter (or nullptr if not found)
     */
    std::shared_ptr<Counter> get_counter(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = counters_.find(name);
        if (it != counters_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    /**
     * @brief Create a rate counter
     * 
     * @param name Counter name
     * @param window_size_ms Size of the rate calculation window in milliseconds
     * @return Shared pointer to the rate counter
     */
    std::shared_ptr<RateCounter> create_rate_counter(const std::string& name, int64_t window_size_ms = 1000) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto counter = std::make_shared<RateCounter>(name, window_size_ms);
        rate_counters_[name] = counter;
        return counter;
    }
    
    /**
     * @brief Get a rate counter
     * 
     * @param name Counter name
     * @return Shared pointer to the rate counter (or nullptr if not found)
     */
    std::shared_ptr<RateCounter> get_rate_counter(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = rate_counters_.find(name);
        if (it != rate_counters_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    /**
     * @brief Create a gauge
     * 
     * @param name Gauge name
     * @param initial_value Initial value (default 0)
     * @return Shared pointer to the gauge
     */
    std::shared_ptr<Gauge> create_gauge(const std::string& name, double initial_value = 0.0) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto gauge = std::make_shared<Gauge>(name, initial_value);
        gauges_[name] = gauge;
        return gauge;
    }
    
    /**
     * @brief Get a gauge
     * 
     * @param name Gauge name
     * @return Shared pointer to the gauge (or nullptr if not found)
     */
    std::shared_ptr<Gauge> get_gauge(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = gauges_.find(name);
        if (it != gauges_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    /**
     * @brief Create a histogram
     * 
     * @param name Histogram name
     * @param buckets Vector of bucket boundaries
     * @return Shared pointer to the histogram
     */
    std::shared_ptr<Histogram> create_histogram(const std::string& name, const std::vector<double>& buckets = {}) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto histogram = std::make_shared<Histogram>(name, buckets);
        histograms_[name] = histogram;
        return histogram;
    }
    
    /**
     * @brief Get a histogram
     * 
     * @param name Histogram name
     * @return Shared pointer to the histogram (or nullptr if not found)
     */
    std::shared_ptr<Histogram> get_histogram(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = histograms_.find(name);
        if (it != histograms_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    /**
     * @brief Create a hardware counter
     * 
     * @param name Counter name
     * @param type Counter type
     * @return Shared pointer to the hardware counter
     */
    std::shared_ptr<HardwareCounter> create_hardware_counter(const std::string& name, HardwareCounter::Type type) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto counter = std::make_shared<HardwareCounter>(name, type);
        hardware_counters_[name] = counter;
        return counter;
    }
    
    /**
     * @brief Get a hardware counter
     * 
     * @param name Counter name
     * @return Shared pointer to the hardware counter (or nullptr if not found)
     */
    std::shared_ptr<HardwareCounter> get_hardware_counter(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = hardware_counters_.find(name);
        if (it != hardware_counters_.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    /**
     * @brief Reset all metrics
     */
    void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        for (auto& pair : counters_) {
            pair.second->reset();
        }
        
        for (auto& pair : rate_counters_) {
            pair.second->reset();
        }
        
        for (auto& pair : gauges_) {
            pair.second->reset();
        }
        
        for (auto& pair : histograms_) {
            pair.second->reset();
        }
        
        for (auto& pair : hardware_counters_) {
            pair.second->reset();
        }
    }
    
    /**
     * @brief Print a report of all metrics
     * 
     * @param out Output stream
     */
    void print_report(std::ostream& out = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        out << "======== Performance Report ========\n";
        
        // Print counters
        if (!counters_.empty()) {
            out << "-- Counters --\n";
            for (const auto& pair : counters_) {
                out << pair.first << ": " << pair.second->value() << "\n";
            }
        }
        
        // Print rate counters
        if (!rate_counters_.empty()) {
            out << "-- Rate Counters --\n";
            for (const auto& pair : rate_counters_) {
                out << pair.first << ": " << pair.second->rate() << " /s\n";
            }
        }
        
        // Print gauges
        if (!gauges_.empty()) {
            out << "-- Gauges --\n";
            for (const auto& pair : gauges_) {
                out << pair.first << ": " << pair.second->value() << "\n";
            }
        }
        
        // Print histograms
        if (!histograms_.empty()) {
            out << "-- Histograms --\n";
            for (const auto& pair : histograms_) {
                const auto& hist = pair.second;
                out << pair.first << ":\n";
                out << "  Count: " << hist->count() << "\n";
                out << "  Min: " << hist->min() << "\n";
                out << "  Max: " << hist->max() << "\n";
                out << "  Mean: " << hist->mean() << "\n";
                out << "  StdDev: " << hist->stddev() << "\n";
                out << "  Percentiles: p50=" << hist->percentile(50)
                    << ", p90=" << hist->percentile(90)
                    << ", p99=" << hist->percentile(99) << "\n";
            }
        }
        
        // Print hardware counters
        if (!hardware_counters_.empty()) {
            out << "-- Hardware Counters --\n";
            for (const auto& pair : hardware_counters_) {
                out << pair.first << ": " << pair.second->count() << "\n";
            }
        }
        
        out << "====================================\n";
    }
    
    /**
     * @brief Increment a counter (creating it if necessary)
     * 
     * @param name Counter name
     * @param amount Amount to increment (default 1)
     * @return New counter value
     */
    int64_t increment_counter(const std::string& name, int64_t amount = 1) {
        auto counter = get_counter(name);
        if (!counter) {
            counter = create_counter(name);
        }
        return counter->increment(amount);
    }
    
    /**
     * @brief Record a timing observation in a histogram (creating it if necessary)
     * 
     * @param name Histogram name
     * @param value Value to record
     */
    void record_observation(const std::string& name, double value) {
        auto histogram = get_histogram(name);
        if (!histogram) {
            histogram = create_histogram(name);
        }
        histogram->observe(value);
    }
    
    /**
     * @brief Set a gauge value (creating it if necessary)
     * 
     * @param name Gauge name
     * @param value New value
     */
    void set_gauge(const std::string& name, double value) {
        auto gauge = get_gauge(name);
        if (!gauge) {
            gauge = create_gauge(name);
        }
        gauge->set(value);
    }
    
private:
    // Private constructor for singleton
    PerfMon() = default;
    
    // Private destructor for singleton
    ~PerfMon() = default;
    
    // Prevent copying and moving
    PerfMon(const PerfMon&) = delete;
    PerfMon& operator=(const PerfMon&) = delete;
    PerfMon(PerfMon&&) = delete;
    PerfMon& operator=(PerfMon&&) = delete;
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Counter>> counters_;
    std::unordered_map<std::string, std::shared_ptr<RateCounter>> rate_counters_;
    std::unordered_map<std::string, std::shared_ptr<Gauge>> gauges_;
    std::unordered_map<std::string, std::shared_ptr<Histogram>> histograms_;
    std::unordered_map<std::string, std::shared_ptr<HardwareCounter>> hardware_counters_;
};

/**
 * @brief RAII class for measuring code block performance
 * 
 * This class automatically records execution time and optionally
 * increments a counter when the code block completes.
 */
class ScopedPerformanceTimer {
public:
    /**
     * @brief Constructor - starts the timer
     * 
     * @param histogram_name Name of the histogram to record time in
     * @param counter_name Optional name of a counter to increment
     * @param counter_amount Amount to increment the counter
     */
    ScopedPerformanceTimer(const std::string& histogram_name,
                          const std::string& counter_name = "",
                          int64_t counter_amount = 1)
        : histogram_name_(histogram_name), counter_name_(counter_name),
          counter_amount_(counter_amount), timer_() {}
    
    /**
     * @brief Destructor - records time and increments counter
     */
    ~ScopedPerformanceTimer() {
        // Record time
        double elapsed_ms = static_cast<double>(timer_.elapsed_us()) / 1000.0;
        PerfMon::instance().record_observation(histogram_name_, elapsed_ms);
        
        // Increment counter if specified
        if (!counter_name_.empty()) {
            PerfMon::instance().increment_counter(counter_name_, counter_amount_);
        }
    }
    
private:
    std::string histogram_name_;
    std::string counter_name_;
    int64_t counter_amount_;
    HighResolutionTimer timer_;
};

/**
 * @brief Macro for convenient creation of performance timers
 * 
 * Usage: PERF_BLOCK("operation_name");
 */
#define PERF_BLOCK(name) \
    profile::ScopedPerformanceTimer CONCAT_IMPL(perf_timer_, __LINE__)(name)

/**
 * @brief Macro for convenient creation of performance timers with counter
 * 
 * Usage: PERF_BLOCK_COUNT("operation_name", "counter_name");
 */
#define PERF_BLOCK_COUNT(name, counter) \
    profile::ScopedPerformanceTimer CONCAT_IMPL(perf_timer_, __LINE__)(name, counter)

} // namespace profile

#endif // PERF_MON_H