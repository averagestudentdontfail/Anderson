#ifndef TIME_MON_H
#define TIME_MON_H

#include <chrono>
#include <string>
#include <functional>
#include <map>
#include <mutex>
#include <vector>
#include <utility>
#include <iostream>
#include <atomic>
#include <memory>

namespace profile {

/**
 * @brief High-resolution timer class
 * 
 * This class provides high-resolution timing capabilities 
 * for performance measurements.
 */
class HighResolutionTimer {
public:
    /**
     * @brief Constructor - starts the timer
     */
    HighResolutionTimer() {
        restart();
    }
    
    /**
     * @brief Restart the timer
     */
    void restart() {
        start_time_ = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief Get elapsed time in nanoseconds
     * 
     * @return Elapsed nanoseconds
     */
    int64_t elapsed_ns() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_).count();
    }
    
    /**
     * @brief Get elapsed time in microseconds
     * 
     * @return Elapsed microseconds
     */
    int64_t elapsed_us() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_).count();
    }
    
    /**
     * @brief Get elapsed time in milliseconds
     * 
     * @return Elapsed milliseconds
     */
    int64_t elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_).count();
    }
    
    /**
     * @brief Get elapsed time in seconds (double precision)
     * 
     * @return Elapsed seconds
     */
    double elapsed_s() const {
        return elapsed_ns() / 1e9;
    }
    
    /**
     * @brief Check if a specific duration has elapsed
     * 
     * @tparam Duration Duration type
     * @param duration The duration to check against
     * @return true if the duration has elapsed, false otherwise
     */
    template <typename Duration>
    bool has_elapsed(Duration duration) const {
        auto now = std::chrono::high_resolution_clock::now();
        return (now - start_time_) >= duration;
    }
    
    /**
     * @brief Get current timestamp in nanoseconds since epoch
     * 
     * @return Timestamp in nanoseconds
     */
    static int64_t now_ns() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
    
    /**
     * @brief Get current timestamp in microseconds since epoch
     * 
     * @return Timestamp in microseconds
     */
    static int64_t now_us() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count();
    }
    
    /**
     * @brief Get current timestamp in milliseconds since epoch
     * 
     * @return Timestamp in milliseconds
     */
    static int64_t now_ms() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
    }
    
    /**
     * @brief Convert nanoseconds to microseconds
     * 
     * @param ns Nanoseconds
     * @return Microseconds
     */
    static inline int64_t ns_to_us(int64_t ns) {
        return ns / 1000;
    }
    
    /**
     * @brief Convert nanoseconds to milliseconds
     * 
     * @param ns Nanoseconds
     * @return Milliseconds
     */
    static inline int64_t ns_to_ms(int64_t ns) {
        return ns / 1000000;
    }
    
    /**
     * @brief Convert nanoseconds to seconds
     * 
     * @param ns Nanoseconds
     * @return Seconds
     */
    static inline double ns_to_s(int64_t ns) {
        return static_cast<double>(ns) / 1e9;
    }
    
    /**
     * @brief Convert microseconds to nanoseconds
     * 
     * @param us Microseconds
     * @return Nanoseconds
     */
    static inline int64_t us_to_ns(int64_t us) {
        return us * 1000;
    }
    
    /**
     * @brief Convert microseconds to milliseconds
     * 
     * @param us Microseconds
     * @return Milliseconds
     */
    static inline int64_t us_to_ms(int64_t us) {
        return us / 1000;
    }
    
    /**
     * @brief Convert microseconds to seconds
     * 
     * @param us Microseconds
     * @return Seconds
     */
    static inline double us_to_s(int64_t us) {
        return static_cast<double>(us) / 1e6;
    }
    
    /**
     * @brief Convert milliseconds to nanoseconds
     * 
     * @param ms Milliseconds
     * @return Nanoseconds
     */
    static inline int64_t ms_to_ns(int64_t ms) {
        return ms * 1000000;
    }
    
    /**
     * @brief Convert milliseconds to microseconds
     * 
     * @param ms Milliseconds
     * @return Microseconds
     */
    static inline int64_t ms_to_us(int64_t ms) {
        return ms * 1000;
    }
    
    /**
     * @brief Convert milliseconds to seconds
     * 
     * @param ms Milliseconds
     * @return Seconds
     */
    static inline double ms_to_s(int64_t ms) {
        return static_cast<double>(ms) / 1e3;
    }
    
    /**
     * @brief Convert seconds to nanoseconds
     * 
     * @param s Seconds
     * @return Nanoseconds
     */
    static inline int64_t s_to_ns(double s) {
        return static_cast<int64_t>(s * 1e9);
    }
    
    /**
     * @brief Convert seconds to microseconds
     * 
     * @param s Seconds
     * @return Microseconds
     */
    static inline int64_t s_to_us(double s) {
        return static_cast<int64_t>(s * 1e6);
    }
    
    /**
     * @brief Convert seconds to milliseconds
     * 
     * @param s Seconds
     * @return Milliseconds
     */
    static inline int64_t s_to_ms(double s) {
        return static_cast<int64_t>(s * 1e3);
    }
    
private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

/**
 * @brief Timing statistics class
 * 
 * This class collects and computes statistics for a series
 * of timing measurements.
 */
class TimingStats {
public:
    /**
     * @brief Constructor
     */
    TimingStats() 
        : count_(0), total_ns_(0), min_ns_(std::numeric_limits<int64_t>::max()), 
          max_ns_(0), m2_(0) {}
    
    /**
     * @brief Add a timing sample
     * 
     * @param elapsed_ns Elapsed time in nanoseconds
     */
    void add_sample(int64_t elapsed_ns) {
        count_++;
        total_ns_ += elapsed_ns;
        min_ns_ = std::min(min_ns_, elapsed_ns);
        max_ns_ = std::max(max_ns_, elapsed_ns);
        
        // Online algorithm for variance calculation (Welford's algorithm)
        double delta = elapsed_ns - mean_ns();
        m2_ += delta * delta * (count_ - 1) / count_;
    }
    
    /**
     * @brief Get the number of samples
     * 
     * @return Sample count
     */
    int64_t count() const {
        return count_;
    }
    
    /**
     * @brief Get the total elapsed time
     * 
     * @return Total time in nanoseconds
     */
    int64_t total_ns() const {
        return total_ns_;
    }
    
    /**
     * @brief Get the minimum elapsed time
     * 
     * @return Minimum time in nanoseconds
     */
    int64_t min_ns() const {
        return count_ > 0 ? min_ns_ : 0;
    }
    
    /**
     * @brief Get the maximum elapsed time
     * 
     * @return Maximum time in nanoseconds
     */
    int64_t max_ns() const {
        return max_ns_;
    }
    
    /**
     * @brief Get the mean elapsed time
     * 
     * @return Mean time in nanoseconds
     */
    double mean_ns() const {
        return count_ > 0 ? static_cast<double>(total_ns_) / count_ : 0.0;
    }
    
    /**
     * @brief Get the variance of elapsed times
     * 
     * @return Variance in square nanoseconds
     */
    double variance_ns2() const {
        return count_ > 1 ? m2_ / count_ : 0.0;
    }
    
    /**
     * @brief Get the standard deviation of elapsed times
     * 
     * @return Standard deviation in nanoseconds
     */
    double stddev_ns() const {
        return std::sqrt(variance_ns2());
    }
    
    /**
     * @brief Reset all statistics
     */
    void reset() {
        count_ = 0;
        total_ns_ = 0;
        min_ns_ = std::numeric_limits<int64_t>::max();
        max_ns_ = 0;
        m2_ = 0;
    }
    
    /**
     * @brief Merge another TimingStats into this one
     * 
     * @param other The other TimingStats to merge
     */
    void merge(const TimingStats& other) {
        if (other.count_ == 0) {
            return;
        }
        
        if (count_ == 0) {
            *this = other;
            return;
        }
        
        // Update min/max
        min_ns_ = std::min(min_ns_, other.min_ns_);
        max_ns_ = std::max(max_ns_, other.max_ns_);
        
        // Merge counts and totals
        int64_t new_count = count_ + other.count_;
        
        // Merge variance using parallel algorithm
        double delta = other.mean_ns() - mean_ns();
        m2_ = m2_ + other.m2_ + 
              delta * delta * count_ * other.count_ / new_count;
        
        // Update count and total
        total_ns_ += other.total_ns_;
        count_ = new_count;
    }
    
private:
    int64_t count_;
    int64_t total_ns_;
    int64_t min_ns_;
    int64_t max_ns_;
    double m2_; // Sum of squared differences from the mean
};

/**
 * @brief RAII class for timing code blocks
 * 
 * This class automatically measures the time elapsed between
 * construction and destruction, and records it in the TimeMon system.
 */
class ScopedTimer {
public:
    /**
     * @brief Constructor - starts the timer
     * 
     * @param name Name of the timer
     * @param auto_report Whether to automatically report to TimeMon
     */
    explicit ScopedTimer(const std::string& name, bool auto_report = true)
        : name_(name), auto_report_(auto_report), timer_() {}
    
    /**
     * @brief Destructor - stops the timer and records the elapsed time
     */
    ~ScopedTimer() {
        if (auto_report_) {
            int64_t elapsed = timer_.elapsed_ns();
            TimeMon::instance().record(name_, elapsed);
        }
    }
    
    /**
     * @brief Get the elapsed time
     * 
     * @return Elapsed time in nanoseconds
     */
    int64_t elapsed_ns() const {
        return timer_.elapsed_ns();
    }
    
    /**
     * @brief Restart the timer
     */
    void restart() {
        timer_.restart();
    }
    
    /**
     * @brief Manually record the current elapsed time
     */
    void record() {
        int64_t elapsed = timer_.elapsed_ns();
        TimeMon::instance().record(name_, elapsed);
    }
    
private:
    std::string name_;
    bool auto_report_;
    HighResolutionTimer timer_;
    
    // Forward declaration of TimeMon
    class TimeMon;
};

/**
 * @brief Singleton class for monitoring execution times
 * 
 * This class collects timing statistics for various operations
 * in the system, and provides methods to query and report them.
 */
class TimeMon {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the TimeMon singleton
     */
    static TimeMon& instance() {
        static TimeMon instance;
        return instance;
    }
    
    /**
     * @brief Record a timing measurement
     * 
     * @param name Name of the operation
     * @param elapsed_ns Elapsed time in nanoseconds
     */
    void record(const std::string& name, int64_t elapsed_ns) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_[name].add_sample(elapsed_ns);
    }
    
    /**
     * @brief Get timing statistics for a specific operation
     * 
     * @param name Name of the operation
     * @return Copy of the timing statistics
     */
    TimingStats get_stats(const std::string& name) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_.find(name);
        if (it != stats_.end()) {
            return it->second;
        }
        return TimingStats();
    }
    
    /**
     * @brief Get all timing statistics
     * 
     * @return Map of operation names to timing statistics
     */
    std::map<std::string, TimingStats> get_all_stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return stats_;
    }
    
    /**
     * @brief Reset statistics for a specific operation
     * 
     * @param name Name of the operation
     */
    void reset(const std::string& name) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = stats_.find(name);
        if (it != stats_.end()) {
            it->second.reset();
        }
    }
    
    /**
     * @brief Reset all statistics
     */
    void reset_all() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : stats_) {
            pair.second.reset();
        }
    }
    
    /**
     * @brief Print a report of all timing statistics
     * 
     * @param out Output stream
     */
    void print_report(std::ostream& out = std::cout) const {
        std::lock_guard<std::mutex> lock(mutex_);
        
        out << "======== Timing Report ========\n";
        for (const auto& pair : stats_) {
            const std::string& name = pair.first;
            const TimingStats& stats = pair.second;
            
            out << name << ":\n";
            out << "  Count: " << stats.count() << "\n";
            out << "  Total: " << HighResolutionTimer::ns_to_ms(stats.total_ns()) << " ms\n";
            out << "  Min:   " << stats.min_ns() << " ns\n";
            out << "  Max:   " << stats.max_ns() << " ns\n";
            out << "  Mean:  " << stats.mean_ns() << " ns\n";
            out << "  StdDev: " << stats.stddev_ns() << " ns\n";
        }
        out << "============================\n";
    }
    
    /**
     * @brief Create a scoped timer
     * 
     * @param name Name of the timer
     * @return Unique pointer to a ScopedTimer
     */
    std::unique_ptr<ScopedTimer> create_scoped_timer(const std::string& name) {
        return std::make_unique<ScopedTimer>(name);
    }
    
private:
    // Private constructor for singleton
    TimeMon() = default;
    
    // Private destructor for singleton
    ~TimeMon() = default;
    
    // Prevent copying and moving
    TimeMon(const TimeMon&) = delete;
    TimeMon& operator=(const TimeMon&) = delete;
    TimeMon(TimeMon&&) = delete;
    TimeMon& operator=(TimeMon&&) = delete;
    
    mutable std::mutex mutex_;
    std::map<std::string, TimingStats> stats_;
};

/**
 * @brief Macro for convenient creation of scoped timers
 * 
 * Usage: TIME_BLOCK("operation_name");
 */
#define TIME_BLOCK(name) \
    auto CONCAT_IMPL(timer_, __LINE__) = profile::TimeMon::instance().create_scoped_timer(name)

/**
 * @brief Macro to concatenate tokens
 */
#define CONCAT_IMPL(a, b) a ## b

} // namespace profile

#endif // TIME_MON_H