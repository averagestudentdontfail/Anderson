#include "latmon.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>

namespace engine {
namespace system {

// -----------------------------------------------------------------------------
// LatencyMonitor Implementation
// -----------------------------------------------------------------------------

LatencyMonitor::LatencyMonitor(const std::string& name, size_t window_size)
    : name_(name), window_size_(window_size) {
    // Reserve space for samples
    std::lock_guard<std::mutex> lock(mutex_);
    recent_samples_.reserve(window_size);
}

void LatencyMonitor::recordLatency(double latency_us) {
    // Update running statistics
    count_.fetch_add(1, std::memory_order_relaxed);
    sum_us_.fetch_add(latency_us, std::memory_order_relaxed);
    sum_squared_us_.fetch_add(latency_us * latency_us, std::memory_order_relaxed);
    
    // Update min/max atomically
    double current_min = min_us_.load(std::memory_order_relaxed);
    while (latency_us < current_min) {
        if (min_us_.compare_exchange_weak(current_min, latency_us, 
                                        std::memory_order_relaxed, 
                                        std::memory_order_relaxed)) {
            break;
        }
    }
    
    double current_max = max_us_.load(std::memory_order_relaxed);
    while (latency_us > current_max) {
        if (max_us_.compare_exchange_weak(current_max, latency_us, 
                                        std::memory_order_relaxed, 
                                        std::memory_order_relaxed)) {
            break;
        }
    }
    
    // Add to recent samples
    {
        std::lock_guard<std::mutex> lock(mutex_);
        recent_samples_.push_back(latency_us);
        
        // Keep only window_size_ most recent samples
        while (recent_samples_.size() > window_size_) {
            recent_samples_.pop_front();
        }
    }
}

uint64_t LatencyMonitor::startMeasurement() {
    uint64_t token = next_token_.fetch_add(1, std::memory_order_relaxed);
    
    // Store start time
    {
        std::lock_guard<std::mutex> lock(start_times_mutex_);
        start_times_[token] = clock_type::now();
    }
    
    return token;
}

double LatencyMonitor::endMeasurement(uint64_t token) {
    time_point start_time;
    
    // Get start time
    {
        std::lock_guard<std::mutex> lock(start_times_mutex_);
        auto it = start_times_.find(token);
        if (it == start_times_.end()) {
            return -1.0;  // Invalid token
        }
        
        start_time = it->second;
        start_times_.erase(it);
    }
    
    // Calculate latency
    auto end_time = clock_type::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        end_time - start_time).count();
    
    // Record the latency
    double latency_us = static_cast<double>(duration);
    recordLatency(latency_us);
    
    return latency_us;
}

LatencyStats LatencyMonitor::getStats() const {
    LatencyStats stats;
    
    // Get count
    uint64_t count = count_.load(std::memory_order_relaxed);
    if (count == 0) {
        return stats;  // Return zeros for empty stats
    }
    
    stats.count = count;
    stats.min_us = min_us_.load(std::memory_order_relaxed);
    stats.max_us = max_us_.load(std::memory_order_relaxed);
    
    // Calculate mean
    double sum = sum_us_.load(std::memory_order_relaxed);
    stats.sum_us = sum;
    stats.avg_us = sum / count;
    
    // Calculate standard deviation
    double sum_squared = sum_squared_us_.load(std::memory_order_relaxed);
    double variance = (sum_squared / count) - (stats.avg_us * stats.avg_us);
    stats.stddev_us = std::sqrt(variance);
    
    // Calculate percentiles
    stats.median_us = calculatePercentile(50.0);
    stats.p95_us = calculatePercentile(95.0);
    stats.p99_us = calculatePercentile(99.0);
    
    return stats;
}

void LatencyMonitor::reset() {
    // Reset running statistics
    count_.store(0, std::memory_order_relaxed);
    min_us_.store(std::numeric_limits<double>::max(), std::memory_order_relaxed);
    max_us_.store(0.0, std::memory_order_relaxed);
    sum_us_.store(0.0, std::memory_order_relaxed);
    sum_squared_us_.store(0.0, std::memory_order_relaxed);
    
    // Clear recent samples
    {
        std::lock_guard<std::mutex> lock(mutex_);
        recent_samples_.clear();
    }
}

void LatencyMonitor::setWindowSize(size_t size) {
    window_size_ = size;
    
    // Adjust recent samples
    std::lock_guard<std::mutex> lock(mutex_);
    while (recent_samples_.size() > window_size_) {
        recent_samples_.pop_front();
    }
}

double LatencyMonitor::calculatePercentile(double percentile) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (recent_samples_.empty()) {
        return 0.0;
    }
    
    // Copy samples for sorting
    std::vector<double> samples(recent_samples_.begin(), recent_samples_.end());
    std::sort(samples.begin(), samples.end());
    
    // Calculate index
    double idx = (percentile / 100.0) * (samples.size() - 1);
    size_t idx_lower = static_cast<size_t>(std::floor(idx));
    size_t idx_upper = static_cast<size_t>(std::ceil(idx));
    
    // Linear interpolation
    if (idx_lower == idx_upper) {
        return samples[idx_lower];
    } else {
        double weight_upper = idx - idx_lower;
        double weight_lower = 1.0 - weight_upper;
        return weight_lower * samples[idx_lower] + weight_upper * samples[idx_upper];
    }
}

// -----------------------------------------------------------------------------
// LatencyMonitorManager Implementation
// -----------------------------------------------------------------------------

LatencyMonitorManager& LatencyMonitorManager::getInstance() {
    static LatencyMonitorManager instance;
    return instance;
}

std::shared_ptr<LatencyMonitor> LatencyMonitorManager::createMonitor(
    const std::string& name, size_t window_size) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Create the monitor
    auto monitor = std::make_shared<LatencyMonitor>(name, window_size);
    
    // Store it
    monitors_[name] = monitor;
    
    return monitor;
}

std::shared_ptr<LatencyMonitor> LatencyMonitorManager::getMonitor(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = monitors_.find(name);
    if (it != monitors_.end()) {
        return it->second;
    }
    
    return nullptr;
}

std::unordered_map<std::string, LatencyStats> LatencyMonitorManager::getAllStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::unordered_map<std::string, LatencyStats> all_stats;
    for (const auto& pair : monitors_) {
        all_stats[pair.first] = pair.second->getStats();
    }
    
    return all_stats;
}

void LatencyMonitorManager::resetAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : monitors_) {
        pair.second->reset();
    }
}

// -----------------------------------------------------------------------------
// ScopedLatencyMeasurement Implementation
// -----------------------------------------------------------------------------

ScopedLatencyMeasurement::ScopedLatencyMeasurement(std::shared_ptr<LatencyMonitor> monitor)
    : monitor_(monitor) {
    
    // Start the measurement
    if (monitor_) {
        token_ = monitor_->startMeasurement();
    }
}

ScopedLatencyMeasurement::~ScopedLatencyMeasurement() {
    // End the measurement
    if (monitor_) {
        monitor_->endMeasurement(token_);
    }
}

double ScopedLatencyMeasurement::getCurrentDuration() const {
    if (!monitor_) {
        return -1.0;
    }
    
    // Get start time
    LatencyMonitor::time_point start_time;
    bool found = false;
    
    {
        std::lock_guard<std::mutex> lock(monitor_->start_times_mutex_);
        auto it = monitor_->start_times_.find(token_);
        if (it != monitor_->start_times_.end()) {
            start_time = it->second;
            found = true;
        }
    }
    
    if (!found) {
        return -1.0;
    }
    
    // Calculate current duration
    auto now = LatencyMonitor::clock_type::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        now - start_time).count();
    
    return static_cast<double>(duration);
}

} // namespace system
} // namespace engine