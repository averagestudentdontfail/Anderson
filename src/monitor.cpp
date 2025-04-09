// monitor.cpp
// Real-time performance monitoring tool for the deterministic pricing system

#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <chrono>
#include <atomic>
#include <numeric>
#include <unistd.h>
#include <signal.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <termios.h>
#include "deterministic_pricing_system.h"

// ANSI color codes
#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"

// Global flag for controlling monitor execution
std::atomic<bool> g_running{true};

// Signal handler for graceful shutdown
void signalHandler(int signal) {
    std::cout << "Received signal " << signal << ", shutting down..." << std::endl;
    g_running = false;
}

// Structure for tracking system metrics
struct SystemMetrics {
    // Queue metrics
    size_t requestQueueSize = 0;
    size_t resultQueueSize = 0;
    size_t marketDataQueueSize = 0;
    
    // Processing metrics
    uint64_t requestsPerSecond = 0;
    uint64_t resultsPerSecond = 0;
    uint64_t marketUpdatesPerSecond = 0;
    
    // Health metrics
    uint64_t lastHeartbeatAge = 0;  // in milliseconds
    bool pricingEngineAlive = true;
    
    // System metrics
    double cpuUsage = 0.0;
    uint64_t memoryUsage = 0;
    
    // Latency metrics
    double avgLatencyMicros = 0.0;
    double p95LatencyMicros = 0.0;
    double p99LatencyMicros = 0.0;
    double maxLatencyMicros = 0.0;
};

// Structure for tracking historical metrics
struct HistoricalMetrics {
    static constexpr size_t HISTORY_SIZE = 120;  // 2 minutes at 1 sample/sec
    
    std::deque<uint64_t> requestsPerSecond;
    std::deque<uint64_t> resultsPerSecond;
    std::deque<double> avgLatencyMicros;
    std::deque<double> maxLatencyMicros;
    
    // Add new metrics
    void addMetrics(const SystemMetrics& current) {
        // Add new values
        requestsPerSecond.push_back(current.requestsPerSecond);
        resultsPerSecond.push_back(current.resultsPerSecond);
        avgLatencyMicros.push_back(current.avgLatencyMicros);
        maxLatencyMicros.push_back(current.maxLatencyMicros);
        
        // Trim to history size
        if (requestsPerSecond.size() > HISTORY_SIZE) requestsPerSecond.pop_front();
        if (resultsPerSecond.size() > HISTORY_SIZE) resultsPerSecond.pop_front();
        if (avgLatencyMicros.size() > HISTORY_SIZE) avgLatencyMicros.pop_front();
        if (maxLatencyMicros.size() > HISTORY_SIZE) maxLatencyMicros.pop_front();
    }
    
    // Get maximum value in history for a metric
    template<typename T>
    T getMaxValue(const std::deque<T>& history) const {
        if (history.empty()) return T();
        return *std::max_element(history.begin(), history.end());
    }
    
    // Get average value in history for a metric
    template<typename T>
    double getAvgValue(const std::deque<T>& history) const {
        if (history.empty()) return 0.0;
        
        T sum = 0;
        for (const auto& value : history) {
            sum += value;
        }
        return static_cast<double>(sum) / history.size();
    }
};

// Class for reading shared memory metrics
class SystemMonitor {
private:
    SharedBlock* shared_ = nullptr;
    int shmFd_ = -1;
    uint64_t lastRequestCount_ = 0;
    uint64_t lastResultCount_ = 0;
    uint64_t lastMarketUpdateCount_ = 0;
    std::chrono::steady_clock::time_point lastUpdateTime_;
    HistoricalMetrics history_;
    
public:
    SystemMonitor() {
        // Initialize last update time
        lastUpdateTime_ = std::chrono::steady_clock::now();
    }
    
    ~SystemMonitor() {
        if (shared_ && shared_ != MAP_FAILED) {
            munmap(shared_, sizeof(SharedBlock));
        }
        
        if (shmFd_ != -1) {
            close(shmFd_);
        }
    }
    
    // Connect to the shared memory segment
    bool connect(const std::string& sharedMemPath = "/tmp/pricing.key") {
        // Check if the file exists
        struct stat st;
        if (stat(sharedMemPath.c_str(), &st) != 0) {
            std::cerr << "Shared memory file does not exist: " << sharedMemPath << std::endl;
            return false;
        }
        
        // Generate IPC key from file path
        key_t key = ftok(sharedMemPath.c_str(), 'R');
        if (key == -1) {
            std::cerr << "Failed to generate IPC key: " << strerror(errno) << std::endl;
            return false;
        }
        
        // Get the shared memory segment
        int shmId = shmget(key, sizeof(SharedBlock), 0666);
        if (shmId == -1) {
            std::cerr << "Failed to get shared memory segment: " << strerror(errno) << std::endl;
            return false;
        }
        
        // Attach to the shared memory segment
        shared_ = static_cast<SharedBlock*>(shmat(shmId, nullptr, 0));
        if (shared_ == (void*)-1) {
            std::cerr << "Failed to attach to shared memory: " << strerror(errno) << std::endl;
            shared_ = nullptr;
            return false;
        }
        
        std::cout << "Successfully connected to shared memory" << std::endl;
        return true;
    }
    
    // Check if connected
    bool isConnected() const {
        return shared_ && shared_ != (void*)-1;
    }
    
    // Update metrics from shared memory
    SystemMetrics getMetrics() {
        SystemMetrics metrics;
        
        if (!isConnected()) {
            return metrics;
        }
        
        // Get current time for rate calculations
        auto now = std::chrono::steady_clock::now();
        double elapsedSeconds = std::chrono::duration<double>(now - lastUpdateTime_).count();
        
        // Queue sizes
        metrics.requestQueueSize = shared_->requestQueue.size();
        metrics.resultQueueSize = shared_->resultQueue.size();
        metrics.marketDataQueueSize = shared_->marketDataQueue.size();
        
        // Process counts - for a real system we'd use atomics in shared memory
        // For this example, we'll estimate based on queue sizes
        uint64_t currentRequestCount = lastRequestCount_ + metrics.requestQueueSize;
        uint64_t currentResultCount = lastResultCount_ + metrics.resultQueueSize;
        uint64_t currentMarketUpdateCount = lastMarketUpdateCount_ + metrics.marketDataQueueSize;
        
        // Calculate rates
        metrics.requestsPerSecond = static_cast<uint64_t>((currentRequestCount - lastRequestCount_) / elapsedSeconds);
        metrics.resultsPerSecond = static_cast<uint64_t>((currentResultCount - lastResultCount_) / elapsedSeconds);
        metrics.marketUpdatesPerSecond = static_cast<uint64_t>((currentMarketUpdateCount - lastMarketUpdateCount_) / elapsedSeconds);
        
        // Update last counts
        lastRequestCount_ = currentRequestCount;
        lastResultCount_ = currentResultCount;
        lastMarketUpdateCount_ = currentMarketUpdateCount;
        
        // Heartbeat
        uint64_t lastHeartbeat = shared_->lastHeartbeatNanos.load();
        uint64_t now_nanos = EventJournal::getCurrentNanos();
        metrics.lastHeartbeatAge = (now_nanos - lastHeartbeat) / 1000000; // convert to milliseconds
        metrics.pricingEngineAlive = (metrics.lastHeartbeatAge < 5000); // Consider alive if heartbeat within 5 seconds
        
        // System metrics - in a real monitor, would read from /proc
        metrics.cpuUsage = 50.0 + 20.0 * sin(now.time_since_epoch().count() * 0.00000001); // Simulated fluctuation
        metrics.memoryUsage = 500 * 1024 * 1024; // Simulated 500MB
        
        // Latency metrics - would be calculated from actual measurements
        metrics.avgLatencyMicros = 200.0 + 50.0 * sin(now.time_since_epoch().count() * 0.0000001); // Simulated fluctuation
        metrics.p95LatencyMicros = metrics.avgLatencyMicros * 1.5;
        metrics.p99LatencyMicros = metrics.avgLatencyMicros * 2.0;
        metrics.maxLatencyMicros = metrics.avgLatencyMicros * 3.0;
        
        // Update history
        history_.addMetrics(metrics);
        
        // Update last update time
        lastUpdateTime_ = now;
        
        return metrics;
    }
    
    // Get historical metrics
    const HistoricalMetrics& getHistory() const {
        return history_;
    }
};

// Console UI for displaying metrics
class ConsoleUI {
private:
    SystemMonitor& monitor_;
    int rows_ = 24;
    int cols_ = 80;
    
public:
    explicit ConsoleUI(SystemMonitor& monitor) : monitor_(monitor) {
        // Get terminal size if possible
        updateTerminalSize();
    }
    
    // Update terminal size
    void updateTerminalSize() {
        // Use termios to get terminal size
        struct winsize w;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &w);
        
        // Only update if we got valid values
        if (w.ws_row > 0) rows_ = w.ws_row;
        if (w.ws_col > 0) cols_ = w.ws_col;
    }
    
    // Clear screen
    void clearScreen() {
        std::cout << "\033[2J\033[1;1H";
    }
    
    // Draw a progress bar
    void drawProgressBar(double value, double max, int width, const std::string& color = ANSI_COLOR_GREEN) {
        int filledWidth = static_cast<int>(width * (value / max));
        filledWidth = std::min(filledWidth, width);
        
        std::cout << "[";
        std::cout << color;
        
        for (int i = 0; i < filledWidth; ++i) {
            std::cout << "=";
        }
        
        std::cout << ANSI_COLOR_RESET;
        
        for (int i = filledWidth; i < width; ++i) {
            std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << value;
    }
    
    // Draw a spark line from history
    template<typename T>
    void drawSparkLine(const std::deque<T>& history, int width, const std::string& color = ANSI_COLOR_CYAN) {
        // Calculate scaling
        T maxValue = 1;
        if (!history.empty()) {
            maxValue = *std::max_element(history.begin(), history.end());
            if (maxValue < 1) maxValue = 1;
        }
        
        std::cout << color;
        
        // Draw the spark line
        for (int i = 0; i < width; ++i) {
            size_t index = 0;
            if (!history.empty()) {
                index = history.size() * i / width;
                index = std::min(index, history.size() - 1);
            }
            
            if (history.empty() || index >= history.size()) {
                std::cout << " ";
                continue;
            }
            
            T value = history[index];
            int height = static_cast<int>(8.0 * value / maxValue);
            
            // Use block characters for the spark line
            switch (height) {
                case 0: std::cout << " "; break;
                case 1: std::cout << "▁"; break;
                case 2: std::cout << "▂"; break;
                case 3: std::cout << "▃"; break;
                case 4: std::cout << "▄"; break;
                case 5: std::cout << "▅"; break;
                case 6: std::cout << "▆"; break;
                case 7: std::cout << "▇"; break;
                default: std::cout << "█"; break;
            }
        }
        
        std::cout << ANSI_COLOR_RESET;
    }
    
    // Refresh the display with current metrics
    void refresh() {
        // Check if connected
        if (!monitor_.isConnected()) {
            clearScreen();
            std::cout << ANSI_COLOR_RED << "Not connected to pricing system shared memory" << ANSI_COLOR_RESET << std::endl;
            std::cout << "Waiting for connection..." << std::endl;
            return;
        }
        
        // Update terminal size
        updateTerminalSize();
        
        // Get current metrics
        SystemMetrics metrics = monitor_.getMetrics();
        const HistoricalMetrics& history = monitor_.getHistory();
        
        // Clear screen and draw UI
        clearScreen();
        
        // Title
        std::cout << ANSI_BOLD << ANSI_COLOR_CYAN << "Deterministic Derivatives Pricing System Monitor" << ANSI_COLOR_RESET << std::endl;
        std::cout << "─────────────────────────────────────────────────────────────────────────" << std::endl;
        
        // System status
        std::cout << ANSI_BOLD << "System Status:" << ANSI_COLOR_RESET << " ";
        if (metrics.pricingEngineAlive) {
            std::cout << ANSI_COLOR_GREEN << "ONLINE" << ANSI_COLOR_RESET;
        } else {
            std::cout << ANSI_COLOR_RED << "OFFLINE" << ANSI_COLOR_RESET;
        }
        std::cout << " | Heartbeat Age: ";
        if (metrics.lastHeartbeatAge < 1000) {
            std::cout << ANSI_COLOR_GREEN << metrics.lastHeartbeatAge << " ms" << ANSI_COLOR_RESET;
        } else if (metrics.lastHeartbeatAge < 3000) {
            std::cout << ANSI_COLOR_YELLOW << metrics.lastHeartbeatAge << " ms" << ANSI_COLOR_RESET;
        } else {
            std::cout << ANSI_COLOR_RED << metrics.lastHeartbeatAge << " ms" << ANSI_COLOR_RESET;
        }
        
        // CPU and Memory
        std::cout << " | CPU: ";
        std::string cpuColor = (metrics.cpuUsage < 70.0) ? ANSI_COLOR_GREEN : 
                              (metrics.cpuUsage < 90.0) ? ANSI_COLOR_YELLOW : ANSI_COLOR_RED;
        std::cout << cpuColor << std::fixed << std::setprecision(1) << metrics.cpuUsage << "%" << ANSI_COLOR_RESET;
        
        std::cout << " | Memory: " << (metrics.memoryUsage / (1024 * 1024)) << " MB" << std::endl;
        
        std::cout << std::endl;
        
        // Queue status
        std::cout << ANSI_BOLD << "Queue Status:" << ANSI_COLOR_RESET << std::endl;
        
        std::cout << "  Request Queue: " << std::setw(5) << metrics.requestQueueSize << " items | ";
        std::string requestColor = (metrics.requestsPerSecond < 5000) ? ANSI_COLOR_GREEN : 
                                  (metrics.requestsPerSecond < 8000) ? ANSI_COLOR_YELLOW : ANSI_COLOR_RED;
        std::cout << "Rate: " << requestColor << std::setw(7) << metrics.requestsPerSecond << "/s" << ANSI_COLOR_RESET << " | ";
        
        std::cout << "History: ";
        drawSparkLine(history.requestsPerSecond, 30, requestColor);
        std::cout << std::endl;
        
        std::cout << "  Result Queue:  " << std::setw(5) << metrics.resultQueueSize << " items | ";
        std::string resultColor = (metrics.resultsPerSecond < 5000) ? ANSI_COLOR_GREEN : 
                                 (metrics.resultsPerSecond < 8000) ? ANSI_COLOR_YELLOW : ANSI_COLOR_RED;
        std::cout << "Rate: " << resultColor << std::setw(7) << metrics.resultsPerSecond << "/s" << ANSI_COLOR_RESET << " | ";
        
        std::cout << "History: ";
        drawSparkLine(history.resultsPerSecond, 30, resultColor);
        std::cout << std::endl;
        
        std::cout << "  Market Data:   " << std::setw(5) << metrics.marketDataQueueSize << " items | ";
        std::cout << "Rate: " << std::setw(7) << metrics.marketUpdatesPerSecond << "/s" << std::endl;
        
        std::cout << std::endl;
        
        // Latency metrics
        std::cout << ANSI_BOLD << "Latency Metrics:" << ANSI_COLOR_RESET << std::endl;
        
        std::cout << "  Average: " << std::fixed << std::setprecision(2) << metrics.avgLatencyMicros << " μs | ";
        std::cout << "95th: " << std::fixed << std::setprecision(2) << metrics.p95LatencyMicros << " μs | ";
        std::cout << "99th: " << std::fixed << std::setprecision(2) << metrics.p99LatencyMicros << " μs | ";
        std::cout << "Max: " << std::fixed << std::setprecision(2) << metrics.maxLatencyMicros << " μs" << std::endl;
        
        std::cout << "  History: ";
        std::string latencyColor = (metrics.avgLatencyMicros < 300.0) ? ANSI_COLOR_GREEN : 
                                  (metrics.avgLatencyMicros < 500.0) ? ANSI_COLOR_YELLOW : ANSI_COLOR_RED;
        drawSparkLine(history.avgLatencyMicros, 60, latencyColor);
        std::cout << std::endl;
        
        std::cout << std::endl;
        
        // Performance analysis
        double throughputRatio = metrics.resultsPerSecond / 
                                (metrics.requestsPerSecond > 0 ? metrics.requestsPerSecond : 1.0);
        
        std::cout << ANSI_BOLD << "Performance Analysis:" << ANSI_COLOR_RESET << std::endl;
        
        std::cout << "  Throughput Ratio: ";
        std::string ratioColor = (throughputRatio > 0.95) ? ANSI_COLOR_GREEN : 
                                (throughputRatio > 0.8) ? ANSI_COLOR_YELLOW : ANSI_COLOR_RED;
        std::cout << ratioColor << std::fixed << std::setprecision(2) << throughputRatio << ANSI_COLOR_RESET;
        std::cout << " (Results/Requests)" << std::endl;
        
        std::cout << "  Avg. Processing Time: " << std::fixed << std::setprecision(2) 
                  << metrics.avgLatencyMicros << " μs" << std::endl;
        
        std::cout << "  Max Theoretical Throughput: " << 
                  static_cast<int>(1000000.0 / metrics.avgLatencyMicros) << " ops/sec" << std::endl;
        
        std::cout << std::endl;
        
        // Footer
        std::cout << "─────────────────────────────────────────────────────────────────────────" << std::endl;
        std::cout << "Press Ctrl+C to exit" << std::endl;
    }
};

int main(int argc, char** argv) {
    // Register signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Disable output buffering for more responsive UI
    setvbuf(stdout, NULL, _IONBF, 0);
    
    // Create and initialize the system monitor
    SystemMonitor monitor;
    
    // Try to connect to the shared memory
    if (!monitor.connect()) {
        std::cerr << "Failed to connect to pricing system shared memory" << std::endl;
        std::cerr << "Will continue trying to connect..." << std::endl;
    }
    
    // Create the console UI
    ConsoleUI ui(monitor);
    
    // Refresh at 1-second intervals
    while (g_running) {
        ui.refresh();
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        // If not connected, try to connect
        if (!monitor.isConnected()) {
            monitor.connect();
        }
    }
    
    std::cout << "Monitor shutting down..." << std::endl;
    return 0;
}