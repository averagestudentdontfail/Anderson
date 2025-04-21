#ifndef ENGINE_ALO_DIST_FRAMEWORK_H
#define ENGINE_ALO_DIST_FRAMEWORK_H

#include <vector>
#include <future>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <mpi.h>
#include <memory>
#include <thread>
#include <mutex>
#include <chrono>
#include "aloscheme.h"

namespace engine {
namespace alo {

// Forward declarations
class ALOEngine;

namespace dist {

/**
 * @brief Tags for MPI messages
 */
enum MessageTags {
    TAG_PARAMETERS = 0,     // Parameters packet
    TAG_CHUNK_COUNT = 1,    // Number of chunks
    TAG_CHUNK_SIZE = 2,     // Size of a chunk
    TAG_CHUNK_DATA = 3,     // Chunk data
    TAG_RESULTS = 4,        // Results data
    TAG_READY = 5,          // Worker ready signal
    TAG_WORK_REQUEST = 6,   // Work stealing request
    TAG_WORK_RESPONSE = 7,  // Work stealing response
    TAG_TERMINATE = 8,      // Termination signal
    TAG_LOAD_BALANCE = 9,   // Load balancing request
    TAG_PROFILING = 10      // Performance profiling data
};

/**
 * @brief Performance metrics for distributed processing
 */
struct PerformanceMetrics {
    std::atomic<uint64_t> tasksProcessed{0};
    std::atomic<uint64_t> bytesSent{0};
    std::atomic<uint64_t> bytesReceived{0};
    std::atomic<uint64_t> totalLatencyNs{0};
    std::atomic<uint64_t> workStealAttempts{0};
    std::atomic<uint64_t> workStealSuccesses{0};
    std::chrono::steady_clock::time_point startTime;
};

/**
 * @brief Non-atomic version of performance metrics for reporting
 */
struct PerformanceMetricsSnapshot {
    uint64_t tasksProcessed;
    uint64_t bytesSent;
    uint64_t bytesReceived;
    uint64_t totalLatencyNs;
    uint64_t workStealAttempts;
    uint64_t workStealSuccesses;
    std::chrono::steady_clock::time_point startTime;
    
    PerformanceMetricsSnapshot() = default;
    
    // Constructor to create snapshot from atomic metrics
    explicit PerformanceMetricsSnapshot(const PerformanceMetrics& metrics)
        : tasksProcessed(metrics.tasksProcessed.load())
        , bytesSent(metrics.bytesSent.load())
        , bytesReceived(metrics.bytesReceived.load())
        , totalLatencyNs(metrics.totalLatencyNs.load())
        , workStealAttempts(metrics.workStealAttempts.load())
        , workStealSuccesses(metrics.workStealSuccesses.load())
        , startTime(metrics.startTime)
    {}
};

/**
 * @brief Work item with extended metadata
 */
struct WorkItem {
    double S;                       // Spot price
    std::vector<double> strikes;    // Strike prices
    double r;                       // Risk-free rate
    double q;                       // Dividend yield
    double vol;                     // Volatility
    double T;                       // Time to maturity
    size_t startIdx;                // Starting index in global result array
    uint32_t priority;              // Priority level for processing
    std::chrono::steady_clock::time_point timestamp; // Creation timestamp
    
    // Comparison operator for priority queue
    bool operator<(const WorkItem& other) const {
        return priority < other.priority;  // Higher priority first
    }
};

/**
 * @class TaskDispatcher
 * @brief High-performance distributed option pricing framework
 */
class TaskDispatcher {
public:
    /**
     * @brief Constructor with advanced configuration
     */
    TaskDispatcher(ALOScheme engineScheme, size_t chunkSize = 1024);
    
    /**
     * @brief Destructor with clean shutdown
     */
    ~TaskDispatcher();
    
    /**
     * @brief Advanced batch pricing with load balancing
     */
    std::vector<double> distributedBatchCalculatePut(
        double S, const std::vector<double>& strikes,
        double r, double q, double vol, double T);
    
    /**
     * @brief Get performance metrics snapshot
     */
    PerformanceMetricsSnapshot getMetrics() const;
    
    /**
     * @brief Configure adaptive chunking
     */
    void setAdaptiveChunking(bool enable);
    
    /**
     * @brief Set work stealing threshold
     */
    void setWorkStealingThreshold(double threshold);
    
private:
    // Core components
    std::unique_ptr<ALOEngine> localEngine_;
    size_t chunkSize_;
    int rank_;
    int worldSize_;
    std::atomic<bool> terminated_;
    
    // Thread management
    std::thread workerThread_;
    std::thread stealingThread_;
    std::thread profilingThread_;
    std::mutex queueMutex_;
    std::condition_variable queueCV_;
    
    // Work management
    std::priority_queue<WorkItem> workQueue_;
    std::atomic<size_t> pendingTasks_;
    
    // Performance tracking
    PerformanceMetrics metrics_;
    std::atomic<bool> adaptiveChunking_;
    std::atomic<double> workStealingThreshold_;
    
    // Advanced processing methods
    std::vector<double> masterNodeProcessingAdvanced(
        double S, const std::vector<double>& strikes,
        double r, double q, double vol, double T);
    
    void sendWorkToNodeNonBlocking(int node, const WorkItem& work);
    void processLocalWorkOptimized(WorkItem& work, std::vector<double>& results);
    void receiveResultsNonBlocking(int node, std::vector<double>& results);
    
    // Legacy methods for backward compatibility
    void sendWorkToNode(int node, const std::vector<size_t>& workload, 
                       double S, const std::vector<double>& strikes,
                       double r, double q, double vol, double T, size_t n);
    void receiveResultsFromNode(int node, const std::vector<size_t>& workload, 
                              std::vector<double>& results, size_t n);
    void processLocalWork(int nodeIdx, const std::vector<size_t>& workload, 
                         double S, const std::vector<double>& strikes,
                         double r, double q, double vol, double T, 
                         size_t n, std::vector<double>& results);
    
    // Thread functions
    void workerThreadFunc();
    void stealingThreadFunc();
    void profilingThreadFunc();
    
    // Worker node behavior
    void workerNodeProcessing();
    void workerNodeOptimized();
    
    // Load balancing
    void performLoadBalancing();
    bool attemptWorkStealing();
    void handleWorkRequest(int requestingNode);
    
    // Performance optimization
    size_t calculateOptimalChunkSize(size_t totalWork);
    void updateProcessingStatistics(const WorkItem& work, double processingTime);
    void adjustAdaptiveParameters();
    
    // Work handling
    void implementWorkStealing(
        double S, const std::vector<double>& strikes,
        double r, double q, double vol, double T,
        std::vector<double>& results);
    void signalReadiness(double S, double r, double q, double vol, double T);
};

/**
 * @brief Create a high-performance task dispatcher
 * 
 * @param scheme Numerical scheme to use
 * @param chunkSize Initial chunk size
 * @return Shared pointer to task dispatcher
 */
std::shared_ptr<TaskDispatcher> createTaskDispatcher(ALOScheme scheme, size_t chunkSize = 1024);

} // namespace dist
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_DIST_FRAMEWORK_H