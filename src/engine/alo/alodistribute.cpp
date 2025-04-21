#include "alodistribute.h"
#include "aloengine.h"
#include "aloscheme.h"  
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <cmath>
#include <memory>

namespace engine {
namespace alo {
namespace dist {

TaskDispatcher::TaskDispatcher(ALOScheme engineScheme, size_t chunkSize)
    : chunkSize_(chunkSize), terminated_(false), adaptiveChunking_(false), 
      workStealingThreshold_(0.75) {
    
    // Initialize MPI with thread support
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
        if (provided < MPI_THREAD_MULTIPLE) {
            throw std::runtime_error("MPI does not provide required thread support");
        }
    }
    
    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
    
    // Create local engine
    localEngine_ = std::make_unique<ALOEngine>(engineScheme);
    
    // Start threads for advanced functionality
    if (worldSize_ > 1) {
        workerThread_ = std::thread(&TaskDispatcher::workerThreadFunc, this);
        stealingThread_ = std::thread(&TaskDispatcher::stealingThreadFunc, this);
        profilingThread_ = std::thread(&TaskDispatcher::profilingThreadFunc, this);
    }
    
    // Initialize metrics
    metrics_.startTime = std::chrono::steady_clock::now();
}

TaskDispatcher::~TaskDispatcher() {
    terminated_ = true;
    queueCV_.notify_all();
    
    // Join all threads
    if (workerThread_.joinable()) workerThread_.join();
    if (stealingThread_.joinable()) stealingThread_.join();
    if (profilingThread_.joinable()) profilingThread_.join();
    
    // Finalize MPI if we initialized it
    int finalized;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
}

std::vector<double> TaskDispatcher::distributedBatchCalculatePut(
    double S, const std::vector<double>& strikes,
    double r, double q, double vol, double T) {
    
    const size_t n = strikes.size();
    std::vector<double> results(n);
    
    // For small batches or single-node runs, use local computation
    if (n <= chunkSize_ || worldSize_ <= 1) {
        return localEngine_->batchCalculatePut(S, strikes, r, q, vol, T);
    }
    
    if (rank_ == 0) {
        // Master node: use advanced processing
        results = masterNodeProcessingAdvanced(S, strikes, r, q, vol, T);
    } else {
        // Worker node: enter optimized worker mode
        workerNodeOptimized();
        
        // Receive final results
        MPI_Bcast(results.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    return results;
}

std::vector<double> TaskDispatcher::masterNodeProcessingAdvanced(
    double S, const std::vector<double>& strikes,
    double r, double q, double vol, double T) {
    
    const size_t n = strikes.size();
    std::vector<double> results(n);
    
    // Calculate optimal chunk size if adaptive chunking is enabled
    size_t adjustedChunkSize = adaptiveChunking_ ? 
        calculateOptimalChunkSize(n) : chunkSize_;
    
    // Create work items with priorities
    std::vector<WorkItem> workItems;
    for (size_t i = 0; i < n; i += adjustedChunkSize) {
        size_t end = std::min(i + adjustedChunkSize, n);
        std::vector<double> chunkStrikes(strikes.begin() + i, strikes.begin() + end);
        
        WorkItem work{S, chunkStrikes, r, q, vol, T, i, 
                     static_cast<uint32_t>(n - i), // Priority: larger chunks first
                     std::chrono::steady_clock::now()};
        
        workItems.push_back(work);
    }
    
    // Distribute work using non-blocking sends
    for (size_t i = 0; i < workItems.size(); ++i) {
        int targetNode = (i % (worldSize_ - 1)) + 1;  // Round-robin excluding master
        
        if (targetNode < worldSize_) {
            // Asynchronous send to improve throughput
            sendWorkToNodeNonBlocking(targetNode, workItems[i]);
        } else {
            // Process locally
            processLocalWorkOptimized(workItems[i], results);
        }
    }
    
    // Collect results using non-blocking receives
    for (int node = 1; node < worldSize_; ++node) {
        receiveResultsNonBlocking(node, results);
    }
    
    // Perform dynamic load balancing during execution
    performLoadBalancing();
    
    // Signal termination to all workers
    for (int node = 1; node < worldSize_; ++node) {
        int terminate = 1;
        MPI_Send(&terminate, 1, MPI_INT, node, TAG_TERMINATE, MPI_COMM_WORLD);
    }
    
    // Broadcast final results
    MPI_Bcast(results.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return results;
}

void TaskDispatcher::sendWorkToNodeNonBlocking(int node, const WorkItem& work) {
    // First, prepare the parameter packet
    double params[5] = {work.S, work.r, work.q, work.vol, work.T};
    
    // Use immediate send for improved performance
    MPI_Request req;
    MPI_Isend(params, 5, MPI_DOUBLE, node, TAG_PARAMETERS, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);  // Complete the send in the background
    
    // Send chunk size
    size_t chunkSize = work.strikes.size();
    MPI_Isend(&chunkSize, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_SIZE, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
    
    // Send start index
    MPI_Isend(&work.startIdx, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_DATA, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
    
    // Send strikes array
    MPI_Isend(work.strikes.data(), chunkSize, MPI_DOUBLE, node, TAG_CHUNK_DATA, MPI_COMM_WORLD, &req);
    MPI_Request_free(&req);
    
    // Update metrics
    metrics_.bytesSent += sizeof(params) + sizeof(chunkSize) + sizeof(work.startIdx) + 
                          chunkSize * sizeof(double);
}

void TaskDispatcher::processLocalWorkOptimized(WorkItem& work, std::vector<double>& results) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // Use SIMD-optimized batch processing
    std::vector<double> localResults = localEngine_->batchCalculatePut(
        work.S, work.strikes, work.r, work.q, work.vol, work.T);
    
    // Copy results to output array
    std::copy(localResults.begin(), localResults.end(), results.begin() + work.startIdx);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    
    // Update statistics
    updateProcessingStatistics(work, processingTime);
}

void TaskDispatcher::receiveResultsNonBlocking(int node, std::vector<double>& results) {
    // Set up non-blocking receives for each node's results
    MPI_Status status;
    int messageAvailable = 0;
    
    // Keep receiving until all results from this node are collected
    while (true) {
        // Check if there's a message available
        MPI_Iprobe(node, TAG_RESULTS, MPI_COMM_WORLD, &messageAvailable, &status);
        
        if (!messageAvailable) {
            // No more results from this node for now
            break;
        }
        
        // Get message size
        int count;
        MPI_Get_count(&status, MPI_DOUBLE, &count);
        
        // Allocate buffer for results
        std::vector<double> chunkResults(count);
        
        // Receive results
        MPI_Recv(chunkResults.data(), count, MPI_DOUBLE, node, TAG_RESULTS, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive start index for this chunk
        size_t startIdx;
        MPI_Recv(&startIdx, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_DATA, 
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Copy results to the main array
        std::copy(chunkResults.begin(), chunkResults.end(), results.begin() + startIdx);
        
        // Update metrics
        metrics_.bytesReceived += count * sizeof(double);
    }
}

void TaskDispatcher::workerThreadFunc() {
    while (!terminated_) {
        WorkItem workItem;
        bool hasWork = false;
        
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            
            // Wait for work with timeout to handle termination
            if (queueCV_.wait_for(lock, std::chrono::milliseconds(100),
                [this] { return !workQueue_.empty() || terminated_; })) {
                
                if (!workQueue_.empty()) {
                    workItem = workQueue_.top();
                    workQueue_.pop();
                    hasWork = true;
                }
            }
        }
        
        if (hasWork) {
            // Process the work item
            std::vector<double> results = localEngine_->batchCalculatePut(
                workItem.S, workItem.strikes, workItem.r, workItem.q, 
                workItem.vol, workItem.T);
            
            // Send results back to master using non-blocking send
            MPI_Request req;
            MPI_Isend(results.data(), results.size(), MPI_DOUBLE, 
                     0, TAG_RESULTS, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);
            
            // Send start index
            MPI_Isend(&workItem.startIdx, 1, MPI_UNSIGNED_LONG, 
                     0, TAG_CHUNK_DATA, MPI_COMM_WORLD, &req);
            MPI_Request_free(&req);
            
            // Update metrics
            metrics_.tasksProcessed++;
            metrics_.bytesSent += results.size() * sizeof(double);
        }
    }
}

void TaskDispatcher::stealingThreadFunc() {
    while (!terminated_) {
        // Use exponential backoff for work stealing attempts
        static int backoffMs = 10;
        std::this_thread::sleep_for(std::chrono::milliseconds(backoffMs));
        
        if (workQueue_.empty() || workQueue_.size() < 2) {
            if (attemptWorkStealing()) {
                backoffMs = 10;  // Reset backoff on success
            } else {
                backoffMs = std::min(backoffMs * 2, 1000);  // Exponential backoff
            }
        }
    }
}

void TaskDispatcher::profilingThreadFunc() {
    while (!terminated_) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        if (adaptiveChunking_) {
            adjustAdaptiveParameters();
        }
    }
}

bool TaskDispatcher::attemptWorkStealing() {
    // Don't attempt if we're already busy
    if (workQueue_.size() > workStealingThreshold_ * chunkSize_) {
        return false;
    }
    
    // Choose a random victim node
    if (worldSize_ <= 2) return false;  // Need at least 3 nodes for work stealing
    
    int victim = (rank_ + 1 + (rand() % (worldSize_ - 2))) % worldSize_;
    if (victim == 0) victim = 1;  // Avoid master node
    
    // Send work stealing request
    int request = 1;
    MPI_Send(&request, 1, MPI_INT, victim, TAG_WORK_REQUEST, MPI_COMM_WORLD);
    
    // Wait for response with timeout
    MPI_Status status;
    int flag = 0;
    MPI_Request recv_req;
    MPI_Irecv(&request, 1, MPI_INT, victim, TAG_WORK_RESPONSE, MPI_COMM_WORLD, &recv_req);
    
    // Wait for response with timeout
    for (int i = 0; i < 50 && !flag; ++i) {  // 500ms timeout
        MPI_Test(&recv_req, &flag, &status);
        if (!flag) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
    
    if (flag && request > 0) {
        // Receive stolen work (simplified for now)
        metrics_.workStealSuccesses++;
        return true;
    }
    
    metrics_.workStealAttempts++;
    return false;
}

void TaskDispatcher::workerNodeOptimized() {
    while (true) {
        // Receive parameters
        double params[5];
        MPI_Status status;
        
        // Check for termination signal
        int terminate = 0;
        MPI_Iprobe(0, TAG_TERMINATE, MPI_COMM_WORLD, &terminate, &status);
        if (terminate) {
            MPI_Recv(&terminate, 1, MPI_INT, 0, TAG_TERMINATE, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }
        
        // Check for parameters
        int has_params = 0;
        MPI_Iprobe(0, TAG_PARAMETERS, MPI_COMM_WORLD, &has_params, &status);
        
        if (!has_params) {
            // Try work stealing if no direct work
            if (workQueue_.empty()) {
                attemptWorkStealing();
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Receive parameters
        MPI_Recv(params, 5, MPI_DOUBLE, 0, TAG_PARAMETERS, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Extract parameters
        double S = params[0];
        double r = params[1];
        double q = params[2];
        double vol = params[3];
        double T = params[4];
        
        // Receive chunk size
        size_t chunkSize;
        MPI_Recv(&chunkSize, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_SIZE, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive start index
        size_t startIdx;
        MPI_Recv(&startIdx, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_DATA, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Receive strikes
        std::vector<double> strikes(chunkSize);
        MPI_Recv(strikes.data(), chunkSize, MPI_DOUBLE, 0, TAG_CHUNK_DATA, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Process chunk
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> results = localEngine_->batchCalculatePut(
            S, strikes, r, q, vol, T);
        auto end = std::chrono::high_resolution_clock::now();
        
        // Send results back
        MPI_Send(results.data(), results.size(), MPI_DOUBLE, 
               0, TAG_RESULTS, MPI_COMM_WORLD);
        
        // Send start index
        MPI_Send(&startIdx, 1, MPI_UNSIGNED_LONG, 
               0, TAG_CHUNK_DATA, MPI_COMM_WORLD);
        
        // Update metrics
        double processingTime = std::chrono::duration<double, std::milli>(end - start).count();
        metrics_.tasksProcessed++;
        metrics_.totalLatencyNs += static_cast<uint64_t>(processingTime * 1e6);
    }
}

size_t TaskDispatcher::calculateOptimalChunkSize(size_t totalWork) {
    // Adaptive chunk sizing based on current performance metrics
    size_t baseChunkSize = chunkSize_;
    
    // Adjust based on average processing time
    if (metrics_.tasksProcessed > 10) {
        double avgProcessingTime = static_cast<double>(metrics_.totalLatencyNs) / 
                                 (metrics_.tasksProcessed * 1e6);
        
        // Target 10ms processing time per chunk
        const double targetTimeMs = 10.0;
        
        if (avgProcessingTime > targetTimeMs) {
            baseChunkSize = static_cast<size_t>(baseChunkSize * 0.8);
        } else if (avgProcessingTime < targetTimeMs * 0.5) {
            baseChunkSize = static_cast<size_t>(baseChunkSize * 1.5);
        }
    }
    
    // Ensure reasonable bounds
    baseChunkSize = std::max(size_t(16), std::min(baseChunkSize, totalWork / worldSize_));
    
    return baseChunkSize;
}

void TaskDispatcher::adjustAdaptiveParameters() {
    if (metrics_.workStealAttempts > 0) {
        double successRate = static_cast<double>(metrics_.workStealSuccesses) / 
                           metrics_.workStealAttempts;
        
        if (successRate < 0.2) {
            workStealingThreshold_ *= 0.9;  // Lower threshold to steal more aggressively
        } else if (successRate > 0.8) {
            workStealingThreshold_ *= 1.1;  // Raise threshold to steal less often
        }
        
        // Keep threshold in bounds
        workStealingThreshold_ = std::max(0.1, std::min(0.9, workStealingThreshold_.load()));
    }
}

void TaskDispatcher::performLoadBalancing() {
    // Simple load balancing strategy: redistribute remaining work
    std::vector<int> nodeLoads(worldSize_, 0);
    
    // Query each node for its load (excluding master)
    for (int node = 1; node < worldSize_; ++node) {
        int load = 0;
        MPI_Send(&load, 1, MPI_INT, node, TAG_LOAD_BALANCE, MPI_COMM_WORLD);
        MPI_Recv(&load, 1, MPI_INT, node, TAG_LOAD_BALANCE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        nodeLoads[node] = load;
    }
    
    // Find imbalance and redistribute if necessary
    int minLoad = *std::min_element(nodeLoads.begin() + 1, nodeLoads.end());
    int maxLoad = *std::max_element(nodeLoads.begin() + 1, nodeLoads.end());
    
    if (maxLoad - minLoad > chunkSize_ * 2) {
        // Trigger redistribution (simplified for now)
    }
}

void TaskDispatcher::updateProcessingStatistics(const WorkItem& work, double processingTime) {
    metrics_.tasksProcessed++;
    metrics_.totalLatencyNs += static_cast<uint64_t>(processingTime * 1e6);
    
    // Update adaptive chunking parameters if enabled
    if (adaptiveChunking_) {
        adjustAdaptiveParameters();
    }
}

void TaskDispatcher::setAdaptiveChunking(bool enable) {
    adaptiveChunking_ = enable;
}

void TaskDispatcher::setWorkStealingThreshold(double threshold) {
    workStealingThreshold_ = threshold;
}

PerformanceMetrics TaskDispatcher::getMetrics() const {
    return metrics_;
}

// Legacy methods for backward compatibility
void TaskDispatcher::sendWorkToNode(int node, const std::vector<size_t>& workload, 
                   double S, const std::vector<double>& strikes,
                   double r, double q, double vol, double T, size_t n) {
    // Legacy implementation remains the same as before
    double params[5] = {S, r, q, vol, T};
    MPI_Send(params, 5, MPI_DOUBLE, node, TAG_PARAMETERS, MPI_COMM_WORLD);
    
    auto nodeWorkload = workload[node];
    MPI_Send(&nodeWorkload, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_COUNT, MPI_COMM_WORLD);
    
    size_t startIdx = 0;
    for (int i = 0; i < node; ++i) {
        startIdx += workload[i] * chunkSize_;
    }
    
    for (size_t chunk = 0; chunk < workload[node]; ++chunk) {
        size_t chunkStart = startIdx + chunk * chunkSize_;
        size_t chunkEnd = std::min(chunkStart + chunkSize_, n);
        size_t chunkSize = chunkEnd - chunkStart;
        
        MPI_Send(&chunkSize, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_SIZE, MPI_COMM_WORLD);
        MPI_Send(&chunkStart, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_DATA, MPI_COMM_WORLD);
        
        std::vector<double> chunkStrikes(strikes.begin() + chunkStart, strikes.begin() + chunkEnd);
        MPI_Send(chunkStrikes.data(), chunkSize, MPI_DOUBLE, node, TAG_CHUNK_DATA, MPI_COMM_WORLD);
    }
}

void TaskDispatcher::processLocalWork(int nodeIdx, const std::vector<size_t>& workload, 
                     double S, const std::vector<double>& strikes,
                     double r, double q, double vol, double T, 
                     size_t n, std::vector<double>& results) {
    // Legacy implementation remains the same
    size_t startIdx = 0;
    for (int i = 0; i < nodeIdx; ++i) {
        startIdx += workload[i] * chunkSize_;
    }
    
    for (size_t chunk = 0; chunk < workload[nodeIdx]; ++chunk) {
        size_t chunkStart = startIdx + chunk * chunkSize_;
        size_t chunkEnd = std::min(chunkStart + chunkSize_, n);
        
        std::vector<double> chunkStrikes(strikes.begin() + chunkStart, strikes.begin() + chunkEnd);
        std::vector<double> chunkResults = localEngine_->batchCalculatePut(
            S, chunkStrikes, r, q, vol, T);
        
        std::copy(chunkResults.begin(), chunkResults.end(), results.begin() + chunkStart);
    }
}

void TaskDispatcher::receiveResultsFromNode(int node, const std::vector<size_t>& workload, 
                          std::vector<double>& results, size_t n) {
    // Legacy implementation remains the same
    size_t startIdx = 0;
    for (int i = 0; i < node; ++i) {
        startIdx += workload[i] * chunkSize_;
    }
    
    for (size_t chunk = 0; chunk < workload[node]; ++chunk) {
        size_t chunkStart = startIdx + chunk * chunkSize_;
        size_t chunkEnd = std::min(chunkStart + chunkSize_, n);
        size_t chunkSize = chunkEnd - chunkStart;
        
        MPI_Recv(&results[chunkStart], chunkSize, MPI_DOUBLE, 
               node, TAG_RESULTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void TaskDispatcher::workerNodeProcessing() {
    // Legacy implementation remains the same
    while (true) {
        double params[5];
        MPI_Status status;
        
        int terminate = 0;
        MPI_Iprobe(0, TAG_TERMINATE, MPI_COMM_WORLD, &terminate, &status);
        if (terminate) {
            MPI_Recv(&terminate, 1, MPI_INT, 0, TAG_TERMINATE, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }
        
        int has_params = 0;
        MPI_Iprobe(0, TAG_PARAMETERS, MPI_COMM_WORLD, &has_params, &status);
        
        if (!has_params) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        MPI_Recv(params, 5, MPI_DOUBLE, 0, TAG_PARAMETERS, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        double S = params[0];
        double r = params[1];
        double q = params[2];
        double vol = params[3];
        double T = params[4];
        
        size_t numChunks;
        MPI_Recv(&numChunks, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_COUNT, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        for (size_t chunk = 0; chunk < numChunks; ++chunk) {
            size_t chunkSize;
            MPI_Recv(&chunkSize, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_SIZE, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            size_t chunkStart;
            MPI_Recv(&chunkStart, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_DATA, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            std::vector<double> chunkStrikes(chunkSize);
            MPI_Recv(chunkStrikes.data(), chunkSize, MPI_DOUBLE, 0, TAG_CHUNK_DATA, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            std::vector<double> chunkResults = localEngine_->batchCalculatePut(
                S, chunkStrikes, r, q, vol, T);
            
            MPI_Send(chunkResults.data(), chunkSize, MPI_DOUBLE, 
                   0, TAG_RESULTS, MPI_COMM_WORLD);
        }
        
        signalReadiness(S, r, q, vol, T);
    }
}

void TaskDispatcher::implementWorkStealing(
    double /* S */, const std::vector<double>& /* strikes */,
    double /* r */, double /* q */, double /* vol */, double /* T */,
    std::vector<double>& /* results */) {
    // Legacy implementation - now handled by the advanced work stealing thread
}

void TaskDispatcher::signalReadiness(double S, double r, double q, double vol, double T) {
    double params[5] = {S, r, q, vol, T};
    MPI_Send(params, 5, MPI_DOUBLE, 0, TAG_READY, MPI_COMM_WORLD);
}

// Implementation of createTaskDispatcher
std::shared_ptr<TaskDispatcher> createTaskDispatcher(ALOScheme scheme, size_t chunkSize) {
    return std::make_shared<TaskDispatcher>(scheme, chunkSize);
}

} // namespace dist
} // namespace alo
} // namespace engine