#include "alodistribute.h"
#include "aloengine.h"
#include "aloscheme.h"
#include "mpi_wrapper.h"
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
    
    // Use the new wrapper instead of direct MPI calls
    try {
        mpi::MPIWrapper::init(nullptr, nullptr);
        rank_ = mpi::MPIWrapper::rank();
        worldSize_ = mpi::MPIWrapper::size();
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("MPI initialization failed: ") + e.what());
    }
    
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
    mpi::MPIWrapper::finalize();
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
        mpi::MPIWrapper::bcast(results.data(), n, MPI_DOUBLE, 0);
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
        mpi::MPIWrapper::send(&terminate, 1, MPI_INT, node, TAG_TERMINATE);
    }
    
    // Broadcast final results
    mpi::MPIWrapper::bcast(results.data(), n, MPI_DOUBLE, 0);
    
    return results;
}

void TaskDispatcher::sendWorkToNodeNonBlocking(int node, const WorkItem& work) {
    // First, prepare the parameter packet
    double params[5] = {work.S, work.r, work.q, work.vol, work.T};
    
    // Use immediate send for improved performance
    MPI_Request req;
    mpi::MPIWrapper::isend(params, 5, MPI_DOUBLE, node, TAG_PARAMETERS, &req);
    MPI_Request_free(&req);  // Complete the send in the background
    
    // Send chunk size
    size_t chunkSize = work.strikes.size();
    mpi::MPIWrapper::isend(&chunkSize, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_SIZE, &req);
    MPI_Request_free(&req);
    
    // Send start index
    mpi::MPIWrapper::isend(&work.startIdx, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_DATA, &req);
    MPI_Request_free(&req);
    
    // Send strikes array
    mpi::MPIWrapper::isend(work.strikes.data(), chunkSize, MPI_DOUBLE, node, TAG_CHUNK_DATA, &req);
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
        mpi::MPIWrapper::recv(chunkResults.data(), count, MPI_DOUBLE, node, TAG_RESULTS);
        
        // Receive start index for this chunk
        size_t startIdx;
        mpi::MPIWrapper::recv(&startIdx, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_DATA);
        
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
            mpi::MPIWrapper::isend(results.data(), results.size(), MPI_DOUBLE, 
                                  0, TAG_RESULTS, &req);
            MPI_Request_free(&req);
            
            // Send start index
            mpi::MPIWrapper::isend(&workItem.startIdx, 1, MPI_UNSIGNED_LONG, 
                                  0, TAG_CHUNK_DATA, &req);
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
    mpi::MPIWrapper::send(&request, 1, MPI_INT, victim, TAG_WORK_REQUEST);
    
    // Wait for response with timeout
    MPI_Status status;
    int flag = 0;
    MPI_Request recv_req;
    mpi::MPIWrapper::irecv(&request, 1, MPI_INT, victim, TAG_WORK_RESPONSE, &recv_req);
    
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
            mpi::MPIWrapper::recv(&terminate, 1, MPI_INT, 0, TAG_TERMINATE);
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
        mpi::MPIWrapper::recv(params, 5, MPI_DOUBLE, 0, TAG_PARAMETERS);
        
        // Extract parameters
        double S = params[0];
        double r = params[1];
        double q = params[2];
        double vol = params[3];
        double T = params[4];
        
        // Receive chunk size
        size_t chunkSize;
        mpi::MPIWrapper::recv(&chunkSize, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_SIZE);
        
        // Receive start index
        size_t startIdx;
        mpi::MPIWrapper::recv(&startIdx, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_DATA);
        
        // Receive strikes
        std::vector<double> strikes(chunkSize);
        mpi::MPIWrapper::recv(strikes.data(), chunkSize, MPI_DOUBLE, 0, TAG_CHUNK_DATA);
        
        // Process chunk
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<double> results = localEngine_->batchCalculatePut(
            S, strikes, r, q, vol, T);
        auto end = std::chrono::high_resolution_clock::now();
        
        // Send results back
        mpi::MPIWrapper::send(results.data(), results.size(), MPI_DOUBLE, 0, TAG_RESULTS);
        
        // Send start index
        mpi::MPIWrapper::send(&startIdx, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_DATA);
        
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
        
        // Use atomic operations correctly
        double currentThreshold = workStealingThreshold_.load();
        
        if (successRate < 0.2) {
            currentThreshold *= 0.9;  // Lower threshold to steal more aggressively
        } else if (successRate > 0.8) {
            currentThreshold *= 1.1;  // Raise threshold to steal less often
        }
        
        // Keep threshold in bounds
        currentThreshold = std::max(0.1, std::min(0.9, currentThreshold));
        workStealingThreshold_.store(currentThreshold);
    }
}

void TaskDispatcher::performLoadBalancing() {
    // Simple load balancing strategy: redistribute remaining work
    std::vector<int> nodeLoads(worldSize_, 0);
    
    // Query each node for its load (excluding master)
    for (int node = 1; node < worldSize_; ++node) {
        int load = 0;
        mpi::MPIWrapper::send(&load, 1, MPI_INT, node, TAG_LOAD_BALANCE);
        mpi::MPIWrapper::recv(&load, 1, MPI_INT, node, TAG_LOAD_BALANCE);
        nodeLoads[node] = load;
    }
    
    // Find imbalance and redistribute if necessary
    int minLoad = *std::min_element(nodeLoads.begin() + 1, nodeLoads.end());
    int maxLoad = *std::max_element(nodeLoads.begin() + 1, nodeLoads.end());
    
    if (static_cast<size_t>(maxLoad - minLoad) > chunkSize_ * 2) {
        // Trigger redistribution (simplified for now)
    }
}

void TaskDispatcher::updateProcessingStatistics(const WorkItem& /* work */, double processingTime) {
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


PerformanceMetricsSnapshot TaskDispatcher::getMetrics() const {
    return PerformanceMetricsSnapshot(metrics_);
}

// Legacy methods for backward compatibility
void TaskDispatcher::sendWorkToNode(int node, const std::vector<size_t>& workload, 
                   double S, const std::vector<double>& strikes,
                   double r, double q, double vol, double T, size_t n) {
    // Legacy implementation remains the same as before
    double params[5] = {S, r, q, vol, T};
    mpi::MPIWrapper::send(params, 5, MPI_DOUBLE, node, TAG_PARAMETERS);
    
    auto nodeWorkload = workload[node];
    mpi::MPIWrapper::send(&nodeWorkload, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_COUNT);
    
    size_t startIdx = 0;
    for (int i = 0; i < node; ++i) {
        startIdx += workload[i] * chunkSize_;
    }
    
    for (size_t chunk = 0; chunk < workload[node]; ++chunk) {
        size_t chunkStart = startIdx + chunk * chunkSize_;
        size_t chunkEnd = std::min(chunkStart + chunkSize_, n);
        size_t chunkSize = chunkEnd - chunkStart;
        
        mpi::MPIWrapper::send(&chunkSize, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_SIZE);
        mpi::MPIWrapper::send(&chunkStart, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_DATA);
        
        std::vector<double> chunkStrikes(strikes.begin() + chunkStart, strikes.begin() + chunkEnd);
        mpi::MPIWrapper::send(chunkStrikes.data(), chunkSize, MPI_DOUBLE, node, TAG_CHUNK_DATA);
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
        
        mpi::MPIWrapper::recv(&results[chunkStart], chunkSize, MPI_DOUBLE, 
                            node, TAG_RESULTS);
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
            mpi::MPIWrapper::recv(&terminate, 1, MPI_INT, 0, TAG_TERMINATE);
            break;
        }
        
        int has_params = 0;
        MPI_Iprobe(0, TAG_PARAMETERS, MPI_COMM_WORLD, &has_params, &status);
        
        if (!has_params) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        mpi::MPIWrapper::recv(params, 5, MPI_DOUBLE, 0, TAG_PARAMETERS);
        
        double S = params[0];
        double r = params[1];
        double q = params[2];
        double vol = params[3];
        double T = params[4];
        
        size_t numChunks;
        mpi::MPIWrapper::recv(&numChunks, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_COUNT);
        
        for (size_t chunk = 0; chunk < numChunks; ++chunk) {
            size_t chunkSize;
            mpi::MPIWrapper::recv(&chunkSize, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_SIZE);
            
            size_t chunkStart;
            mpi::MPIWrapper::recv(&chunkStart, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_DATA);
            
            std::vector<double> chunkStrikes(chunkSize);
            mpi::MPIWrapper::recv(chunkStrikes.data(), chunkSize, MPI_DOUBLE, 0, TAG_CHUNK_DATA);
            
            std::vector<double> chunkResults = localEngine_->batchCalculatePut(
                S, chunkStrikes, r, q, vol, T);
            
            mpi::MPIWrapper::send(chunkResults.data(), chunkSize, MPI_DOUBLE, 0, TAG_RESULTS);
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
    mpi::MPIWrapper::send(params, 5, MPI_DOUBLE, 0, TAG_READY);
}

// Implementation of createTaskDispatcher
std::shared_ptr<TaskDispatcher> createTaskDispatcher(ALOScheme scheme, size_t chunkSize) {
    return std::make_shared<TaskDispatcher>(scheme, chunkSize);
}

} // namespace dist
} // namespace alo
} // namespace engine