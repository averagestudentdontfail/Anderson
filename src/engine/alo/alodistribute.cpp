#include "alodistribute.h"
#include "aloengine.h"
#include <stdexcept>
#include <algorithm>
#include <chrono>

namespace engine {
namespace alo {
namespace dist {

TaskDispatcher::TaskDispatcher(ALOScheme engineScheme, size_t chunkSize)
    : chunkSize_(chunkSize), terminated_(false) {
    
    // Initialize MPI if not already done
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) {
        int provided;
        MPI_Init_thread(nullptr, nullptr, MPI_THREAD_MULTIPLE, &provided);
    }
    
    // Get rank and size
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize_);
    
    // Create local engine
    localEngine_ = std::make_unique<ALOEngine>(engineScheme);
    
    // Start the worker thread for work stealing if we have multiple ranks
    if (worldSize_ > 1) {
        workerThread_ = std::thread(&TaskDispatcher::workerThreadFunc, this);
    }
}

TaskDispatcher::~TaskDispatcher() {
    terminated_ = true;
    
    // Join worker thread if it exists
    if (workerThread_.joinable()) {
        workerThread_.join();
    }
    
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
    
    // For small batches or single-node runs, just do the calculation locally
    if (n <= chunkSize_ || worldSize_ <= 1) {
        return localEngine_->batchCalculatePut(S, strikes, r, q, vol, T);
    }
    
    if (rank_ == 0) {
        // Master node: distribute work and collect results
        results = masterNodeProcessing(S, strikes, r, q, vol, T);
    }
    else {
        // Worker node: receive work and send results
        workerNodeProcessing();
        
        // After all work is done, receive the final results from master
        MPI_Bcast(results.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    return results;
}

std::vector<double> TaskDispatcher::masterNodeProcessing(
    double S, const std::vector<double>& strikes,
    double r, double q, double vol, double T) {
    
    const size_t n = strikes.size();
    std::vector<double> results(n);
    
    // Calculate initial distribution
    size_t chunkCount = (n + chunkSize_ - 1) / chunkSize_;
    std::vector<size_t> workload(worldSize_, 0);
    
    // Try to distribute work evenly
    for (size_t i = 0; i < chunkCount; ++i) {
        workload[i % worldSize_]++;
    }
    
    // Send work to worker nodes
    std::vector<std::future<void>> sendTasks;
    
    for (int node = 1; node < worldSize_; ++node) {
        if (workload[node] > 0) {
            // Send work asynchronously to avoid blocking
            sendTasks.push_back(std::async(std::launch::async, 
                [this, node, &workload, S, &strikes, r, q, vol, T, n]() {
                    sendWorkToNode(node, workload, S, strikes, r, q, vol, T, n);
                }
            ));
        }
    }
    
    // Process master node's portion
    if (workload[0] > 0) {
        processLocalWork(0, workload, S, strikes, r, q, vol, T, n, results);
    }
    
    // Wait for all send tasks to complete
    for (auto& task : sendTasks) {
        task.wait();
    }
    
    // Collect results from worker nodes
    std::vector<std::future<void>> receiveTasks;
    
    for (int node = 1; node < worldSize_; ++node) {
        if (workload[node] > 0) {
            // Receive results asynchronously
            receiveTasks.push_back(std::async(std::launch::async, 
                [this, node, &workload, &results, n]() {
                    receiveResultsFromNode(node, workload, results, n);
                }
            ));
        }
    }
    
    // Wait for all receive tasks to complete
    for (auto& task : receiveTasks) {
        task.wait();
    }
    
    // Implement dynamic work stealing for remaining work
    implementWorkStealing(S, strikes, r, q, vol, T, results);
    
    // Signal all workers to terminate
    for (int node = 1; node < worldSize_; ++node) {
        int terminate = 1;
        MPI_Send(&terminate, 1, MPI_INT, node, TAG_TERMINATE, MPI_COMM_WORLD);
    }
    
    // Broadcast final results to all nodes
    MPI_Bcast(results.data(), n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    return results;
}

void TaskDispatcher::sendWorkToNode(int node, const std::vector<size_t>& workload, 
                   double S, const std::vector<double>& strikes,
                   double r, double q, double vol, double T, size_t n) {
    // Send parameters
    double params[5] = {S, r, q, vol, T};
    MPI_Send(params, 5, MPI_DOUBLE, node, TAG_PARAMETERS, MPI_COMM_WORLD);
    
    // Send number of chunks
    auto nodeWorkload = workload[node];
    MPI_Send(&nodeWorkload, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_COUNT, MPI_COMM_WORLD);
    
    // Calculate start index for this node
    size_t startIdx = 0;
    for (int i = 0; i < node; ++i) {
        startIdx += workload[i] * chunkSize_;
    }
    
    // Send each chunk
    for (size_t chunk = 0; chunk < workload[node]; ++chunk) {
        size_t chunkStart = startIdx + chunk * chunkSize_;
        size_t chunkEnd = std::min(chunkStart + chunkSize_, n);
        size_t chunkSize = chunkEnd - chunkStart;
        
        // Send chunk size
        MPI_Send(&chunkSize, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_SIZE, MPI_COMM_WORLD);
        
        // Send chunk indices
        MPI_Send(&chunkStart, 1, MPI_UNSIGNED_LONG, node, TAG_CHUNK_DATA, MPI_COMM_WORLD);
        
        // Send chunk strikes
        std::vector<double> chunkStrikes(strikes.begin() + chunkStart, strikes.begin() + chunkEnd);
        MPI_Send(chunkStrikes.data(), chunkSize, MPI_DOUBLE, node, TAG_CHUNK_DATA, MPI_COMM_WORLD);
    }
}

void TaskDispatcher::processLocalWork(int nodeIdx, const std::vector<size_t>& workload, 
                     double S, const std::vector<double>& strikes,
                     double r, double q, double vol, double T, 
                     size_t n, std::vector<double>& results) {
    // Calculate start index for this node
    size_t startIdx = 0;
    for (int i = 0; i < nodeIdx; ++i) {
        startIdx += workload[i] * chunkSize_;
    }
    
    // Process each chunk
    for (size_t chunk = 0; chunk < workload[nodeIdx]; ++chunk) {
        size_t chunkStart = startIdx + chunk * chunkSize_;
        size_t chunkEnd = std::min(chunkStart + chunkSize_, n);
        
        // Extract strikes for this chunk
        std::vector<double> chunkStrikes(strikes.begin() + chunkStart, strikes.begin() + chunkEnd);
        
        // Price the chunk
        std::vector<double> chunkResults = localEngine_->batchCalculatePut(
            S, chunkStrikes, r, q, vol, T);
        
        // Copy results to output array
        std::copy(chunkResults.begin(), chunkResults.end(), results.begin() + chunkStart);
    }
}

void TaskDispatcher::receiveResultsFromNode(int node, const std::vector<size_t>& workload, 
                          std::vector<double>& results, size_t n) {
    // Calculate indices for this node
    size_t startIdx = 0;
    for (int i = 0; i < node; ++i) {
        startIdx += workload[i] * chunkSize_;
    }
    
    // Receive each chunk of results
    for (size_t chunk = 0; chunk < workload[node]; ++chunk) {
        size_t chunkStart = startIdx + chunk * chunkSize_;
        size_t chunkEnd = std::min(chunkStart + chunkSize_, n);
        size_t chunkSize = chunkEnd - chunkStart;
        
        // Receive chunk results
        MPI_Recv(&results[chunkStart], chunkSize, MPI_DOUBLE, 
               node, TAG_RESULTS, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

void TaskDispatcher::workerNodeProcessing() {
    while (true) {
        // Receive parameters
        double params[5];
        MPI_Status status;
        
        // Check for termination signal
        int terminate = 0;
        MPI_Iprobe(0, TAG_TERMINATE, MPI_COMM_WORLD, &terminate, &status);
        if (terminate) {
            // Receive and acknowledge termination
            MPI_Recv(&terminate, 1, MPI_INT, 0, TAG_TERMINATE, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }
        
        // Wait for parameters message
        int has_params = 0;
        MPI_Iprobe(0, TAG_PARAMETERS, MPI_COMM_WORLD, &has_params, &status);
        
        if (!has_params) {
            // No work received yet, sleep a bit to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        
        // Receive parameters
        MPI_Recv(params, 5, MPI_DOUBLE, 0, TAG_PARAMETERS, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        double S = params[0];
        double r = params[1];
        double q = params[2];
        double vol = params[3];
        double T = params[4];
        
        // Receive number of chunks
        size_t numChunks;
        MPI_Recv(&numChunks, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_COUNT, 
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Process each chunk
        for (size_t chunk = 0; chunk < numChunks; ++chunk) {
            // Receive chunk size
            size_t chunkSize;
            MPI_Recv(&chunkSize, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_SIZE, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Receive chunk start index
            size_t chunkStart;
            MPI_Recv(&chunkStart, 1, MPI_UNSIGNED_LONG, 0, TAG_CHUNK_DATA, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Receive chunk strikes
            std::vector<double> chunkStrikes(chunkSize);
            MPI_Recv(chunkStrikes.data(), chunkSize, MPI_DOUBLE, 0, TAG_CHUNK_DATA, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            // Process chunk
            std::vector<double> chunkResults = localEngine_->batchCalculatePut(
                S, chunkStrikes, r, q, vol, T);
            
            // Send results back
            MPI_Send(chunkResults.data(), chunkSize, MPI_DOUBLE, 
                   0, TAG_RESULTS, MPI_COMM_WORLD);
        }
        
        // Signal readiness for work stealing
        signalReadiness(S, r, q, vol, T);
    }
}

void TaskDispatcher::workerThreadFunc() {
    while (!terminated_) {
        WorkItem workItem;
        bool hasWork = false;
        
        // Try to get work from the queue
        {
            std::unique_lock<std::mutex> lock(workQueueMutex_);
            
            if (!workQueue_.empty()) {
                workItem = workQueue_.front();
                workQueue_.pop();
                hasWork = true;
            }
        }
        
        if (hasWork) {
            // Process the work item
            std::vector<double> results = localEngine_->batchCalculatePut(
                workItem.S, workItem.strikes, workItem.r, workItem.q, 
                workItem.vol, workItem.T);
            
            // Send results back to master
            MPI_Send(results.data(), results.size(), MPI_DOUBLE, 
                   0, TAG_RESULTS, MPI_COMM_WORLD);
        } else {
            // No work, sleep a bit to avoid busy waiting
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void TaskDispatcher::signalReadiness(double S, double r, double q, double vol, double T) {
    // Send ready signal to master
    double params[5] = {S, r, q, vol, T};
    MPI_Send(params, 5, MPI_DOUBLE, 0, TAG_READY, MPI_COMM_WORLD);
}

void TaskDispatcher::implementWorkStealing(
    double S, const std::vector<double>& strikes,
    double r, double q, double vol, double T,
    std::vector<double>& results) {
    // Simplified implementation - OpenMP can handle work distribution internally
    // This is a stub implementation for now
    // In a production system, we would implement more sophisticated work stealing

    // Track which workers are ready for more work
    std::vector<bool> workerReady(worldSize_, false);
    
    // Check for ready workers
    for (int node = 1; node < worldSize_; ++node) {
        int ready = 0;
        MPI_Status status;
        MPI_Iprobe(node, TAG_READY, MPI_COMM_WORLD, &ready, &status);
        
        if (ready) {
            // Node is ready for more work
            double params[5];
            MPI_Recv(params, 5, MPI_DOUBLE, node, TAG_READY, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            workerReady[node] = true;
        }
    }
    
    // Currently, we don't redistribute additional work
    // This could be extended in the future
}

} // namespace dist
} // namespace alo
} // namespace engine