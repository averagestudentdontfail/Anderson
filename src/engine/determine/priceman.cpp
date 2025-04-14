#include "priceman.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <random>
#include <sstream>
#include <iomanip>
#include <immintrin.h>

#include "../../alo/aloengine.h"
#include "../../alo/aloscheme.cpp"    

namespace engine {
namespace determine {

// DeterministicPricer implementation
class DeterministicPricer::Impl {
public:
    Impl(int scheme) : engine_(static_cast<ALOScheme>(scheme)), 
                       useVectorization_(true), 
                       fixedExecutionTime_(true) {}
    
    PricingResult price(const PricingRequest& request) {
        PricingResult result;
        result.requestId = request.requestId;
        result.instrumentId = request.instrumentId;
        result.statusCode = 0; // Success
        
        // Fixed execution time with busy-wait to ensure determinism
        uint64_t startCycles = __rdtsc();
        
        try {
            // Actual pricing calculation
            result.price = engine_.calculatePut(
                request.S, request.K, request.r, request.q, request.vol, request.T);
            
            // Calculate Greeks using finite differences
            calculateGreeks(request, result);
        } catch (const std::exception&) {
            result.statusCode = 1; // Error
            result.price = 0.0;
            result.delta = 0.0;
            result.gamma = 0.0;
            result.vega = 0.0;
            result.theta = 0.0;
            result.rho = 0.0;
        }
        
        // Busy-wait if needed to ensure fixed execution time
        if (fixedExecutionTime_) {
            const uint64_t TARGET_CYCLES = 3'000'000; // ~1ms at 3GHz
            while (__rdtsc() - startCycles < TARGET_CYCLES) {
                _mm_pause(); // Reduce power consumption
            }
        }
        
        return result;
    }
    
    std::vector<PricingResult> batchPrice(const std::vector<PricingRequest>& requests) {
        std::vector<PricingResult> results;
        results.reserve(requests.size());
        
        for (const auto& request : requests) {
            results.push_back(price(request));
        }
        
        return results;
    }
    
    void configure(int scheme, bool useVectorization, bool fixedExecutionTime) {
        // Create a new engine with the specified scheme
        engine_ = ALOEngine(static_cast<ALOScheme>(scheme));
        useVectorization_ = useVectorization;
        fixedExecutionTime_ = fixedExecutionTime;
    }
    
    void preWarm(int iterations) {
        // Pre-warm the pricer
        PricingRequest dummyRequest;
        dummyRequest.S = 100.0;
        dummyRequest.K = 100.0;
        dummyRequest.r = 0.05;
        dummyRequest.q = 0.01;
        dummyRequest.vol = 0.2;
        dummyRequest.T = 1.0;
        
        for (int i = 0; i < iterations; ++i) {
            engine_.calculatePut(
                dummyRequest.S, dummyRequest.K, 
                dummyRequest.r, dummyRequest.q, 
                dummyRequest.vol, dummyRequest.T);
        }
    }
    
    void resetCache() {
        engine_.clearCache();
    }
    
    void calculateGreeks(const PricingRequest& req, PricingResult& result) {
        const double h_s = std::max(0.001, req.S * 0.001);  // For delta/gamma
        const double h_vol = std::max(0.0001, req.vol * 0.01); // For vega
        const double h_t = std::min(1.0/365.0, req.T * 0.01); // For theta
        const double h_r = 0.0001; // For rho
        
        // Delta: dV/dS
        double price_up = engine_.calculatePut(req.S + h_s, req.K, req.r, req.q, req.vol, req.T);
        double price_down = engine_.calculatePut(req.S - h_s, req.K, req.r, req.q, req.vol, req.T);
        result.delta = (price_up - price_down) / (2 * h_s);
        
        // Gamma: d²V/dS²
        result.gamma = (price_up - 2 * result.price + price_down) / (h_s * h_s);
        
        // Vega: dV/dσ
        double price_vol_up = engine_.calculatePut(req.S, req.K, req.r, req.q, req.vol + h_vol, req.T);
        result.vega = (price_vol_up - result.price) / h_vol;
        
        // Theta: -dV/dT
        double price_t_down = engine_.calculatePut(req.S, req.K, req.r, req.q, req.vol, req.T - h_t);
        result.theta = -(price_t_down - result.price) / h_t;
        
        // Rho: dV/dr
        double price_r_up = engine_.calculatePut(req.S, req.K, req.r + h_r, req.q, req.vol, req.T);
        result.rho = (price_r_up - result.price) / h_r;
    }
    
private:
    ALOEngine engine_;
    bool useVectorization_;
    bool fixedExecutionTime_;
};

DeterministicPricer::DeterministicPricer(int scheme)
    : impl_(std::make_unique<Impl>(scheme)) {
}

DeterministicPricer::~DeterministicPricer() = default;

PricingResult DeterministicPricer::price(const PricingRequest& request) {
    return impl_->price(request);
}

std::vector<PricingResult> DeterministicPricer::batchPrice(const std::vector<PricingRequest>& requests) {
    return impl_->batchPrice(requests);
}

void DeterministicPricer::configure(int scheme, bool useVectorization, bool fixedExecutionTime) {
    impl_->configure(scheme, useVectorization, fixedExecutionTime);
}

void DeterministicPricer::preWarm(int iterations) {
    impl_->preWarm(iterations);
}

void DeterministicPricer::resetCache() {
    impl_->resetCache();
}

void DeterministicPricer::calculateGreeks(const PricingRequest& request, PricingResult& result) {
    impl_->calculateGreeks(request, result);
}

// PricerPool implementation
PricerPool::PricerPool(size_t poolSize, int scheme)
    : poolSize_(poolSize) {
    
    // Create the pricers
    pricers_.reserve(poolSize);
    for (size_t i = 0; i < poolSize; ++i) {
        pricers_.push_back(std::make_shared<DeterministicPricer>(scheme));
    }
    
    // Pre-warm all pricers
    preWarmAll();
}

std::shared_ptr<DeterministicPricer> PricerPool::getPricer() {
    // Simple round-robin allocation
    size_t index = nextPricerIndex_.fetch_add(1) % poolSize_;
    return pricers_[index];
}

void PricerPool::returnPricer(std::shared_ptr<DeterministicPricer> pricer) {
    // No-op in this implementation since we use a static pool
    // Could be extended to handle dynamic allocation
}

void PricerPool::preWarmAll(int iterations) {
    for (auto& pricer : pricers_) {
        pricer->preWarm(iterations);
    }
}

void PricerPool::resetAllCaches() {
    for (auto& pricer : pricers_) {
        pricer->resetCache();
    }
}

// PricingManager implementation
PricingManager& PricingManager::getInstance() {
    static PricingManager instance;
    return instance;
}

void PricingManager::initialize(size_t numPricers, int scheme, bool useScheduler) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    if (initialized_) {
        // Already initialized
        return;
    }
    
    // Create the pricer pool
    pricerPool_ = std::make_unique<PricerPool>(numPricers, scheme);
    
    // Set up scheduler manager if needed
    useScheduler_ = useScheduler;
    if (useScheduler_) {
        schedulerManager_ = std::make_shared<SchedulerManager>();
        schedulerManager_->initialize(1, 1000000, 32); // 1ms cycle, 32 tasks per cycle
    }
    
    // Initialize performance tracking
    stats_.uptimeTimer.start();
    
    initialized_ = true;
}

bool PricingManager::start() {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    if (!initialized_) {
        std::cerr << "Error: PricingManager must be initialized before starting" << std::endl;
        return false;
    }
    
    if (running_) {
        // Already running
        return true;
    }
    
    // Start the scheduler if we're using it
    if (useScheduler_ && schedulerManager_) {
        if (!schedulerManager_->startAll()) {
            std::cerr << "Error: Failed to start schedulers" << std::endl;
            return false;
        }
    }
    
    running_ = true;
    return true;
}

void PricingManager::stop() {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    if (!running_) {
        // Already stopped
        return;
    }
    
    // Stop the scheduler if we're using it
    if (useScheduler_ && schedulerManager_) {
        schedulerManager_->stopAll();
    }
    
    running_ = false;
}

uint64_t PricingManager::submitRequest(const PricingRequest& request, 
                                     std::function<void(const PricingResult&)> callback) {
    if (!initialized_ || !running_) {
        return 0;
    }
    
    // Assign a request ID
    uint64_t requestId = nextRequestId_.fetch_add(1);
    
    // Create a copy of the request with the assigned ID
    PricingRequest requestCopy = request;
    requestCopy.requestId = requestId;
    
    // Record the callback if provided
    if (callback) {
        std::lock_guard<std::mutex> lock(managerMutex_);
        callbacks_[requestId] = callback;
    }
    
    // Update statistics
    stats_.totalRequests.fetch_add(1);
    size_t queueDepth = stats_.currentQueueDepth.fetch_add(1) + 1;
    stats_.maxQueueDepth.store(std::max(stats_.maxQueueDepth.load(), queueDepth));
    
    // Record in the journal if available
    if (journal_) {
        journal_->recordRequest(requestCopy);
    }
    
    if (useScheduler_ && schedulerManager_) {
        // Schedule the pricing task on the scheduler
        auto task = [this, requestCopy, requestId]() -> int32_t {
            return this->pricingTask(requestCopy, requestId);
        };
        
        schedulerManager_->scheduleFunction(
            task, 
            "Pricing_" + std::to_string(requestId),
            ExecutionMode::DETERMINISTIC);
    } else {
        // Execute directly
        pricingTask(requestCopy, requestId);
    }
    
    return requestId;
}

std::vector<uint64_t> PricingManager::submitBatchRequest(
    const std::vector<PricingRequest>& requests,
    std::function<void(const std::vector<PricingResult>&)> callback) {
    
    if (!initialized_ || !running_ || requests.empty()) {
        return {};
    }
    
    // Create a batch request ID
    uint64_t batchId = nextRequestId_.fetch_add(1);
    
    // Assign individual request IDs
    std::vector<uint64_t> requestIds;
    std::vector<PricingRequest> requestCopies;
    requestIds.reserve(requests.size());
    requestCopies.reserve(requests.size());
    
    for (const auto& request : requests) {
        uint64_t requestId = nextRequestId_.fetch_add(1);
        requestIds.push_back(requestId);
        
        // Create a copy with the assigned ID
        PricingRequest requestCopy = request;
        requestCopy.requestId = requestId;
        requestCopies.push_back(requestCopy);
    }
    
    // Create a batch request object
    auto batchRequest = std::make_shared<BatchRequest>();
    batchRequest->requestIds = requestIds;
    batchRequest->results.resize(requests.size());
    batchRequest->callback = callback;
    
    // Store the batch request
    {
        std::lock_guard<std::mutex> lock(managerMutex_);
        batchRequests_[batchId] = batchRequest;
    }
    
    // Update statistics
    stats_.totalRequests.fetch_add(requests.size());
    stats_.totalBatchRequests.fetch_add(1);
    size_t queueDepth = stats_.currentQueueDepth.fetch_add(requests.size()) + requests.size();
    stats_.maxQueueDepth.store(std::max(stats_.maxQueueDepth.load(), queueDepth));
    
    // Record in the journal if available
    if (journal_) {
        for (const auto& request : requestCopies) {
            journal_->recordRequest(request);
        }
    }
    
    if (useScheduler_ && schedulerManager_) {
        // Schedule the batch pricing task on the scheduler
        auto task = [this, requestCopies, batchId]() -> int32_t {
            return this->batchPricingTask(requestCopies, batchId);
        };
        
        schedulerManager_->scheduleFunction(
            task, 
            "BatchPricing_" + std::to_string(batchId),
            ExecutionMode::DETERMINISTIC);
    } else {
        // Execute directly
        batchPricingTask(requestCopies, batchId);
    }
    
    return requestIds;
}

PricingResult PricingManager::getResult(uint64_t requestId) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    auto it = resultCache_.find(requestId);
    if (it != resultCache_.end()) {
        return it->second;
    }
    
    // Result not found, return an empty result
    PricingResult emptyResult;
    emptyResult.requestId = requestId;
    emptyResult.statusCode = 2;  // Not found
    return emptyResult;
}

bool PricingManager::waitForResult(uint64_t requestId, uint64_t timeoutMs) {
    auto startTime = std::chrono::steady_clock::now();
    
    while (true) {
        // Check if the result is available
        {
            std::lock_guard<std::mutex> lock(managerMutex_);
            if (resultCache_.find(requestId) != resultCache_.end()) {
                return true;
            }
        }
        
        // Check timeout
        if (timeoutMs > 0) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                now - startTime).count();
            
            if (elapsed >= timeoutMs) {
                return false;
            }
        }
        
        // Sleep briefly to avoid spinning
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void PricingManager::processMarketUpdate(const MarketUpdate& update) {
    // Record in the journal if available
    if (journal_) {
        journal_->recordMarketUpdate(update);
    }
    
    // In a real implementation, this would update the market data cache
    // and potentially trigger repricing of affected instruments
    // This is just a placeholder
}

void PricingManager::setJournal(std::shared_ptr<EventJournal> journal) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    journal_ = journal;
}

std::string PricingManager::getStatistics() const {
    // Calculate uptime
    stats_.uptimeTimer.stop();
    double uptimeSeconds = stats_.uptimeTimer.elapsedSeconds();
    stats_.uptimeTimer.start();
    
    std::ostringstream oss;
    oss << "Pricing Manager Statistics:" << std::endl;
    oss << "  Uptime: " << std::fixed << std::setprecision(2) << uptimeSeconds << " seconds" << std::endl;
    oss << "  Total requests: " << stats_.totalRequests.load() << std::endl;
    oss << "  Total batch requests: " << stats_.totalBatchRequests.load() << std::endl;
    oss << "  Completed requests: " << stats_.totalCompletedRequests.load() << std::endl;
    oss << "  Failed requests: " << stats_.totalFailedRequests.load() << std::endl;
    oss << "  Current queue depth: " << stats_.currentQueueDepth.load() << std::endl;
    oss << "  Maximum queue depth: " << stats_.maxQueueDepth.load() << std::endl;
    
    if (uptimeSeconds > 0) {
        double requestsPerSecond = stats_.totalRequests.load() / uptimeSeconds;
        oss << "  Request throughput: " << std::fixed << std::setprecision(2) 
            << requestsPerSecond << " requests/sec" << std::endl;
    }
    
    return oss.str();
}

int32_t PricingManager::pricingTask(const PricingRequest& request, uint64_t requestId) {
    // Get a pricer from the pool
    auto pricer = pricerPool_->getPricer();
    
    if (!pricer) {
        // No pricer available
        PricingResult result;
        result.requestId = requestId;
        result.instrumentId = request.instrumentId;
        result.statusCode = 3;  // No pricer available
        
        processResult(result, requestId);
        return -1;
    }
    
    try {
        // Execute the pricing
        PricingResult result = pricer->price(request);
        
        // Process the result
        processResult(result, requestId);
        
        // Return the pricer to the pool
        pricerPool_->returnPricer(pricer);
        
        return 0;
    } catch (const std::exception& e) {
        // Handle pricing error
        PricingResult result;
        result.requestId = requestId;
        result.instrumentId = request.instrumentId;
        result.statusCode = 1;  // Error
        
        processResult(result, requestId);
        
        // Return the pricer to the pool
        pricerPool_->returnPricer(pricer);
        
        return -1;
    }
}

int32_t PricingManager::batchPricingTask(std::vector<PricingRequest> requests, uint64_t batchId) {
    // Get the batch request
    std::shared_ptr<BatchRequest> batchRequest;
    {
        std::lock_guard<std::mutex> lock(managerMutex_);
        auto it = batchRequests_.find(batchId);
        if (it == batchRequests_.end()) {
            return -1;
        }
        
        batchRequest = it->second;
    }
    
    // Get a pricer from the pool
    auto pricer = pricerPool_->getPricer();
    
    if (!pricer) {
        // No pricer available, fail all requests
        for (const auto& request : requests) {
            PricingResult result;
            result.requestId = request.requestId;
            result.instrumentId = request.instrumentId;
            result.statusCode = 3;  // No pricer available
            
            processResult(result, request.requestId);
        }
        
        return -1;
    }
    
    // Execute all requests
    for (size_t i = 0; i < requests.size(); ++i) {
        try {
            PricingResult result = pricer->price(requests[i]);
            
            // Store the result in the batch
            size_t batchIndex = std::distance(
                batchRequest->requestIds.begin(),
                std::find(batchRequest->requestIds.begin(), batchRequest->requestIds.end(), requests[i].requestId)
            );
            
            if (batchIndex < batchRequest->results.size()) {
                batchRequest->results[batchIndex] = result;
            }
            
            // Process the result
            processResult(result, requests[i].requestId);
            
            // Update completion count
            batchRequest->completedCount.fetch_add(1);
        } catch (const std::exception& e) {
            // Handle pricing error
            PricingResult result;
            result.requestId = requests[i].requestId;
            result.instrumentId = requests[i].instrumentId;
            result.statusCode = 1;  // Error
            
            // Store the result in the batch
            size_t batchIndex = std::distance(
                batchRequest->requestIds.begin(),
                std::find(batchRequest->requestIds.begin(), batchRequest->requestIds.end(), requests[i].requestId)
            );
            
            if (batchIndex < batchRequest->results.size()) {
                batchRequest->results[batchIndex] = result;
            }
            
            processResult(result, requests[i].requestId);
            
            // Update completion count
            batchRequest->completedCount.fetch_add(1);
        }
    }
    
    // Return the pricer to the pool
    pricerPool_->returnPricer(pricer);
    
    // Check if the batch is complete
    checkBatchCompletion(batchId);
    
    return 0;
}

void PricingManager::processResult(const PricingResult& result, uint64_t requestId) {
    // Record in the journal if available
    if (journal_) {
        journal_->recordResult(result);
    }
    
    // Update statistics
    stats_.currentQueueDepth.fetch_sub(1);
    if (result.statusCode == 0) {
        stats_.totalCompletedRequests.fetch_add(1);
    } else {
        stats_.totalFailedRequests.fetch_add(1);
    }
    
    // Store the result and invoke the callback
    std::function<void(const PricingResult&)> callback;
    
    {
        std::lock_guard<std::mutex> lock(managerMutex_);
        
        // Store the result
        resultCache_[requestId] = result;
        
        // Get the callback if any
        auto it = callbacks_.find(requestId);
        if (it != callbacks_.end()) {
            callback = it->second;
            callbacks_.erase(it);
        }
    }
    
    // Invoke the callback outside the lock
    if (callback) {
        try {
            callback(result);
        } catch (const std::exception& e) {
            std::cerr << "Error in callback for request " << requestId 
                      << ": " << e.what() << std::endl;
        }
    }
}

void PricingManager::checkBatchCompletion(uint64_t batchId) {
    std::shared_ptr<BatchRequest> batchRequest;
    std::function<void(const std::vector<PricingResult>&)> callback;
    
    {
        std::lock_guard<std::mutex> lock(managerMutex_);
        
        auto it = batchRequests_.find(batchId);
        if (it == batchRequests_.end()) {
            return;
        }
        
        batchRequest = it->second;
        
        // Check if all requests in the batch are completed
        if (batchRequest->completedCount.load() >= batchRequest->requestIds.size()) {
            callback = batchRequest->callback;
            batchRequests_.erase(it);
        }
    }
    
    // Invoke the callback outside the lock
    if (callback) {
        try {
            callback(batchRequest->results);
        } catch (const std::exception& e) {
            std::cerr << "Error in batch callback for batch " << batchId 
                      << ": " << e.what() << std::endl;
        }
    }
}

} 
} 