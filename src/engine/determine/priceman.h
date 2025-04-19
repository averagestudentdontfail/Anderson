// priceman.h
// Pricing Manager for the deterministic execution framework

#ifndef ENGINE_DETERMINE_PRICEMAN_H
#define ENGINE_DETERMINE_PRICEMAN_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <functional>
#include "shmem.h"
#include "schedman.h"
#include "jourman.h"

namespace engine {
namespace determine {

// Forward declarations
class ALOEngine;
class DeterministicPricer;
class PricingManager;

/**
 * @brief Core pricing engine with deterministic execution
 */
class DeterministicPricer {
public:
    /**
     * @brief Create a deterministic pricer
     * @param scheme ALO scheme to use
     */
    explicit DeterministicPricer(int scheme = 0);
    
    /**
     * @brief Destructor
     */
    ~DeterministicPricer();
    
    /**
     * @brief Price an American option
     * @param request Pricing request
     * @return Pricing result
     */
    PricingResult price(const PricingRequest& request);
    
    /**
     * @brief Batch price multiple options
     * @param requests Vector of pricing requests
     * @return Vector of pricing results
     */
    std::vector<PricingResult> batchPrice(const std::vector<PricingRequest>& requests);
    
    /**
     * @brief Configure the pricer
     * @param scheme ALO scheme to use
     * @param useVectorization Whether to use SIMD vectorization
     * @param fixedExecutionTime Whether to ensure fixed execution time
     */
    void configure(int scheme, bool useVectorization = true, bool fixedExecutionTime = true);
    
    /**
     * @brief Pre-warm the pricer
     * @param iterations Number of warm-up iterations
     */
    void preWarm(int iterations = 1000);
    
    /**
     * @brief Reset the pricer cache
     */
    void resetCache();

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    /**
     * @brief Calculate option Greeks using finite differences
     * @param request Pricing request
     * @param result Pricing result to populate
     */
    void calculateGreeks(const PricingRequest& request, PricingResult& result);
};

/**
 * @brief Pool of pricing engines for deterministic execution
 */
class PricerPool {
public:
    /**
     * @brief Create a pricer pool
     * @param poolSize Number of pricers in the pool
     * @param scheme ALO scheme to use
     */
    PricerPool(size_t poolSize = 4, int scheme = 0);
    
    /**
     * @brief Get a pricer from the pool
     * @return Shared pointer to a pricer
     */
    std::shared_ptr<DeterministicPricer> getPricer();
    
    /**
     * @brief Return a pricer to the pool
     * @param pricer Pricer to return
     */
    void returnPricer(std::shared_ptr<DeterministicPricer> pricer);
    
    /**
     * @brief Pre-warm all pricers in the pool
     * @param iterations Number of warm-up iterations
     */
    void preWarmAll(int iterations = 1000);
    
    /**
     * @brief Reset cache for all pricers in the pool
     */
    void resetAllCaches();

private:
    std::vector<std::shared_ptr<DeterministicPricer>> pricers_;
    std::atomic<size_t> nextPricerIndex_{0};
    size_t poolSize_;
};

/**
 * @brief Manager for deterministic pricing operations
 */
class PricingManager {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the pricing manager
     */
    static PricingManager& getInstance();
    
    /**
     * @brief Initialize the pricing manager
     * @param numPricers Number of pricers to create
     * @param scheme ALO scheme to use
     * @param useScheduler Whether to use the scheduler for execution
     */
    void initialize(size_t numPricers = 4, int scheme = 0, bool useScheduler = true);
    
    /**
     * @brief Start the pricing manager
     * @return True if started successfully
     */
    bool start();
    
    /**
     * @brief Stop the pricing manager
     */
    void stop();
    
    /**
     * @brief Submit a pricing request
     * @param request Pricing request
     * @param callback Callback function for the result
     * @return Request ID
     */
    uint64_t submitRequest(const PricingRequest& request, 
                         std::function<void(const PricingResult&)> callback = nullptr);
    
    /**
     * @brief Submit a batch of pricing requests
     * @param requests Vector of pricing requests
     * @param callback Callback function for the results
     * @return Vector of request IDs
     */
    std::vector<uint64_t> submitBatchRequest(
        const std::vector<PricingRequest>& requests,
        std::function<void(const std::vector<PricingResult>&)> callback = nullptr);
    
    /**
     * @brief Get a pricing result
     * @param requestId Request ID
     * @return Pricing result
     */
    PricingResult getResult(uint64_t requestId);
    
    /**
     * @brief Wait for a result
     * @param requestId Request ID
     * @param timeoutMs Timeout in milliseconds (0 = wait forever)
     * @return True if the result is available
     */
    bool waitForResult(uint64_t requestId, uint64_t timeoutMs = 0);
    
    /**
     * @brief Process a market data update
     * @param update Market data update
     */
    void processMarketUpdate(const MarketUpdate& update);
    
    /**
     * @brief Set the journal for event recording
     * @param journal Journal to use
     */
    void setJournal(std::shared_ptr<EventJournal> journal);
    
    /**
     * @brief Get performance statistics
     * @return String containing performance statistics
     */
    std::string getStatistics() const;

private:
    PricingManager() = default;
    ~PricingManager() = default;
    
    // Disable copy and move
    PricingManager(const PricingManager&) = delete;
    PricingManager& operator=(const PricingManager&) = delete;
    PricingManager(PricingManager&&) = delete;
    PricingManager& operator=(PricingManager&&) = delete;
    
    // Configuration
    bool initialized_ = false;
    bool running_ = false;
    bool useScheduler_ = true;
    
    // Pricing resources
    std::unique_ptr<PricerPool> pricerPool_;
    std::shared_ptr<SchedulerManager> schedulerManager_;
    std::shared_ptr<EventJournal> journal_;
    
    // Request tracking
    std::atomic<uint64_t> nextRequestId_{1};
    std::unordered_map<uint64_t, PricingResult> resultCache_;
    std::unordered_map<uint64_t, std::function<void(const PricingResult&)>> callbacks_;
    std::mutex managerMutex_;
    
    // Batch request tracking
    struct BatchRequest {
        std::vector<uint64_t> requestIds;
        std::vector<PricingResult> results;
        std::atomic<size_t> completedCount{0};
        std::function<void(const std::vector<PricingResult>&)> callback;
    };
    std::unordered_map<uint64_t, std::shared_ptr<BatchRequest>> batchRequests_;
    
    // Performance tracking
    struct PerfStats {
        std::atomic<uint64_t> totalRequests{0};
        std::atomic<uint64_t> totalBatchRequests{0};
        std::atomic<uint64_t> totalCompletedRequests{0};
        std::atomic<uint64_t> totalFailedRequests{0};
        std::atomic<uint64_t> maxQueueDepth{0};
        std::atomic<uint64_t> currentQueueDepth{0};
        HighResolutionTimer uptimeTimer;
    };
    PerfStats stats_;
    
    /**
     * @brief Task to price a single request
     * @param request Request to price
     * @param requestId Request ID
     * @return Result code
     */
    int32_t pricingTask(const PricingRequest& request, uint64_t requestId);
    
    /**
     * @brief Task to price a batch of requests
     * @param requests Requests to price
     * @param batchId Batch ID
     * @return Result code
     */
    int32_t batchPricingTask(std::vector<PricingRequest> requests, uint64_t batchId);
    
    /**
     * @brief Process a completed result
     * @param result Pricing result
     * @param requestId Request ID
     */
    void processResult(const PricingResult& result, uint64_t requestId);
    
    /**
     * @brief Check if a batch request is complete
     * @param batchId Batch ID
     */
    void checkBatchCompletion(uint64_t batchId);
};

}
} 

#endif 